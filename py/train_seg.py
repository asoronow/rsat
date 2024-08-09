from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import SegformerImageProcessor
from skimage.exposure import equalize_adapthist
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import evaluate
from skimage.morphology import binary_dilation, skeletonize, binary_erosion, binary_closing, disk
from torch.nn import functional as F
import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import cv2

def refine_mask(mask):
    cleaned_mask = binary_closing(mask, disk(3)).astype(np.uint8)
    cleaned_mask = skeletonize(cleaned_mask).astype(np.uint8)
    cleaned_mask = binary_dilation(cleaned_mask, disk(1)).astype(np.uint8)
    # cleaned_mask = skeletonize(cleaned_mask).astype(np.uint8)
    
    return cleaned_mask

def get_axon_mask(image, model_path="./best.ckpt"):
    assert os.path.exists(model_path), f"Model path {model_path} does not exist. Did you download the model?"
    segformer_finetuner = SegformerFinetuner.load_from_checkpoint(
    model_path,
    id2label={
        0: "background",
        1: "axon",
        },
    )

    image = equalize_adapthist(np.array(image), clip_limit=0.0005)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_array = np.array(image)
    
    tiles = []
    tile_indices = []

    step_size = 256
    tile_size = 256

    height, width = image_array.shape[:2]
    original_shape = image_array.shape[:2]

    # Calculate padding if the dimensions are not divisible by the tile size
    height_pad = (tile_size - (height % tile_size)) % tile_size
    width_pad = (tile_size - (width % tile_size)) % tile_size
    
    # Apply padding to the image
    image_array = cv2.copyMakeBorder(image_array, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=0)
    
    # Update the dimensions after padding
    height, width = image_array.shape[:2]
    
    for i in range(0, height, step_size): 
        for j in range(0, width, step_size):
                # Check if the tile is out of bounds
                if i+tile_size > height or j+tile_size > width:
                    continue

                tile_indices.append((i, j))
                tile = image_array[i:i+tile_size, j:j+tile_size]
                tiles.append(tile)


    tiles = np.array(tiles)

    if len(tiles) > 0:
        
        mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        
        feature_extractor = SegformerImageProcessor()
        
        with torch.no_grad():
            for (i, j), tile in zip(tile_indices, tiles):
                tile = Image.fromarray(tile)
                encoded_inputs = feature_extractor(tile, return_tensors="pt")
                pixel_values = encoded_inputs["pixel_values"].to("mps")
                outputs = segformer_finetuner(pixel_values)
                logits = outputs[0]
                upsampled_logits = F.interpolate(
                    logits, 
                    size=(tile_size, tile_size), 
                    mode="bilinear", 
                    align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1).cpu().numpy().squeeze()
                mask[i:i+tile_size, j:j+tile_size] = predicted
        
        mask = refine_mask(mask)

        pad_x = (mask.shape[1] - original_shape[1])
        pad_y = (mask.shape[0] - original_shape[0])

        if pad_x > 0:
            mask = mask[:,:-pad_x]
        if pad_y > 0:
            mask = mask[:-pad_y,:]

    return mask
        

def dice_loss(pred, target, smooth=1.0):
    # For multi-class segmentation, apply softmax and keep the probabilities
    if pred.size(1) > 1:
        pred = F.softmax(pred, dim=1)
        # Assuming we're interested in the second class (e.g., axon)
        pred = pred[:, 1, :, :]  # Select the class of interest
    else:
        pred = torch.sigmoid(pred)  # For binary classification, use sigmoid
    
    # Flatten the tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()

    # If the target mask is uint8 and uses 255 for the masked pixels, convert to binary
    if target.max() > 1:
        target = target / 255.0  # Normalize to 0 and 1

    # Compute Dice Coefficient
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice
    
class SegformerFinetuner(pl.LightningModule):
    def __init__(self, id2label, train_dataloader=None, class_weights=None, val_dataloader=None, test_dataloader=None, metrics_interval=10, dice_weight=0.25):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.validation_step_outputs = []
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        self.num_classes = len(self.id2label)
        self.label2id = {v:k for k,v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b4",
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")
        
    def forward(self, images, masks=None):
        if masks is None:
            return self.model(pixel_values=images)
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
    
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )


        predicted = upsampled_logits.argmax(dim=1)        
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )


        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

            self.log_dict(metrics, prog_bar=True)

            return(metrics)

        return({'loss': loss})
        
    # Additional NaN checks for validation and test steps
    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)

        self.validation_step_outputs.append(loss)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )

        
        self.log("val_loss", loss)

        return({'val_loss': loss})

    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255,
            reduce_labels=False,
        )
        
        avg_val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        val_mean_iou = metrics.get("mean_iou", float('0.0'))
        val_mean_accuracy = metrics.get("mean_accuracy", float('0.0'))
        self.validation_step_outputs = []

        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou, "val_mean_accuracy": val_mean_accuracy}

        self.log_dict(metrics, prog_bar=True)
        return(metrics)
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        self.root_dir = Path(root_dir).expanduser()
        self.feature_extractor = feature_extractor
        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i,l in enumerate(fid) if i !=0]
        self.id2label = {x[0].strip():x[1].strip() for x in data}
        
        image_file_names = [f for f in os.listdir(os.path.join(self.root_dir, 'images')) if '.png' in f]
        mask_file_names = [f for f in os.listdir(os.path.join(self.root_dir, 'labels')) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.root_dir, "images", self.images[idx]))
        image = equalize_adapthist(np.array(image), clip_limit=0.03)
        # cast back to int8
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')
        segmentation_map = Image.open(os.path.join(self.root_dir, "labels", self.masks[idx]))

        segmentation_map = np.array(segmentation_map)
        # Convert all 255 to 1
        segmentation_map[segmentation_map == 255] = 1

        # Dilate
        segmentation_map = binary_dilation(segmentation_map, disk(5))

        # Conver to uint8
        segmentation_map = segmentation_map.astype(np.uint8)
        segmentation_map = Image.fromarray(segmentation_map)

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model.")

    parser.add_argument("-t","--train", type=bool, help="Train the model.", default=False)
    parser.add_argument("-e", "--test", type=bool, help="Test the model.", default=False)
    parser.add_argument("-c", "--check", type=str, help="Path to an image to check with the model.", default=False)
    parser.add_argument("-cm", "--check_mask", type=str, help="Path to a mask to verify the checked image prediction.", default=False)
    parser.add_argument("-m", "--model_path", type=str, help="The model path.", default="./best.ckpt")

    args = parser.parse_args()

    feature_extractor = SegformerImageProcessor()
    feature_extractor.size = 256

    train_dataset = SemanticSegmentationDataset("~/Projects/rsat-segment/datasets/train/", feature_extractor)
    val_dataset = SemanticSegmentationDataset("~/Projects/rsat-segment/datasets/validate/", feature_extractor)

    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, persistent_workers=True)
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00,
        patience=30, 
        verbose=False, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, save_last=True, monitor="val_loss")

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=500,
        val_check_interval=len(train_dataloader),
        log_every_n_steps=2,
    )

    if args.train:
        segformer_finetuner = SegformerFinetuner(
            id2label=train_dataset.id2label, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            metrics_interval=2,
            class_weights=[0.05, 0.95],
        )

        trainer.fit(segformer_finetuner)
    elif args.test:
        segformer_finetuner = SegformerFinetuner.load_from_checkpoint(
            "best.ckpt",
            id2label=train_dataset.id2label,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
            metrics_interval=2,
            class_weights=[0.05, 0.95],
        )

        color_map = {
            0:(0,0,255),
            1:(255,0,0),
        }

        def prediction_to_vis(prediction):
            vis_shape = prediction.shape + (3,)
            vis = np.zeros(vis_shape)
            for i,c in color_map.items():
                vis[prediction == i] = color_map[i]
            return Image.fromarray(vis.astype(np.uint8))

        for batch in val_dataloader:
            images, masks = batch['pixel_values'], batch['labels']

            # ensure all the values are on the same device
            images, masks = images.to("mps"), masks.to("mps")

            outputs = segformer_finetuner.model(images, masks)
                
            loss, logits = outputs[0], outputs[1]

            upsampled_logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )

            # visualize probabilities
            # fig, ax = plt.subplots(1,2)
            # # get min and max values for plotting
            # vmin, vmax = upsampled_logits.min(), upsampled_logits.max()
            # for i in range(upsampled_logits.shape[1]):
            #     ax[0].imshow(upsampled_logits[0,i,:,:].detach().cpu().numpy(), cmap='hot', vmin=vmin, vmax=vmax)
            #     ax[0].set_title(f"Class {i+1}")
            #     ax[0].axis('off')
            #     ax[0] = plt.gca()
            # ax[0].set_xticks([])
            # ax[0].set_yticks([])
            
            # plt.tight_layout()
            # plt.show()

            predicted = upsampled_logits.argmax(dim=1).cpu().numpy()
            masks = masks.cpu().numpy()


            f, axarr = plt.subplots(predicted.shape[0],2)
            for i in range(predicted.shape[0]):
                axarr[i,0].imshow(prediction_to_vis(predicted[i,:,:]))
                axarr[i,1].imshow(prediction_to_vis(masks[i,:,:]))

            plt.show()
    elif args.check:
        segformer_finetuner = SegformerFinetuner.load_from_checkpoint(
        args.model_path,
        id2label={
            0: "background",
            1: "axon",
            },
        )
        
        image = Image.open(args.check).convert('RGB')
        image = equalize_adapthist(np.array(image), clip_limit=0.03)
        image = (image * 255).astype(np.uint8)
        image_array = np.array(image)
        
        tiles = []
        tile_indices = []

        step_size = 128
        tile_size = 256

        height, width = image_array.shape[:2]
        # Pad the image so its divisible by the tile size
        if height % tile_size != 0 or width % tile_size != 0:
            height_pad = tile_size - (height % tile_size)
            width_pad = tile_size - (width % tile_size)
            image_array = cv2.copyMakeBorder(image_array, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=0)
            height, width = image_array.shape[:2]

        for i in range(0, height, step_size): 
            for j in range(0, width, step_size):
                # Check if the tile is out of bounds
                if i+tile_size > height or j+tile_size > width:
                    continue

                tile_indices.append((i, j))
                tile = image_array[i:i+tile_size, j:j+tile_size]
                tiles.append(tile)


        tiles = np.array(tiles)

        print(f"Tiles: {len(tiles)}")

    if len(tiles) > 0:
        
        mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
        
        feature_extractor = SegformerImageProcessor()
        
        with torch.no_grad():
            for (i, j), tile in zip(tile_indices, tiles):
                tile = Image.fromarray(tile)
                encoded_inputs = feature_extractor(tile, return_tensors="pt")
                pixel_values = encoded_inputs["pixel_values"].to("mps")
                outputs = segformer_finetuner(pixel_values)
                logits = outputs[0]
                upsampled_logits = F.interpolate(
                    logits, 
                    size=(tile_size, tile_size), 
                    mode="bilinear", 
                    align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1).cpu().numpy().squeeze()
                mask[i:i+tile_size, j:j+tile_size] = predicted
        
        mask = refine_mask(mask)

        mask *= 255
        if args.check_mask:
            # Load and preprocess the real mask
            real_mask = Image.open(args.check_mask).convert('L')
            real_mask = np.array(real_mask)
            real_mask = real_mask.astype(np.uint8)
            original_shape = real_mask.shape
            # pad real mask to the same size as the predicted mask
            real_mask = cv2.copyMakeBorder(real_mask, 0, mask.shape[0] - real_mask.shape[0], 0, mask.shape[1] - real_mask.shape[1], cv2.BORDER_CONSTANT, value=0)

            # Convert masks to boolean arrays
            real_mask_bool = (real_mask > 0).astype(bool)
            mask_bool = (mask > 0).astype(bool)

            # Crop bottom 200 px
            real_mask_bool = real_mask_bool
            mask_bool = mask_bool

            real_mask_bool = refine_mask(real_mask_bool)

            # Compute true positive and false positive rate
            tp = np.logical_and(real_mask_bool, mask_bool).sum()
            fp = np.logical_and(real_mask_bool, np.logical_not(mask_bool)).sum()
            fn = np.logical_and(np.logical_not(real_mask_bool), mask_bool).sum()
            tn = np.logical_and(np.logical_not(real_mask_bool), np.logical_not(mask_bool)).sum()

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            print(f"True Positive Rate: {tpr}")
            print(f"False Positive Rate: {fpr}")

            # F1 score
            f1 = 2 * (tpr * fpr) / (tpr + fpr)
            print(f"F1 Score: {f1}")

            # MSE
            mse = np.mean((real_mask_bool.astype(float) - mask_bool.astype(float)) ** 2)
            print(f"MSE: {mse}")

            # Precision
            precision = tp / (tp + fp)
            print(f"Precision: {precision}")

            # Recall
            recall = tp / (tp + fn)
            print(f"Recall: {recall}")

            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            print(f"Accuracy: {accuracy}")

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        # calculate padding
        pad_x = (real_mask_bool.shape[1] - original_shape[1])
        pad_y = (real_mask_bool.shape[0] - original_shape[0])
        print(f"Padding: {pad_x}, {pad_y}")

        if pad_x > 0:
            mask_bool = mask_bool[:,:-pad_x]
            real_mask_bool = real_mask_bool[:,:-pad_x]
            image_array = image_array[:,:-pad_x,:]
        if pad_y > 0:
            mask_bool = mask_bool[:-pad_y,:]
            real_mask_bool = real_mask_bool[:-pad_y,:]
            image_array = image_array[:-pad_y,:,:]
       

        image_array = (equalize_adapthist(np.array(image_array), clip_limit=0.03) * 255).astype(np.uint8)
        panel_one = image_array.copy()
        panel_two = image_array.copy()
        panel_three = image_array.copy()

        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        ax[3].axis("off")

        ax[0].set_title("Input Image")
        ax[0].imshow(image_array)
        panel_one[real_mask_bool > 0] = (255, 0, 0)

        ax[1].set_title("Real Mask")
        ax[1].imshow(panel_one)
        panel_two[mask_bool > 0] = (0, 255, 0)

        ax[2].set_title("Predicted Mask")
        ax[2].imshow(panel_two)
        panel_three[mask_bool > 0] = (0, 255, 0)
        panel_three[real_mask_bool > 0 & mask_bool] = (255, 0, 255)

        ax[3].set_title("Overlap Result")
        ax[3].imshow(panel_three)
        
        # Save as svg
        fig.savefig("segmentation_result.svg")

        # save mask
        mask = Image.fromarray(mask) 
        mask.save("segmentation_result.png")