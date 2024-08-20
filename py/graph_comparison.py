from train_seg import get_axon_mask
import argparse
import cv2
import os
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_dilation, binary_erosion, disk
from sklearn.metrics import precision_score, recall_score

def calculate_metrics(rsat_mask, human_masks):
    # Flatten the arrays to 1D
    rsat_mask_flat = (rsat_mask.flatten() / 255).astype(np.uint8)
    human_mask1_flat = (human_masks[0].flatten() / 255).astype(np.uint8)
    human_mask2_flat = (human_masks[1].flatten() / 255).astype(np.uint8)

    # Calculate precision and recall
    precision_detector_vs_human1 = precision_score(human_mask1_flat, rsat_mask_flat, average="weighted")
    precision_detector_vs_human2 = precision_score(human_mask2_flat, rsat_mask_flat, average="weighted")
    precision_human1_vs_human2 = precision_score(human_mask1_flat, human_mask2_flat, average="weighted")

    recall_detector_vs_human1 = recall_score(human_mask1_flat, rsat_mask_flat, average="weighted")
    recall_detector_vs_human2 = recall_score(human_mask2_flat, rsat_mask_flat, average="weighted")
    recall_human1_vs_human2 = recall_score(human_mask1_flat, human_mask2_flat, average="weighted")

    return {
        "precision": [precision_detector_vs_human1, precision_detector_vs_human2, precision_human1_vs_human2],
        "recall": [recall_detector_vs_human1, recall_detector_vs_human2, recall_human1_vs_human2]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare graphs.")
    parser.add_argument("--input", type=str, help="The input image.")
    parser.add_argument("--human_masks", type=str, help="The masks of the input images.")
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    files = os.listdir(args.human_masks)
    human_masks = []
    for file in files:
        mask = cv2.imread(os.path.join(args.human_masks, file), cv2.IMREAD_GRAYSCALE)
        mask = binary_dilation(mask, disk(2)).astype(np.uint8) * 255
        human_masks.append(mask)

    rsat_mask = get_axon_mask(image)
    rsat_mask = binary_erosion(rsat_mask, disk(2)).astype(np.uint8) * 255

    eq_image = equalize_adapthist(image, clip_limit=0.05)

    # Create a color overlay for the masks
    human_mask1_color = np.zeros((human_masks[0].shape[0], human_masks[0].shape[1], 3), dtype=np.uint8)
    human_mask2_color = np.zeros((human_masks[1].shape[0], human_masks[1].shape[1], 3), dtype=np.uint8)
    rsat_mask_color = np.zeros((rsat_mask.shape[0], rsat_mask.shape[1], 3), dtype=np.uint8)

    human_mask1_color[:, :, 0] = human_masks[0]  # Red channel
    human_mask2_color[:, :, 1] = human_masks[1]  # Green channel
    rsat_mask_color[:, :, 2] = rsat_mask  # Blue channel

    combined_human_masks = cv2.addWeighted(human_mask1_color, 0.5, human_mask2_color, 0.5, 0)
    combined_all_masks = cv2.addWeighted(combined_human_masks, 0.5, rsat_mask_color, 0.5, 0)

    metrics = calculate_metrics(rsat_mask, human_masks)

    # Create a GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 3, figure=fig)

    # Plotting the images in a 2x3 grid
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(eq_image, cmap="gray")
    ax1.set_title("Equalized Image")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rsat_mask_color)
    ax2.set_title("RSAT Detection")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(combined_human_masks)
    ax3.set_title("Rater Masks Overlaid")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(human_mask1_color)
    ax4.set_title("Rater Mask 1")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(human_mask2_color)
    ax5.set_title("Rater Mask 2")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(combined_all_masks)
    ax6.set_title("All Masks Overlay")

    # Precision bar plot
    ax7 = fig.add_subplot(gs[2, 0])  # Span the first two columns
    bars = ax7.bar(
        ["RSAT vs Rater 1", "RSAT vs Rater 2", "Rater 1 vs Rater 2"],
        metrics["precision"],
        color=["blue", "orange", "green"]
    )
    ax7.set_title("Precision")
    ax7.set_ylim([0.95, 1])
    ax7.tick_params(axis='x', rotation=90)

    # Add the precision values above each bar
    for bar in bars:
        yval = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.3f}', ha='center', va='bottom')

    # Recall bar plot
    ax8 = fig.add_subplot(gs[2, 1])  # Span the middle and last columns
    bars = ax8.bar(
        ["RSAT vs Rater 1", "RSAT vs Rater 2", "Rater 1 vs Rater 2"],
        metrics["recall"],
        color=["blue", "orange", "green"]
    )
    ax8.set_title("Recall")
    ax8.set_ylim([0.95, 1])
    ax8.tick_params(axis='x', rotation=90)

    # Add the recall values above each bar
    for bar in bars:
        yval = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.3f}', ha='center', va='bottom')

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["svg.fonttype"] = "none"

    # Remove axes from image plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("./graph_comparison.svg", bbox_inches="tight", dpi=300)
