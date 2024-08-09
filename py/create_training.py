import os
import cv2
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data.")
    parser.add_argument("--images", type=str, help="The input image file.")
    parser.add_argument("--labels", type=str, help="The label of the image.")
    parser.add_argument("--output", type=str, help="The output directory.")
    args = parser.parse_args()

    data_pairs = [] # save tuples of image tile and label tile
    for image, label in zip(os.listdir(args.images), os.listdir(args.labels)):
        if not image.endswith(".png") or not label.endswith(".png"):
            continue
        image_path = os.path.join(args.images, image)
        label_path = os.path.join(args.labels, label)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        data_pairs.append((image, label))

    print(f"Found {len(data_pairs)} pairs of images and labels.")
    data_pairs_tiles = []
    for image, label in data_pairs:
        # split pairs in 256x256 tiles
        for i in range(0, image.shape[0], 256):
            for j in range(0, image.shape[1], 256):
                if i+256 > image.shape[0] or j+256 > image.shape[1]:
                    break
                image_tile = image[i:i+256, j:j+256]
                label_tile = label[i:i+256, j:j+256]
                data_pairs_tiles.append((image_tile, label_tile))

    

    images_folder = os.path.join(args.output, "images")
    labels_folder = os.path.join(args.output, "labels")
    for index, (image_tile, label_tile) in enumerate(data_pairs_tiles):
        # make directories if they don't exist
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        # save tiles
        cv2.imwrite(os.path.join(images_folder, f"{index}.png"), image_tile)
        cv2.imwrite(os.path.join(labels_folder, f"{index}.png"), label_tile)