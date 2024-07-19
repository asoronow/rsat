# A tool for checking the efficacy and accuracy of the detection algorithm
import numpy as np
import cv2
import argparse
from scipy.ndimage import  binary_dilation
from skimage.morphology import skeletonize, remove_small_objects

def detect_axons(gray_image, label_image, margin=5):
    # Invert and binarize the label image
    label_image = cv2.bitwise_not(label_image)
    label_image = cv2.threshold(label_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply Gaussian blur to gradients
    gaussA = cv2.GaussianBlur(gray_image, (7, 7), 0.5)
    gaussB = cv2.GaussianBlur(gray_image, (7, 7), 20)
    
    # Difference of Gaussian (DoG)
    dof = gaussB - gaussA
    dof = cv2.normalize(dof, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold the DoG result
    edges = cv2.threshold(dof, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = binary_dilation(edges, structure=np.ones((2, 2)), iterations=1)

    # Remove small objects
    edges = remove_small_objects(edges, min_size=50)
    edges = edges.astype(np.uint8) * 255
    # Compute Dice coefficient with margin of error
    label_image_binary = label_image // 255
    edges_binary = edges // 255
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    
    # Dilate the binary images to include margin of error
    kernel = np.ones((margin, margin), np.uint8)
    label_dilated = cv2.dilate(label_image_binary, kernel)
    
    # Overlay the skeleton on the label image in red
    color_label_image = cv2.cvtColor(label_image, cv2.COLOR_GRAY2BGR)
    color_label_image[:, :, 0] = label_dilated * 255  # Make label blue
    color_label_image[:, :, 1] = 0  # Remove green channel
    color_label_image[:, :, 2] = 0  # Remove red channel
    
    red_overlay = np.zeros_like(color_label_image)
    red_overlay[:, :, 2] = edges
    overlayed_image = cv2.addWeighted(color_label_image, 0.5, red_overlay, 1.0, 0)
    
    intersection = np.sum(label_dilated * edges_binary)
    union = np.sum(label_dilated) + np.sum(edges_binary) - intersection
    dice_coefficient = 2 * intersection / union
    
    # Display the overlayed image and print Dice coefficient
    cv2.imshow("Skeletonized Axons Overlay", overlayed_image)
    print(f"Dice Coefficient with {margin}px margin: {dice_coefficient:.4f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect axons in an image.")
    parser.add_argument("--input", type=str, help="The input image file.")
    parser.add_argument("--label", type=str, help="The label of the image.")
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(args.label, cv2.IMREAD_GRAYSCALE)
    detect_axons(image, label)