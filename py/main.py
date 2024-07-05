import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
import cv2
from scipy.ndimage import binary_dilation, convolve
def correct_edges(outside_points, binary_image, max_distance=20):
    """
    Fast version of correct_edges function.

    Parameters:
        outside_points (list): A list of points that are outside the ROI.
        binary_image (np.array): A binary image of the ROI, where 1 is a detected point and 0 is not.
        max_distance (int): The maximum distance from outside points to consider.

    Returns:
        np.array: The cleaned binary image.
    """
    # Create a mask from outside points
    mask = np.zeros_like(binary_image, dtype=np.uint8)
    mask[tuple(zip(*outside_points))] = 1

    # Dilate the mask
    dilated_mask = binary_dilation(mask, iterations=max_distance)

    # Remove edge points directly from the binary image
    binary_image[dilated_mask == 1] = 0

    return binary_image 

class ROI:
    """
    ROI Object

    """

    def __init__(self, name, intensity, filename, coverage=None):
        self.name = name
        self.intensity = intensity
        self.filename = filename
        self.coverage = coverage
        self.area = self.total_area()
        self.mask = None

    def mean(self):
        # find the mean intensity value
        values = []
        for i, j in self.intensity.keys():
            values.append(self.intensity[(i, j)])

        return np.mean(values)

    def adjust_to_coverage(self):
        """Multiplies all intensity values by the coverage value"""
        if self.coverage is None:
            print(f"Coverage value not set. Cannot adjust {self.filename} to coverage.")
            return

        for i, j in self.intensity.keys():
            self.intensity[(i, j)] = int(
                np.floor(float(self.intensity[(i, j)]) * self.coverage)
            )

    def bounds(self):
        # return the bounds of the ROI
        verts = list(self.intensity.keys())
        bounds = (
            np.min(verts[:, 0]),
            np.max(verts[:, 0]),
            np.min(verts[:, 1]),
            np.max(verts[:, 1]),
        )
        # now get the width and height of the ROI
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        return bounds, width, height

    def total_area(self):
        # return the total area of the ROI
        return len(self.intensity)
    
    def create_axon_mask(self, tuned_params):
        """Identify axons in the ROI"""
        verts = list(self.intensity.keys())
        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Filestem
        stem = Path(self.filename).stem

        # params
        sigma = tuned_params["sigma"]
        contrast = tuned_params["contrast"]
        brightness = tuned_params["brightness"]

        # Create an image of the ROI
        image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        mask = np.zeros_like(image)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            image[y, x] = self.intensity[vert]
            mask[y, x] = 1
            
        # Apply contrast and brightness adjustments
        image = np.clip(tuned_params["contrast"] * image + tuned_params["brightness"], 0, 255).astype(np.uint8)

        # Edge detection
        image = (image / np.max(image) * 255).astype(np.uint8)

        gauss = gaussian(image, sigma=sigma)
        horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
        vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
        edges = np.abs(convolve(gauss, horizontal, mode="constant")) + np.abs(
            convolve(gauss, vertical, mode="constant")
        )
        thresh = threshold_otsu(edges)
        binary = edges > thresh
        outside_points = np.argwhere(mask == 0)
        binary = correct_edges(outside_points, binary, max_distance=20)
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        colored_image[binary == 1] = [0, 0, 255]

        plt.rcParams["font.size"] = 12
        # export text
        plt.rcParams["svg.fonttype"] = "none"
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(edges, cmap=plt.cm.gray)
        ax[1].set_title("Edge Detection")
        ax[1].axis("off")

        ax[2].imshow(binary * 255, cmap=plt.cm.gray)
        ax[2].set_title("Thresholded")
        ax[2].axis("off")

        # make a folder to save the image
        output_folder = Path(f"./screening/{stem[:4]}/{self.name}").resolve()
        output_folder.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(f"{str(output_folder)}/{stem}_thresholded.png", dpi=600)
        plt.savefig(f"{str(output_folder)}/{stem}_thresholded.svg", format="svg", dpi=600)
        plt.close()

        x_range = max_x - min_x
        y_range = max_y - min_y
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{str(output_folder)}/{stem}_axon_mask.png", colored_image)

        # Normalize the coordinates to fit into a 101x101 grid
        normalized_mask = np.zeros((101, 101), dtype=np.uint8)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            norm_y, norm_x = int(y / y_range * 100), int(x / x_range * 100)
            if 0 < norm_y < 100 and 0 < norm_x < 100:
                normalized_mask[norm_y, norm_x] += 1 if (binary[y, x] == 1) else 0
                image[y, x] = (0, 0, 255) if (binary[y, x] == 1) else (0, 255, 0)
        cv2.imwrite(f"{str(output_folder)}/{stem}_colorized.png", image)

        # dump intensity data to save memory
        self.intensity = None
        self.verts = None
        self.mask = normalized_mask


def load_roi_from_file(filename):
    """
    Load ROI from file.

    Parameters:
        filename (str): The filename of the ROI.

    Returns:
        ROI: The ROI object.
    """
    with open(filename, "rb") as f:
        package = pickle.load(f)
        try:
            intensity = package["roi"]
            name = package["name"]

            try:
                coverage = package["coverage"]
            except KeyError:
                coverage = None

            filename = filename
            roi = ROI(name, intensity, filename, coverage=coverage)
            return roi
        except:
            print("Error loading ROI: {}".format(filename))
            return None
        
def loadROI(path):
    """
    Load ROIs in as a generator or a single ROI.
    """
    toLoad = []
    if os.path.isfile(path):
        toLoad.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl"):
                    if "raw" not in file:
                        toLoad.append(os.path.join(root, file))
    else:
        print("Invalid path")
        return None

    # Load each ROI
    while len(toLoad) > 0:
        file = toLoad.pop()
        with open(file, "rb") as f:
            package = pickle.load(f)
            try:
                intensity = package["roi"]
                name = package["name"]

                try:
                    coverage = package["coverage"]
                except KeyError:
                    coverage = None

                filename = file
                roi = ROI(name, intensity, filename, coverage=coverage)
                yield roi
            except:
                print("Error loading ROI: {}".format(file))
                continue
