import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import  binary_dilation
from skimage.exposure import equalize_adapthist
from train_seg import get_axon_mask
def correct_edges(outside_points, binary_image, iterations=5):
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
    dilated_mask = binary_dilation(mask, structure=np.ones((3, 3)), iterations=iterations)

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
        self.verts = list(intensity.keys())
        self.filename = filename
        self.coverage = coverage
        self.area = self.total_area()
        self.h2b_distribution = np.zeros(101)
        self.mask = None


    def calculate_h2b_distribution(self, h2b_centers):
        """Calculate the distribution of H2B values in the ROI"""
        points = set(self.intensity.keys())
        min_y = min(vert[0] for vert in self.verts)
        max_y = max(vert[0] for vert in self.verts)
        h2b_distribution = np.zeros(101)
        for j, i in points.intersection(h2b_centers):
            # cacluate point in distribution
            relative_y = (j - min_y) / (max_y - min_y)
            h2b_distribution[int(np.floor(relative_y * 100))] += 1

        self.h2b_distribution = h2b_distribution


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

    def create_axon_mask(self, clip_limit, image_only=False):
        """Identify axons in the ROI"""
        verts = list(self.intensity.keys())
        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Filestem
        stem = Path(self.filename).stem

        # Create an image of the ROI
        image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        mask = np.zeros_like(image)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            image[y, x] = self.intensity[vert]
            mask[y, x] = 1

        if image_only:
            self.intensity = None
            self.verts = None
            output_path = Path(f"./manual/{stem[:4]}/{self.name}")
            output_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{str(output_path / stem)}.tif", image)
            return

        # the binary mask of the axons
        binary = get_axon_mask(image, clip_limit=clip_limit)
        outside_points = np.argwhere(mask == 0)
        binary = correct_edges(outside_points, binary)
        x_range = max_x - min_x
        y_range = max_y - min_y

        # post-hoc equalization for visualization
        image = (equalize_adapthist(image, clip_limit=0.0005) * 255).astype(np.uint8)

        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        colored_image[binary == 1] = (0, 0, 255)
        output_folder = Path(f"./screening/{stem[:4]}/{self.name}").resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{str(output_folder)}/{stem}.png", colored_image)

        # Normalize the coordinates to fit into a 101x101 grid
        normalized_mask = np.zeros((101, 101), dtype=np.uint8)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            norm_y, norm_x = int(y / y_range * 100), int(x / x_range * 100)
            if 0 < norm_y < 100 and 0 < norm_x < 100:
                normalized_mask[norm_y, norm_x] += 1 if (binary[y, x] > 0) else 0

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
