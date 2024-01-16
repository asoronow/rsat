import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu, try_all_threshold
import cv2


def sobel(image):
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)

    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    return combined


class ROI:
    """
    ROI Object

    """

    def __init__(self, name, intensity, filename, coverage=None):
        self.name = name
        self.intensity = intensity
        self.filename = filename
        self.coverage = coverage
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

    def outliers(self, top=4000, bottom=0):
        # Keep only values between the lower and upper threshold and set zero elsewhere
        filtered_intensity = {}
        for i, j in self.intensity.keys():
            if self.intensity[(i, j)] <= top and self.intensity[(i, j)] >= bottom:
                filtered_intensity[(i, j)] = self.intensity[(i, j)]
            else:
                filtered_intensity[(i, j)] = 0

        # Update the intensity
        self.intensity = filtered_intensity

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

    def normalize(self):
        """Normalize the coordinates of the ROI and Mask to 0-100"""
        keys_array = np.array(list(self.intensity.keys()))
        max_coords = keys_array.max(axis=0)
        min_coords = keys_array.min(axis=0)

        # Compute normalized values
        normalized_keys = (
            (keys_array - min_coords) / (max_coords - min_coords) * 100
        ).astype(int)
        self.intensity = {
            (i, j): self.intensity[(x, y)]
            for (i, j), (x, y) in zip(normalized_keys, keys_array)
        }

    def create_axon_mask(self):
        """Identify axons in the ROI"""

        verts = list(self.intensity.keys())

        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Create an image of the ROI
        image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint16)
        mask = np.zeros_like(image, dtype=np.uint8)

        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            image[y, x] = self.intensity[vert]
            mask[y, x] = 1
        stem = Path(self.filename).stem
        # Edge detection
        edges = sobel(image)
        # Thresholding
        thresh = threshold_otsu(edges)
        # Create binary image
        binary = edges > thresh
        binary = clear_border(binary)
        # Find the range of x and y coordinates
        # plot the three image, edges, and binary
        # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
        # ax = axes.ravel()
        # ax[0] = plt.subplot(1, 3, 1, adjustable="box")
        # ax[1] = plt.subplot(1, 3, 2)
        # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable="box")

        # ax[0].imshow(image, cmap=plt.cm.gray)
        # ax[0].set_title("Original")
        # ax[0].axis("off")

        # ax[1].imshow(edges, cmap=plt.cm.gray)
        # ax[1].set_title("Sobel Edge Detection")
        # ax[1].axis("off")

        # ax[2].imshow(binary * 255, cmap=plt.cm.gray)
        # ax[2].set_title("Thresholded")
        # ax[2].axis("off")

        # plt.savefig(f"{stem}_thresholded.png", dpi=600)

        x_range = max_x - min_x
        y_range = max_y - min_y
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Normalize the coordinates to fit into a 101x101 grid
        normalized_mask = np.zeros((101, 101), dtype=np.uint8)
        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            norm_y, norm_x = int(y / y_range * 100), int(x / x_range * 100)
            if 0 < norm_y < 100 and 0 < norm_x < 100:
                normalized_mask[norm_y, norm_x] += 1 if (binary[y, x] == 1) else 0
            image[y, x] = (0, 0, 255) if (binary[y, x] == 1) else (0, 255, 0)

        # Convert to uint8 and save
        # cv2.imwrite(f"image_{self.name}_colorized.png", image)
        # cv2.imwrite(f"normalized_binary_{self.name}.png", binary_uint8)
        self.mask = normalized_mask


def loadROI(path):
    """
    Load ROIs in as a generator.
    """
    toLoad = []
    if os.path.isfile(path):
        toLoad.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pkl"):
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
