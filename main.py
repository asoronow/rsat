import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border, flood_fill
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from skimage import img_as_float64
from skimage.filters import gaussian, laplace


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
        # Convert keys to numpy array for efficient computation
        keys_array = np.array(list(self.intensity.keys()))
        max_coords = keys_array.max(axis=0)
        min_coords = keys_array.min(axis=0)
        
        # Compute normalized values
        normalized_keys = ((keys_array - min_coords) / (max_coords - min_coords) * 100).astype(int)
        self.intensity = {(i, j): self.intensity[(x, y)] for (i, j), (x, y) in zip(normalized_keys, keys_array)}

    def create_axon_mask(self):
        """Identify axons in the ROI"""

        verts = list(self.intensity.keys())

        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Create a image of the ROI
        image = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint16)
        for vert in verts:
            image[vert[1] - min_x, vert[0] - min_y] = self.intensity[vert]

        masked_image = np.ma.masked_where(image == 0, image)
        image = img_as_float64(masked_image)
        log_image = laplace(gaussian(image, sigma=2))
        log_abs = np.abs(log_image)
        thresh = threshold_otsu(log_abs)
        binary = log_abs > thresh
        binary = clear_border(binary)
        # binary = remove_small_holes(binary, 1000, connectivity=2)

        binary_uint8 = (binary * 255).astype(np.uint8)
        self.mask = binary_uint8

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


if __name__ == "__main__":
    roi_path = Path("samples")
    rois = loadROI(str(roi_path))
    num_rois = roi_path.rglob("*.pkl")

    binary_images = []
    compare_images = []
    all_values = []
    c = 0
    for roi in rois:
        verts = list(roi.intensity.keys())

        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Create a image of the ROI
        image = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint16)
        for vert in verts:
            image[vert[1] - min_x, vert[0] - min_y] = roi.intensity[vert]

        masked_image = np.ma.masked_where(image == 0, image)
        image = img_as_float64(masked_image)
        log_image = laplace(gaussian(image, sigma=2))
        log_abs = np.abs(log_image)
        thresh = threshold_otsu(log_abs)
        binary = log_abs > thresh
        binary = clear_border(binary)
        binary = remove_small_holes(binary, 1000, connectivity=2)

        binary_uint8 = (binary * 255).astype(np.uint8)
        c += 1
        binary_images.append(binary_uint8)
        compare_images.append(image)

    # side by side compare and binary images
    fig, ax = plt.subplots(c, 3, figsize=(10, 10))
    for i in range(c):
        ax[i, 0].imshow(compare_images[i], cmap="gray")
        ax[i, 0].set_title("Original")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(binary_images[i])
        ax[i, 1].axis("off")
        ax[i, 1].set_title("Binary")
        ax[i, 2].hist(compare_images[i].ravel(), bins=256, range=(0, 255))
        # Add vertical line on each histogram to show threshold
        ax[i, 2].set_title("Histogram")

    plt.show()
