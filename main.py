import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.segmentation import clear_border
from skimage.morphology import binary_closing, binary_dilation, closing, rectangle
from skimage import measure, transform
from skimage.filters import gaussian
import cv2


class ROI:
    """
    ROI Object
    """

    def __init__(self, name, intensity, filename, coverage=None):
        self.name = name
        self.intensity = intensity
        self.filename = filename
        self.coverage = coverage

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

    def outliers(self):
        values = list(self.intensity.values())

        # Determine the thresholds based on percentiles
        lower_threshold = 200

        # Keep only values between the lower and upper threshold and set zero elsewhere
        filtered_intensity = {}
        for i, j in self.intensity.keys():
            if self.intensity[(i, j)] >= lower_threshold:
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
        # normalize the intensity keys to 0-100
        keys = self.intensity.keys()
        # get the max and min values of both x and y
        max_x = np.max([i for i, j in keys])
        min_x = np.min([i for i, j in keys])
        max_y = np.max([j for i, j in keys])
        min_y = np.min([j for i, j in keys])
        # normalize the x , y and z values
        remapped = {}
        values = []
        for i, j in keys:
            remapped[
                (
                    int((i - min_x) / (max_x - min_x) * 100),
                    int((j - min_y) / (max_y - min_y) * 100),
                )
            ] = self.intensity[(i, j)]
            values.append(self.intensity[(i, j)])

        self.intensity = remapped


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

        h_matrix = hessian_matrix(
            image,
            sigma=3,
            use_gaussian_derivatives=True,
        )
        i1, i2 = hessian_matrix_eigvals(h_matrix)
        determinant = i1 * i2
        threshold = np.percentile(determinant, 95)
        mask = determinant > threshold
        mask = binary_dilation(mask, footprint=np.ones((5, 5)))
        mask = binary_closing(mask, footprint=np.ones((5, 5)))
        mask = closing(mask, rectangle(1, 5))
        mask = closing(mask, rectangle(5, 1))
        mask = clear_border(mask)

        min_area = 200
        label_image = measure.label(mask)
        filtered_mask = np.zeros_like(mask, dtype=bool)
        for region in measure.regionprops(label_image):
            if region.area >= min_area:
                filtered_mask = filtered_mask | (label_image == region.label)

        cv2.imwrite(
            f"masks/{roi.name}_{c}.png",
            (filtered_mask * 255).astype(np.uint8),
        )
        cv2.imwrite(
            f"images/{roi.name}_{c}.png",
            image * 255,
        )
        c += 1
        # resize
        filtered_mask = transform.resize(filtered_mask, (1024, 1024))
        image = transform.resize(image, (1024, 1024))

        binary_images.append(filtered_mask)
        compare_images.append(image)

    # side by side compare and binary images
    fig, ax = plt.subplots(6, 3, figsize=(12, 8))
    for i in range(6):
        ax[i, 0].imshow(compare_images[i], cmap="gray")
        ax[i, 0].set_title("Original")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(binary_images[i], cmap="gray")
        ax[i, 1].axis("off")
        ax[i, 1].set_title("Binary")
        ax[i, 2].hist(compare_images[i].ravel(), bins=256, range=(0, 255))
        # Add vertical line on each histogram to show threshold
        ax[i, 2].set_title("Histogram")

    plt.tight_layout()
    plt.show()
