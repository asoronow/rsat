import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import mode
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
    rois = list(loadROI(str(roi_path)))
    num_rois = roi_path.rglob("*.pkl")

    binary_images = []
    compare_images = []
    all_values = []
    fig, ax = plt.subplots(4, 3, figsize=(12, 8))

    for roi in rois:
        all_values.extend(list(roi.intensity.values()))

    all_values = np.array(all_values)
    all_values = all_values[all_values > 0]

    threshold = mode(all_values, axis=None)[0][0]
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

        compare_images.append(cv2.resize(image, (512, 512)))
        # Cvt to 8 bit
        max_value = np.max(image)
        min_value = np.min(image)
        image = ((image - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        # Threshold
        binary_mask = np.zeros_like(image)
        binary_mask[image < threshold] = 0
        binary_mask[image >= threshold] = 255

        # Convert original image to color
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color_image[binary_mask == 255] = [193, 100, 255]
        
        # resize the image to 256x256
        color_image = cv2.resize(color_image, (512, 512))
        binary_images.append(color_image)

    # side by side compare and binary images
    for i in range(4):
        ax[i, 0].imshow(compare_images[i], cmap="gray")
        ax[i, 0].set_title("Original")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(binary_images[i], cmap="gray")
        ax[i, 1].axis("off")
        ax[i, 1].set_title("Binary")
        ax[i, 2].hist(compare_images[i].ravel(), bins=256, range=(0, 255))
        ax[i, 2].set_xscale("log")
        ax[i, 2].set_yscale("log")
        # Add vertical line on each histogram to show threshold
        ax[i, 2].axvline(threshold, color="r", linestyle="dashed", linewidth=1)
        ax[i, 2].set_title("Histogram")


    plt.tight_layout()
    plt.show()

