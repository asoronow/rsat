import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

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
    all_rois = []
    for roi in rois:
        verts = list(roi.intensity.keys())

        # Find the bounding box for the ROI
        min_x = min(vert[1] for vert in verts)
        max_x = max(vert[1] for vert in verts)
        min_y = min(vert[0] for vert in verts)
        max_y = max(vert[0] for vert in verts)

        # Create a 2D array based on the bounding box dimensions
        img = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint16)
        for point, value in roi.intensity.items():
            img[point[1] - min_x, point[0] - min_y] = value
        
        # Threshold the image
        binary_img = (img > 200).astype(np.uint8)
        binary_images.append(binary_img)


    # Find the largest image
    max_shape = np.max([img.shape for img in binary_images], axis=0)
    # Pad all images to the same size
    for i in range(len(binary_images)):
        binary_images[i] = np.pad(
            binary_images[i],
            (
                (0, max_shape[0] - binary_images[i].shape[0]),
                (0, max_shape[1] - binary_images[i].shape[1]),
            ),
            "constant",
            constant_values=0,
        )
    # Create the 3D reconstruction
    volume = np.stack(binary_images, axis=2)

    # Visualize the 3D reconstruction using marching cubes and matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract surfaces using marching cubes
    verts, faces, _, _ = marching_cubes(volume)

    # Create a mesh to visualize the 3D structure
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    ax.add_collection3d(mesh)

    plt.show()
