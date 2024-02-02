import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
from scipy.ndimage import label, binary_dilation, center_of_mass
from scipy.spatial import cKDTree
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

def draw_nearest_neighbor_lines(binary_image, original_image, max_distance_percent=0.1):
    """
    Draw lines connecting each component in the binary image to its nearest neighbor.

    Parameters:
        binary_image (np.array): A binary image where 1 represents a detected point.

    Returns:
        np.array: An image with lines connecting each component to its nearest neighbor.
    """
    # Label connected components
    labeled_array, num_features = label(binary_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # Find centroids of components
    centroids = center_of_mass(binary_image, labeled_array, range(1, num_features + 1))
    centroids = np.array(centroids)

    # Create an empty image to draw lines
    line_image = np.zeros_like(binary_image, dtype=np.uint8)
    max_dimension = max(binary_image.shape)
    max_distance = max_distance_percent * max_dimension
    # Build KD-tree with centroids
    if len(centroids) > 1:
        kd_tree = cKDTree(centroids)

        # Draw lines to nearest neighbor within max_distance
        for centroid in centroids:
            dist, nearest_idx = kd_tree.query(centroid, k=2)  # k=2 because the first one is the point itself
            if dist[1] <= max_distance:
                nearest_centroid = centroids[nearest_idx[1]]
                cv2.line(line_image,
                         (int(centroid[1]), int(centroid[0])),
                         (int(nearest_centroid[1]), int(nearest_centroid[0])),
                         (255), 1)
                cv2.line(original_image,
                            (int(centroid[1]), int(centroid[0])),
                            (int(nearest_centroid[1]), int(nearest_centroid[0])),
                            (0, 0, 255), 1)
    
    return line_image, original_image

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
        image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        mask = np.zeros_like(image, dtype=np.uint8)

        for vert in verts:
            y, x = vert[0] - min_y, vert[1] - min_x
            image[y, x] = self.intensity[vert]
            mask[y, x] = 1

        # Filename
        stem = Path(self.filename).stem
        # Edge detection
        gauss = cv2.GaussianBlur(image, (3, 3), 0)
        lapalace = cv2.Laplacian(gauss, cv2.CV_64F, ksize=3)
        edges = cv2.convertScaleAbs(lapalace)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Remove border
        thresh = threshold_otsu(edges)
        binary = edges > (thresh + 15)
        outside_points = np.argwhere(mask == 0)
        binary = correct_edges(outside_points, binary, max_distance=20)

        # Overlay the lines on the edges image
        # Assuming the lines are in white (255), we can color them (e.g., in red)
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


        # make a folder to save the images
        output_folder = Path(f"./screening/{stem[:4]}/{self.name}").resolve()
        output_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{str(output_folder)}/{stem}_thresholded.png", dpi=600)
        plt.close()

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
        cv2.imwrite(f"{str(output_folder)}/{stem}_colorized.png", image)

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
