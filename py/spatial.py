import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import seaborn as sns


class Cluster:
    def __init__(self, points, label, color, name):
        self.points = points
        self.label = label
        self.color = color
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.points)


class ExperimentalAnimal:
    def __init__(self, name, age, rois):
        self.name = name
        self.age = age
        self.rois = rois

    def sum_project(self):
        """Use sum projection to create"""
        for roi in self.rois:
            data = np.array(self.rois[roi]).astype(np.float64)
            data = np.sum(data, axis=0, dtype=np.float64)
            data = np.sum(data, axis=0, dtype=np.float64)
            data /= np.max(data)
            data = np.nan_to_num(data)
            self.rois[roi] = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AnimalDataLoader:
    def __init__(self):
        self.animals = []

    def load(self, pkl_path, age_group):
        """Load in a group of animals from a pickle file."""
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
            animals = raw[age_group]
            for animal_name, animal_data in animals.items():
                animal = ExperimentalAnimal(animal_name, age_group, animal_data)
                self.animals.append(animal)
                animal.sum_project()


class AxonDataClustering:
    def __init__(self, animals, n_clusters=2):
        self.animals = animals
        self.n_clusters = n_clusters
        self.all_data = []
        self.roi_labels_ml = []  # Medial/Lateral labels
        self.roi_labels_dv = []  # Dorsal/Ventral labels
        self.roi_labels_age = []  # Age labels
        # Classification based on typical mouse brain organization
        self.medial_lateral_map = {
            "tea": "lateral",
            "visal": "lateral",
            "visrl": "medial",
            "visa": "medial",
            "rspagl": "medial",
            "visli": "lateral",
            "visl": "lateral",
            "visam": "medial",
            "rspd": "medial",
            "vispor": "lateral",
            "vispl": "lateral",
            "vispm": "medial",
            "rspv": "medial",
            "str": "medial",
        }

        self.dorsal_ventral_map = {
            "tea": "dorsal",
            "visal": "dorsal",
            "visrl": "dorsal",
            "visa": "dorsal",
            "rspagl": "dorsal",
            "visli": "ventral",
            "visl": "ventral",
            "visam": "dorsal",
            "rspd": "dorsal",
            "vispor": "ventral",
            "vispl": "ventral",
            "vispm": "dorsal",
            "rspv": "dorsal",
            "str": "ventral",
        }

    def prepare_data_all(self):
        for animal in self.animals:
            for roi, data in animal.rois.items():
                if roi == "visp" or roi == "str":
                    continue
                self.all_data.append(data)  # Appending the ROI data

                # Appending the true labels based on anatomical knowledge
                self.roi_labels_ml.append(self.medial_lateral_map[roi])
                self.roi_labels_dv.append(self.dorsal_ventral_map[roi])
                self.roi_labels_age.append(animal.age)

        self.all_data = np.array(self.all_data)
        self.roi_labels_ml = np.array(self.roi_labels_ml)
        self.roi_labels_dv = np.array(self.roi_labels_dv)
        self.roi_labels_age = np.array(self.roi_labels_age)

    def prepare_data_per_animal(self):
        max_length = 0
        whole_brains = []  # List to hold all concatenated data arrays

        # First, find the maximum length needed by iterating over the animals and their ROIs
        for animal in self.animals:
            whole_brain = []
            for roi, data in animal.rois.items():
                if roi == "visp" or roi == "str":
                    continue
                whole_brain.append(data.reshape(-1))  # Flattening the data array
            if len(whole_brain) == 0:
                continue
            elif len(whole_brain) == 1:
                concatenated = whole_brain[0]
            else:
                concatenated = np.concatenate(whole_brain)
            if concatenated.size > max_length:
                max_length = concatenated.size
            whole_brains.append(concatenated)  # Store for use in the next step

        # Next, pad each 'whole_brain' array to ensure equal lengths
        padded_brains = []
        for brain in whole_brains:
            padded_brain = np.pad(
                brain, (0, max_length - brain.size), "constant", constant_values=0
            )
            padded_brains.append(padded_brain)
            # This creates a list of arrays where each array is of the same length

        self.all_data = np.array(
            padded_brains
        )  # Convert list of arrays into a 2D array

        # Now, self.all_data is suitable for PCA
        self.roi_labels_age = [
            animal.age for animal in self.animals
        ]  # I assume this is correct for your context

    def perform_pca(self):
        pca = PCA(n_components=2)
        pca.fit(
            self.all_data
        )  # self.all_data is now a 2D array, rows are samples, columns are features
        self.all_data = pca.transform(self.all_data)

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        kmeans.fit(self.all_data)
        return kmeans.labels_

    def perform_gmm(self):
        gmm = GaussianMixture(n_components=self.n_clusters, n_init=100)
        gmm.fit(self.all_data)
        return gmm.predict(self.all_data)

    def determine_cluster_labels(self, predicted_labels, real_labels):
        """Using the proportions of the labels in each cluster, determine the cluster labels"""

        # Create a dictionary to hold the count of each label within each cluster.
        cluster_counts = {}
        for cluster_id in set(predicted_labels):
            # For each label, we check how many times it appears in each cluster.
            current_cluster_indices = [
                i for i, x in enumerate(predicted_labels) if x == cluster_id
            ]
            labels_in_cluster = [real_labels[i] for i in current_cluster_indices]

            label_counts = {label: 0 for label in set(real_labels)}
            for label in labels_in_cluster:
                label_counts[label] += 1

            # Store the label counts dictionary in our cluster counts dictionary.
            cluster_counts[cluster_id] = label_counts

        cluster_labels = {}
        for label in set(real_labels):
            # sort clusters by label count
            sorted_clusters = sorted(
                cluster_counts.keys(),
                key=lambda x: cluster_counts[x][label],
                reverse=True,
            )

            # assign the label to the cluster with the highest count
            for cid in sorted_clusters:
                if cid not in cluster_labels:
                    cluster_labels[cid] = label
                    break
                else:
                    continue

        return cluster_labels

    def plot_umap_and_cm_age_only(self, predicted_labels):
        """Plot the UMAP and confusion matrix for age only, for use with whole animal data"""
        # UMAP reduction
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(self.all_data)

        # Plot UMAP
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # UMAP Plot
        ax[0].scatter(
            embedding[:, 0], embedding[:, 1], c=predicted_labels, cmap="Spectral", s=5
        )

        ax[0].set_xlabel("UMAP1")
        ax[0].set_ylabel("UMAP2")
        ax[0].set_title("UMAP Projection")

        # Age Confusion Matrix
        cluster_labels_age = self.determine_cluster_labels(
            predicted_labels, self.roi_labels_age
        )
        age_predicted_labels = [cluster_labels_age[label] for label in predicted_labels]
        conf_mat_age = confusion_matrix(self.roi_labels_age, age_predicted_labels)
        sns.heatmap(
            conf_mat_age,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[1],
            xticklabels=[cluster_labels_age[i] for i in range(self.n_clusters)],
            yticklabels=[cluster_labels_age[i] for i in range(self.n_clusters)],
        )

        ax[1].set_title("Confusion matrix of Age classification")
        ax[1].set_ylabel("True label")
        ax[1].set_xlabel("Predicted label")

    def plot_umap_and_confusion_matrix(self, predicted_labels):
        # UMAP reduction
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(self.all_data)

        # Plot UMAP
        fig, ax = plt.subplots(1, 4, figsize=(12, 6))

        # UMAP Plot
        ax[0].scatter(
            embedding[:, 0], embedding[:, 1], c=predicted_labels, cmap="Spectral", s=5
        )
        ax[0].set_xlabel("UMAP1")
        ax[0].set_ylabel("UMAP2")
        ax[0].set_title("UMAP Projection")

        # Medial/Lateral Confusion Matrix
        cluster_labels_ml = self.determine_cluster_labels(
            predicted_labels, self.roi_labels_ml
        )
        ml_predicted_labels = [cluster_labels_ml[label] for label in predicted_labels]
        conf_mat_ml = confusion_matrix(self.roi_labels_ml, ml_predicted_labels)
        sns.heatmap(
            conf_mat_ml,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[1],
            xticklabels=[cluster_labels_ml[i] for i in range(self.n_clusters)],
            yticklabels=[cluster_labels_ml[i] for i in range(self.n_clusters)],
        )  # Notice we are setting the 'ax' parameter here
        ax[1].set_title("Confusion matrix of Medial/Lateral classification")
        ax[1].set_ylabel("True label")
        ax[1].set_xlabel("Predicted label")

        # Dorsal/Ventral Confusion Matrix
        cluster_labels_dv = self.determine_cluster_labels(
            predicted_labels, self.roi_labels_dv
        )
        dv_predicted_labels = [cluster_labels_dv[label] for label in predicted_labels]
        conf_mat_dv = confusion_matrix(self.roi_labels_dv, dv_predicted_labels)
        sns.heatmap(
            conf_mat_dv,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[2],
            xticklabels=[cluster_labels_dv[i] for i in range(self.n_clusters)],
            yticklabels=[cluster_labels_dv[i] for i in range(self.n_clusters)],
        )  # Notice we are setting the 'ax' parameter here
        ax[2].set_title("Confusion matrix of Dorsal/Ventral classification")
        ax[2].set_ylabel("True label")
        ax[2].set_xlabel("Predicted label")

        # Age Confusion Matrix
        cluster_labels_age = self.determine_cluster_labels(
            predicted_labels, self.roi_labels_age
        )
        age_predicted_labels = [cluster_labels_age[label] for label in predicted_labels]
        conf_mat_age = confusion_matrix(self.roi_labels_age, age_predicted_labels)
        sns.heatmap(
            conf_mat_age,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax[3],
            xticklabels=[cluster_labels_age[i] for i in range(self.n_clusters)],
            yticklabels=[cluster_labels_age[i] for i in range(self.n_clusters)],
        )
        ax[3].set_title("Confusion matrix of Age classification")
        ax[3].set_ylabel("True label")
        ax[3].set_xlabel("Predicted label")

        # Adjusting layout and displaying the plots
        plt.tight_layout()
        plt.show()


def main():
    loader = AnimalDataLoader()
    p3to7_path = "p3to7/raw_experiments_p3to7.pkl"  # Adjust the path as necessary
    old_data_path = (
        "old_data/raw_experiments_old_data.pkl"  # Adjust the path as necessary
    )
    p3or4to7_path = Path(r"D:\VSV raw data\raw_experiments_p3or4to7.pkl")
    p5to9_path = Path(r"D:\VSV raw data\raw_experiments_p5to9.pkl")
    p3to7_path = Path(r"D:\VSV raw data\raw_experiments_p3to7.pkl")
    # loader.load(p3to7_path, "p3to7")
    # loader.load(old_data_path, "adult")
    # loader.load(p3or4to7_path, "p3or4to7")
    loader.load(p3to7_path, "p3to7")
    loader.load(p5to9_path, "p5to9")
    # Just animals
    clustering = AxonDataClustering(loader.animals)
    clustering.prepare_data_per_animal()
    clustering.perform_pca()
    cluster_labels = clustering.perform_gmm()
    clustering.plot_umap_and_cm_age_only(cluster_labels)

    # Clustering and PCA
    clustering = AxonDataClustering(loader.animals)
    clustering.prepare_data_all()
    cluster_labels = clustering.perform_gmm()

    # Plotting UMAP
    clustering.plot_umap_and_confusion_matrix(cluster_labels)


if __name__ == "__main__":
    main()
