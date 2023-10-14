import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import umap
from matplotlib import pyplot as plt

class RSAT:
    def __init__(self, data):
        self.data = data
        self.labels = None

    def _reduce_dimension(self, normalized_data, n_components=30):
        """
        Perform PCA for dimension reduction.
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(normalized_data)

    def _compute_SNN(self, reduced_data, n_neighbors=15):
        """
        Compute Shared Nearest Neighbor (SNN) graph.
        """
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(reduced_data)
        return nbrs.kneighbors_graph(reduced_data).toarray()

    def _clustering(self, SNN_graph, n_clusters):
        """
        Apply clustering on SNN graph.
        """
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        return clustering.fit_predict(SNN_graph)

    def perform_clustering(self, n_clusters):
        """
        RSAT clustering pipeline.
        """
        reduced_data = self._reduce_dimension(self.data)
        SNN_graph = self._compute_SNN(reduced_data)
        self.labels = self._clustering(SNN_graph, n_clusters)

    def visualize_clusters(self, reduced_dimensions=2):
        """
        Visualize data using UMAP.
        """
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='euclidean', n_components=reduced_dimensions).fit_transform(self.data)
        return embedding, self.labels

if __name__ == "__main__":
    # Generating mock data for the demonstration
    mock_data = np.random.randint(2, size=(100, 100))

    rsat = RSAT(mock_data)
    rsat.perform_clustering(n_clusters=5)

    embedding, labels = rsat.visualize_clusters()
    # plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the RSAT clustering', fontsize=24)

    plt.show()