# Implement spectral clustering algorithm. Your final result should be spectral.py,
# which should include class named SpectralClustering, which includes __init__, fit and predict methods,
# which return outputs as usual. Add solution to github and send a link to the repository.
# Please note, that the algorithm should work for any number of clusters, not only for 2.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


class SpectralClustering:
    def __init__(self, n_clusters=2, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.X = X
        self.graph = self._create_similarity_graph(X)
        self.normalized_graph = normalize(self.graph)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.normalized_graph)
        self.embedding = self.eigenvectors[:, :self.n_clusters]
        self.labels = KMeans(n_clusters=self.n_clusters).fit_predict(self.embedding)

    def predict(self, X):
        graph = self._create_similarity_graph(X, self.X)
        normalized_graph = normalize(graph)
        embedding = np.dot(normalized_graph, self.eigenvectors)
        return KMeans(n_clusters=self.n_clusters).fit_predict(embedding)

    def _create_similarity_graph(self, X, Y=None):
        if Y is None:
            Y = X

        graph = kneighbors_graph(Y, self.n_neighbors, mode='connectivity', include_self=True)
        return 0.5 * (graph + graph.T)