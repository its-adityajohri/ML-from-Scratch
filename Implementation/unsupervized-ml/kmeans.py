from __future__ import print_function, division
import numpy as np
from mlfromscratch.utils import normalize, euclidean_distance, Plot
from mlfromscratch.unsupervised_learning import *

class KMeansClustering():
    """A clustering method that forms clusters by iteratively reassigning
    samples to the closest centroids and then moving the centroids to the center
    of the new formed clusters.

    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The maximum number of iterations the algorithm will run for if it does
        not converge before that. 
    """
    def __init__(self, k=2, max_iterations=500):
        self.max_iterations = max_iterations
        self.k = k

    def _random_centroids_init(self, X):
        """ Initializes the centroids as k random samples from X"""
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        """ Finds the index of the closest centroid to the sample """
        closest_index = 0
        closest_distance = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_distance:
                closest_index = i
                closest_distance = distance
        return closest_index

    def _create_clusters(self, centroids, X):
        """ Assigns the samples to the closest centroids to create clusters """
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_index, sample in enumerate(X):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(sample_index)
        return clusters

    def _calculate_centroids(self, clusters, X):
        """ Calculates new centroids as the means of the samples in each cluster  """
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _get_cluster_labels(self, clusters, X):
        """ Classifies samples by the index of their clusters """
    
