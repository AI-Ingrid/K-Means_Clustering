import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
sns.set_style('darkgrid')
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=2):
        self.k = k
        self.centroids = np.array([None])

    def k_means_plus_plus(self, X):
        """
         Implements a smart way of randomization of centroids

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # Set random seed
        np.random.seed(0)

        # Create centroid numpy array of size k
        self.centroids = np.array([[0.0, 0.0] for i in range(self.k)])

        # Choose a random centroid from the data set
        self.centroids[0] = X[randint(0, X.shape[0])]

        # Compute the remaining k - 1 centroids
        for i in range(1, self.k):
            # Find all distances to current centroids
            distances_to_centroid = cross_euclidean_distance(X, self.centroids[:i])

            # Calculate the sum of distance from the current centroid for each data point
            sum_distances = np.sum(distances_to_centroid, axis=1)

            # Find datapoint with largest distance to current centroids
            max_distance_index = np.argmax(sum_distances)

            best_centroid = X[max_distance_index]
            self.centroids[i] = best_centroid

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # Call method to randomize centroids if array is empty
        if self.centroids.size:
            self.k_means_plus_plus(X)

        # Fit data points
        while True:
            # Find assigned cluster for each data point
            cluster_assignments = self.predict(X)

            # Create temporary centroids storage space
            temporary_centroids = self.centroids.copy()

            # Calculate new centroids
            for i in range(len(self.centroids)):
                sum = 0
                count = 0
                # Calculate the mean value for each cluster
                for index, item in enumerate(cluster_assignments):
                    if item == i:
                        sum += X[index]
                        count += 1
                if count != 0:
                    temporary_centroids[i] = sum/count

            # Check for convergence
            if np.array_equal(temporary_centroids, self.centroids):
                break
            # Change value of centroids
            else:
                self.centroids = temporary_centroids

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # Calculate the euclidean distances for all data points
        # to each centroid
        euclidean_distances = cross_euclidean_distance(X, self.centroids)
        cluster_assignment = []

        # Assign data point to nearest cluster
        for row in euclidean_distances:
            min_value = min(row)
            cluster_assignment.append(np.where(row == min_value)[0][0])

        return cluster_assignment

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:>>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))

# Self made main functions for each data set
# taken most parts from Jupyter notebook


def main_data_set_1():
    # Load data set
    data_1 = pd.read_csv('data_1.csv')

    # Plot assigned data set
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', data=data_1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    # Fit Model
    X = data_1[['x0', 'x1']]
    model_1 = KMeans()
    X_numpy = X.to_numpy()
    model_1.fit(X_numpy)

    # Compute Silhouette Score
    z = model_1.predict(X_numpy)
    print(f'Silhouette Score: {euclidean_silhouette(X, z) :.3f}')
    print(f'Distortion: {euclidean_distortion(X, z) :.3f}')

    # Plot cluster assignments
    C = model_1.get_centroids()
    K = len(C)
    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax)
    sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove()
    plt.show()


def main_data_set_2():
    # Load data set
    data_2 = pd.read_csv('data_2.csv')
    # Normalize the data
    normalized_df = (data_2-data_2.mean())/data_2.std()

    # Plot assigned data set
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', data=data_2)
    plt.show()

    # Fit Model
    X = normalized_df[['x0', 'x1']].to_numpy()
    model_2 = KMeans(10)
    model_2.fit(X)

    # Compute Silhouette Score
    z = model_2.predict(X)
    print(f'Distortion: {euclidean_distortion(X, z) :.3f}')
    print(f'Silhouette Score: {euclidean_silhouette(X, z) :.3f}')

    # Plot cluster assignments
    C = model_2.get_centroids()
    K = len(C)
    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=normalized_df, ax=ax)
    sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove()
    plt.show()
