import random

import numpy as np

"""
Followed this reference for implementation: https://www.cse.iitb.ac.in/~swaprava/courses/cs217/references/kmeans-convergence.pdf
"""

class KMeansClustering:
    def __init__(self, n_clusters: int, n_iters: int = 500):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.cluster_centers = None
        
    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.cluster_centers = np.random.default_rng().choice(X, self.n_clusters, axis=0, replace=False, shuffle=True)        
        
        iters = 0
        prev_assignments = None
        
        while iters < self.n_iters:
            # Using broadcasting to calculate the distance of the points to all the cluster centers
            reshaped_centers = self.cluster_centers.reshape(1, self.n_clusters, n_features)
            
            # Broadcast X to take the shape of (n_samples, 1, n_features)
            X_broadcast = X.reshape((n_samples, 1, n_features))
            # new_X = X[:, np.newaxis, :]   # Another way of implementation
            
            distances = ((X_broadcast - reshaped_centers)**2).sum(axis=2)
            closest_clusters = np.argmin(distances, axis=1)

            # Checking convergence condition
            if prev_assignments is not None and all(prev_assignments == closest_clusters):
                print(f"Covergence achieved, no point assignments were updated. Current iterations = {iters}")
                break
            prev_assignments = closest_clusters.copy()
                
                
            # Re-calculate the centers for each cluster                                  
            points_mean = np.zeros((self.n_clusters, n_features))                
            for i in range(self.n_clusters):
                points_mean[i] = np.mean(X[closest_clusters==i], axis=0)
                
                # In case there are no points alloted to the cluster, retain closest_clusters
                if np.isnan(np.min(points_mean[i])):
                    points_mean[i] = X[random.choice(range(1, n_samples))]
                    
            self.cluster_centers = points_mean
                
            iters += 1        

    def transform(self, x):
        if self.cluster_centers is None:
            raise AssertionError("Please fit the algorithm to data before using the transform function")

        n_samples, n_features = x.shape
        reshaped_x = x.reshape((n_samples, 1, n_features))
        reshaped_centers = self.cluster_centers.reshape(1, self.n_clusters, n_features)
        distances = ((reshaped_x - reshaped_centers)**2).sum(axis=2)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    from time import time

    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    
    X, y = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2707)    
    
    # print(X.shape, len(y))

    # Plot 1: True values of y_test against their centers
    _, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(5):
        ax[0].scatter(
            X_test[y_test == i, 0],
            X_test[y_test == i, 1],
            label=f"Cluster {i}",
            alpha=0.6
        )
    ax[0].set_title("True Cluster Assignments")
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].legend()
        
    # Testing self implementation
    kmeans = KMeansClustering(n_clusters=5)
    kmeans.fit(X_train)
    y_preds = kmeans.transform(X_test)

    # Plot 2: Predictions from self implementation of K-Means
    for i in range(5):
        ax[1].scatter(
            X_test[y_preds == i, 0],
            X_test[y_preds == i, 1],
            label=f"Cluster {i}",
            alpha=0.6
        )
    ax[1].set_title("Self Implemented K-Means Predictions")
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].legend()

    # Testing sklearn implementation
    sk_kmeans = KMeans(n_clusters=5, random_state=42)
    sk_kmeans.fit(X_train)
    sk_preds = sk_kmeans.predict(X_test)

    # Plot 3: Predictions from the sklearn implementation    
    for i in range(5):
        ax[2].scatter(
            X_test[sk_preds == i, 0],
            X_test[sk_preds == i, 1],
            label=f"Cluster {i}",
            alpha=0.6
        )
    ax[2].set_title("Sklearn KMeans Assignments")
    ax[2].set_xlabel("Feature 1")
    ax[2].set_ylabel("Feature 2")
    ax[2].legend()

    # Displaying all the plots
    plt.tight_layout()
    plt.show()
