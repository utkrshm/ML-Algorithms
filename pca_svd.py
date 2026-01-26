# Implementation of the PCA Algorithm using the Singular Value Decomposition Method
import numpy as np

# Reference: https://cs357.cs.illinois.edu/textbook/notes/pca.html

# Process of PCA
# 1. Center all the features using their mean
# 2. Calculate SVD (getting singular values and Eigenvectors)
# 3. Transform singular values into eigenvalues
# 4. Sorting eigenvectors and eigenvalues, and transforming the data

class PrincipalComponentAnalysis:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.explained_variances = None
        self._components = None
        self.mean = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # Centering the data
        self.mean = X.mean(axis=0)
        X = X - self.mean
        
        # Calculating SVD
        U, S, V_t = np.linalg.svd(X, full_matrices=False)
        
        eigenvalues = S**2 / (n_samples - 1)
        total_variance = np.sum(eigenvalues)
        print("Variance ratios:", eigenvalues[:self.n_components] / total_variance)        
        
        self._explained_variances = eigenvalues[:self.n_components]
        self._components = V_t[:self.n_components]
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self._components.T)


if __name__ == "__main__":
    from time import time

    import matplotlib.pyplot as plt
    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2707)    
    
    # print(X.shape, len(y))
    
    # Setting up matplotlib
    fig, ax = plt.subplots(1, 2, sharex="none", sharey="none", width_ratios=[1, 1])
    ax = ax.flatten()
    
    # Testing self implementation
    start_time = time()
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X_train, y_train)
    X_new = pca.transform(X_test)
    
    print("Time taken by self PCA: ", time() - start_time)
    
    first_pc = X_new[:, 0]
    second_pc = X_new[:, 1]
    ax[0].scatter(first_pc, second_pc, c=y_test)
    ax[0].set_title("Self PCA")
    
    # Testing sklearn implementation
    start_time = time()
    
    sk_pca = PCA()
    sk_pca.fit(X_train, y_train)
    sk_proj = sk_pca.transform(X_test)
    
    print("Time taken by sklearn PCA: ", time() - start_time)

    ax[1].scatter(sk_proj[:, 0], sk_proj[:, 1], c=y_test)
    ax[1].set_title("sklearn PCA")
    # Showing plots
    plt.show()    