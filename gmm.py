"""Multivariate GMM"""
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_comps=5, n_iters = 100):
        self.k = n_comps
        self.n_iters = n_iters

        # Give type annotations for means and covariance matrix for now, they'll be defined later
        self.means: np.ndarray
        self.cov_mat: np.ndarray    
   
    def _init_hidden(self, x: np.ndarray):
        N, D = x.shape          # (Number of data points, no of features per data point)

        # In the start, we want the weights to be equally split across all the components
        self.pi = np.ones(self.k) / self.k

        rand_idx = np.random.choice(N, self.k, replace=False)
        self.means = x[rand_idx]
        self.cov_mat = np.stack([np.eye(D) for _ in range(self.k)])

    def _expectation(self, x: np.ndarray):
        # We now use the bayesian formula to get the probability of the data point being in a particular class
        # Formula = w_k * N(x_i | mu_k, sigma_k) / sum(w_j * N(x_j | mu_j, sigma_j) for j = 1 -> k) = p(C_k | x_i, w, mu, sigma)
        # This has to be done for all the clusters
        N, D = x.shape
        probs = np.zeros((N, self.k))        # The shape of this is identical to self.pi

        for i in range(self.k):
            print(self.means[i].shape, x.shape, self.cov_mat[i].shape, end="\t")
            print(self.pi[i].shape, probs[:, i].shape, multivariate_normal.pdf(x, self.means[i], self.cov_mat[i]).shape)
            probs[:, i] = self.pi[i] * multivariate_normal.pdf(x, self.means[i], self.cov_mat[i])

        total_probs = probs.sum(axis=1, keepdims=True)

        return probs / total_probs

    def _maximization(self, x, probs):
        N = x.shape[0]

        for i in range(self.k):
            self.pi[i] = np.sum(probs[:, i]) / N
            self.means[i] = np.sum(probs[:, i].reshape(-1, 1) * x, axis=0) / np.sum(probs[:, i])
            
            # Calculating the covariance matrix
            diff = x - self.means[i]
            weighted_diff = probs[:, i].reshape(-1, 1) * diff
            numer = np.dot(diff.T, weighted_diff)
            self.cov_mat[i, :, :] =  numer / np.sum(probs[:, i])


    def fit(self, x):
        self._init_hidden(x)

        iter_count = 0
        while (iter_count < self.n_iters):
            probs = self._expectation(x)
            self._maximization(x, probs)
            iter_count += 1

    def predict(self, x) -> np.ndarray:
        probs = self._expectation(x)
        return np.argmax(x, axis=1)


if __name__ == "__main__":
    from time import time
    
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X = X[:, :2] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    fig, ax = plt.subplots(1, 2, sharex="none", sharey="none", width_ratios=[1, 1])
    ax = ax.flatten()
    
    # Testing self implementation
    start_time = time()
    
    my_gmm = GMM(n_comps=3, n_iters=100)
    my_gmm.fit(X_train)
    my_preds = my_gmm.predict(X_test)

    print(f"Time taken by self GMM (in seconds): {time() - start_time}")
    
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=my_preds, cmap='viridis')
    ax[0].set_title("Self GMM")
    
    # Testing sklearn implementation
    start_time = time()
    
    sk_gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
    sk_gmm.fit(X_train)
    sk_preds = sk_gmm.predict(X_test)
    
    print(f"Time taken by sklearn GMM (in seconds): {time() - start_time}")

    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=sk_preds, cmap='viridis')
    ax[1].set_title("sklearn GMM")
    
    # Showing plots
    plt.show()
