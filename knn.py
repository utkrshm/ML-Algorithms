import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def _euclidean(self, a: np.ndarray, b: np.ndarray):
        return np.linalg.norm(a - b)
        # return np.sqrt(np.sum((a -  b) ** 2))

    def _argsort(self, distances) -> list:
        return sorted(range(len(distances)), key=distances.__getitem__)

    def _count_classes(self, classes: np.ndarray) -> tuple | None:
        uniques, counts = np.unique(classes, return_counts=True)
        mapping = tuple(zip(uniques, counts))
        sorted_mapping = tuple(reversed(sorted(mapping, key=lambda x: x[1])))
        return sorted_mapping
        

    def _predict(self, x: np.ndarray):
        distances = np.array([self._euclidean(x, train_set_pt) for train_set_pt in self.X_train])
        sorted_distances_indices = self._argsort(distances)
        
        k_nearest_indices = sorted_distances_indices[:self.k]
        k_nearest_lbls = self.y_train[k_nearest_indices]

        most_probable_lbls = self._count_classes(k_nearest_lbls)
        
        return most_probable_lbls[0][0]
    
        
    def fit(self, X, y) -> None:
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X) -> np.ndarray:
        if X.ndim == 1:
            return np.array(self._predict(X))
        elif X.ndim == 2:
            return np.array([self._predict(x) for x in X])    
        else:
            raise ValueError("The provided array should not have more than 2 dimensions.")


if __name__ == "__main__":
    from time import time

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier  # For verification

    X, y = load_iris(return_X_y=True)
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Testing self implementation
    start_time = time()
    
    knn = KNN(k=10)
    knn.fit(X_train, y_train)
    y_preds = knn.predict(X_test)
    acc = (y_preds == y_test).sum() / len(y_preds)
    
    print(f"Accuracy with self implementation: {(acc * 100):.2f}%")
    print(f"Time taken (in seconds): {time() - start_time}")
    
    # Testing sklearn implementation
    start_time = time()
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=5, algorithm="brute", p=2)
    sklearn_knn.fit(X_train, y_train)
    sk_preds = sklearn_knn.predict(X_test)
    
    sk_acc = (sk_preds == y_test).sum() / len(sk_preds)
    print(f"Accuracy with sklearn implementation: {(sk_acc * 100):.2f}%")
    print(f"Time taken (in seconds): {time() - start_time}")
