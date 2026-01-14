import matplotlib.pyplot as plt
import numpy as np


# Implementing a Linear Regression using SGD
class LinearRegression:
    def __init__(self, lr=1e-2, epochs=1000):
        self.lr = lr
        self.n_epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Reshaping vectors
        if X.ndim == 1: 
            X = X.reshape((-1, 1))
        elif X.ndim > 2:
            raise ValueError("X vector should not contain more than 2 dimensions.")
        
        if y.ndim == 1: 
            y = y.reshape((-1, 1))
        elif y.ndim > 2:
            raise ValueError("y vector should not contain more than 2 dimensions.")

        # Weight initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))    # Vector of nx1 shape
        self.bias = 0                               # Scalar (will be broadcasted)
    
        for i in range(self.n_epochs):
            # Forward pass
            preds = X @ self.weights + self.bias

            # Backward pass
            dw = 1 / n_samples * np.sum(X * (preds - y)) * 2
            db = 1 / n_samples * np.sum(preds - y) * 2
                        
            # Gradient update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
                
    def predict(self, X) -> Exception | list:
        if X.ndim > 2:
            return ValueError("Inputs must not have more than 2 dimensions.")
        else:
            return (X @ self.weights + self.bias).tolist()


if __name__ == "__main__":
    from time import time

    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import (
        LinearRegression as SkLinearRegression,  # For verification
    )
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # print(X.shape, len(y))
    
    # Testing self implementation
    start_time = time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    err = np.mean((y_preds - y_test) ** 2)
    
    print(f"MSE with self implementation: {err}")
    print(f"Time taken (in seconds): {time() - start_time}")
        
    # Testing sklearn implementation
    start_time = time()
    
    sk_model = SkLinearRegression()
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    
    sk_err = 1 / len(y_test) * np.sum((sk_preds - y_test) ** 2)    
    print(f"MSE with self implementation: {sk_err}")
    print(f"Time taken (in seconds): {time() - start_time}")
