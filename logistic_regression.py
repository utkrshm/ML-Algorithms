import numpy as np


class LogisticRegression:
    def __init__(self, lr=1e-5, epochs=1000):
        self.lr = lr
        self.n_epochs = epochs
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Weight initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)   
        self.bias = 0                          
    
        for _ in range(self.n_epochs):
            # Forward pass
            preds = (X @ self.weights) + self.bias
            preds = self.sigmoid(preds)

            # Backward pass
            dw = (1 / n_samples) * (X.T @ (preds - y))
            db = (1 / n_samples) * np.sum(preds - y)            
            
            # Gradient update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
                
    def predict(self, X) -> Exception | list:
        if X.ndim > 2:
            return ValueError("Inputs must not have more than 2 dimensions.")
        else:
            linear_outputs = np.dot(X, self.weights) + self.bias
            preds = self.sigmoid(linear_outputs)
            return [1 if i > 0.5 else 0 for i in preds]


if __name__ == "__main__":
    from time import time

    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import (
        LogisticRegression as SkLogisticRegression,  # For verification
    )
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # print(X.shape, len(y))
    
    # Testing self implementation
    start_time = time()
    model = LogisticRegression(epochs=100)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    acc = (y_preds == y_test).sum() / len(y_test)
    print(y_preds, y_test, (y_preds == y_test).sum())
    
    print(f"Accuracy with self implementation: {(acc * 100):.2f} %")
    print(f"Time taken (in seconds): {time() - start_time}")
    # Accuracy on the breast cancer dataset: 80.70%
        
    # Testing sklearn implementation
    start_time = time()
    
    sk_model = SkLogisticRegression()
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    
    sk_acc = (sk_preds == y_test).sum() / len(y_test)    
    print(f"Accuracy with self implementation: {(sk_acc * 100):.2f} %")
    print(f"Time taken (in seconds): {time() - start_time}")
    # Accuracy on the breast cancer dataset: 94.74% (failed to converge after 100 iterations)
