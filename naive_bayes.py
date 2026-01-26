import numpy as np

# Formulae:
# Class conditional probability - P(x_i | y) = exp(-(x_i-mean) / var) / root(2*pi*var)
# This is calculated for all x_i belonging to X for all classes and then the argmax over y is calculated to get the Naive Bayes prediction

class NaiveBayesClassifier:    
    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # From the training data, we only need the mean, var and prior probabilities for each class
        self.means = np.zeros((n_classes, n_features), dtype=np.float32)
        self.vars = np.zeros((n_classes, n_features), dtype=np.float32)
        self.priors = np.zeros(n_classes, dtype=np.float32)
        
        for cls_num in range(len(self.classes)):
            X_class = X[cls_num == y, :]
            self.means[cls_num, :] = X_class.mean(axis=0)    
            self.vars[cls_num, :] = X_class.var(axis=0)    
            self.priors[cls_num] = X_class.shape[0] / len(y)
        
    def _get_class_conditional_probability(self):
        pass
    
    def _predict(self, x):
        # Formula:
        # y_preds = argmax[y=all labels] (P(y | x))
        # y_preds = argmax[y=all labels] (P(x | y) * P(y) / P(x))
        # y_preds = argmax[y=all labels] (P(x | y) * P(y))        [Drop P(x) as the prob of data is constant, and independent of the label]
        # P(x | y) = sum[i = 0...n_features] (P(x_i | y) / P(y))
        # y_preds = argmax[y=all labels] (sum[i = 0...n_features] (P(x_i | y)))
        pred_probs = np.zeros(len(self.classes), dtype=np.float64)
        
        for cls_num in range(len(self.classes)):
            means = self.means[cls_num, :]
            vars = self.vars[cls_num, :]
            conditional_probs = 1 / np.sqrt(2 * np.pi * vars) * np.exp(-(x - means)**2 / vars)
            pred_probs[cls_num] = np.prod(conditional_probs) * self.priors[cls_num]
        
        return self.classes[np.argmax(pred_probs)].astype(int)
        
                
    def predict(self, X):
        return [self._predict(x) for x in X]
    
 
if __name__ == "__main__":
    from time import time

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=2707)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2707)
    
    # print(X.shape, y.shape)

    # Testing self implementation
    start_time = time()
    
    nb_clf = NaiveBayesClassifier()
    nb_clf.fit(X_train, y_train)
    y_preds = nb_clf.predict(X_test)
    acc = (y_preds == y_test).sum() / len(y_preds)
    
    print(f"Accuracy with self implementation: {(acc * 100):.2f}%")
    print(f"Time taken (in seconds): {time() - start_time}")

    # Testing sklearn implementation
    start_time = time()

    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    y_preds_sklearn = sklearn_nb.predict(X_test)
    acc_sklearn = (y_preds_sklearn == y_test).sum() / len(y_preds_sklearn)

    print(f"Accuracy with sklearn implementation: {(acc_sklearn * 100):.2f}%")
    print(f"Time taken (in seconds): {time() - start_time}")