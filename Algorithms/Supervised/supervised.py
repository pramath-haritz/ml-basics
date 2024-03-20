import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        #Calculate Distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #Sort Them from lowest to highest
        k_idx = np.argsort(distances)[:self.k]
        k_near_labels = [self.y_train[i] for i in k_idx]
        #Pick k instances and use majority rule to determine class
        majority = Counter(k_near_labels).most_common(1)

        return majority[0][0]

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len( self._classes)

        #initalize mean, variance and prior probabilites 
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)

        for idx, cls in enumerate(self._classes):
            Xc = X[y==cls]
            self._mean[idx, :] = Xc.mean(axis=0)
            self._var[idx, :] = Xc.var(axis=0)
            self._priors[idx] = Xc.shape[0]/float(n_samples)
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx,x)))
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean)**2)/(2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator
class LinearRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T,(y_hat-y))
            db = (1/n_samples) * np.sum((y_hat-y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights)+ self.bias
        return y_pred

class LogisticRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear)
            
            dw = (1/n_samples) * np.dot(X.T,(y_hat-y))
            db = (1/n_samples) * np.sum((y_hat-y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear)
        y_pred_cls = [1 if i>0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
