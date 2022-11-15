import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k: int = 5):
        self.k = k
        self.X = None
        self.y = None
        self.distance = lambda x, y: ((x - y[:, None]) ** 2).sum(-1)
        self.indexer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.k = min(self.k, X.shape[0])
        self.indexer = np.vectorize(lambda i: y[i])
        self.X = X
        self.y = y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distance = self.distance(self.X, X)
        k_nearest_index = np.argpartition(distance, 1)[:, : self.k]
        k_nearest = self.indexer(k_nearest_index)
        return mode(k_nearest, axis=1).mode.flatten()

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
