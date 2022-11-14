import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union

from src.task02_classification.decision_tree.node import Node, NodeType

__all__ = ["DecisionTreeClassifier"]


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self._tree = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.depth = 0

    def _fit_node(self, X: np.ndarray, y: np.ndarray, node: Node) -> None:
        self.depth = max(self.depth, node.depth)

        if (
            np.all(y == y[0])
            or len(y) < self.min_samples_split
            or node.depth >= self.max_depth
        ):
            node.type = NodeType.LEAF
            node.classification_class = mode(y).mode[0]
            return

        feature_best = None
        threshold_best = None
        gini_best = None
        split = None

        for feature in range(X.shape[1]):
            feature_vector = X[:, feature]

            if np.unique(feature_vector).size <= 1:
                continue

            threshold, gini = self.__find_best_split(feature_vector, y)
            if gini_best is None or gini > gini_best:
                split = feature_vector < threshold
                feature_best = feature
                gini_best = gini
                threshold_best = threshold

        if feature_best is None:
            node.type = NodeType.LEAF
            node.classification_class = mode(y).mode[0]
            return

        node.type = NodeType.INTERNAL

        node.feature_split = feature_best
        node.threshold = threshold_best

        X_left = X[np.logical_not(split)]
        X_right = X[split]

        y_left = y[np.logical_not(split)]
        y_right = y[split]

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            node.type = NodeType.LEAF
            node.classification_class = mode(y).mode[0]
            return

        node.left = Node(depth=node.depth + 1)
        node.right = Node(depth=node.depth + 1)

        self._fit_node(X_left, y_left, node.left)
        self._fit_node(X_right, y_right, node.right)

    def _predict_node(self, x: np.ndarray, node: Node) -> int:
        if node.type == NodeType.LEAF:
            return node.classification_class

        next_node = node.right if x[node.feature_split] < node.threshold else node.left

        return self._predict_node(x, next_node)

    def fit(
        self, X: np.ndarray, y: Union[np.ndarray, pd.Series]
    ) -> "DecisionTreeClassifier":
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self._fit_node(X, y, self._tree)

        return self

    def predict(self, X) -> np.ndarray:
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @staticmethod
    def __find_best_split(feature_vector, target_vector):
        def gini(t):
            threshold_mask = feature_vector >= t

            R_l = feature_vector[threshold_mask]
            p_l1 = np.count_nonzero(target_vector[threshold_mask]) / len(
                target_vector[threshold_mask]
            )
            p_l0 = 1 - p_l1

            R_r = feature_vector[~threshold_mask]
            p_r1 = np.count_nonzero(target_vector[~threshold_mask]) / len(
                target_vector[~threshold_mask]
            )
            p_r0 = 1 - p_r1

            left_part_score = -(len(R_l) / len(feature_vector)) * (
                1 - p_l1**2 - p_l0**2
            )
            right_part_score = -(len(R_r) / len(feature_vector)) * (
                1 - p_r1**2 - p_r0**2
            )

            return left_part_score + right_part_score

        sorted_feature_vector = np.sort(np.unique(feature_vector))
        thresholds = (sorted_feature_vector[:-1] + sorted_feature_vector[1:]) / 2
        ginis = np.vectorize(gini)(thresholds)
        threshold_best = thresholds[np.argmax(ginis)]
        gini_best = np.max(ginis)
        return threshold_best, gini_best
