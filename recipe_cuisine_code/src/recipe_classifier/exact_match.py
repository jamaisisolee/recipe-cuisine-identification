"""Exact-match classifier wrapper.

The wrapper memorizes duplicated training feature vectors and predicts their
majority class directly. For feature vectors that were not observed during
training, it delegates to a fallback estimator.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def row_key(row):
    """Convert a feature row into a hashable exact-match key."""
    return tuple(np.asarray(row, dtype=float))


class ExactMatchClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that uses exact vector lookup before a learned fallback model.

    Parameters
    ----------
    fallback_estimator:
        Any fitted scikit-learn-compatible estimator with ``predict_proba``.
    tie_policy:
        How to handle exact-match training groups with tied class counts.
        ``"fallback"`` leaves tied rows to the fallback estimator. ``"first"``
        uses ``Counter.most_common(1)``, matching the shortest code version in
        the project report.
    """

    def __init__(self, fallback_estimator, tie_policy: str = "fallback"):
        self.fallback_estimator = fallback_estimator
        self.tie_policy = tie_policy

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.sort(np.unique(y))
        self.fallback_ = clone(self.fallback_estimator)
        self.fallback_.fit(X, y)

        groups = defaultdict(list)
        for row, label in zip(X, y):
            groups[row_key(row)].append(int(label))

        self.map_ = {}
        for key, labels in groups.items():
            counts = Counter(labels).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                if self.tie_policy == "fallback":
                    self.map_[key] = None
                elif self.tie_policy == "first":
                    self.map_[key] = counts[0][0]
                else:
                    raise ValueError("tie_policy must be 'fallback' or 'first'")
            else:
                self.map_[key] = counts[0][0]

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        fallback_prob = self.fallback_.predict_proba(X).copy()
        fallback_classes = np.asarray(getattr(self.fallback_, "classes_", self.classes_))

        out = np.zeros((len(X), len(self.classes_)))
        for j, cls in enumerate(fallback_classes):
            class_position = np.where(self.classes_ == cls)[0][0]
            out[:, class_position] = fallback_prob[:, j]

        for i, row in enumerate(X):
            label = self.map_.get(row_key(row))
            if label is not None:
                out[i] = 0.0
                class_position = np.where(self.classes_ == label)[0][0]
                out[i, class_position] = 1.0

        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)].astype(int)
