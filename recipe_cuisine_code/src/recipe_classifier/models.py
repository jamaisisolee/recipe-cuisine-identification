"""Model pool for the recipe cuisine classification ensemble."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .exact_match import ExactMatchClassifier
from .features import PresenceTransformer, SupervisedRecipeFeatureAdder

RANDOM_STATE = 2026


class ZeroInflatedGaussianNB(BaseEstimator, ClassifierMixin):
    """Naive Bayes model for sparse non-negative TF-IDF features.

    Each feature is modeled as a point mass at zero for absent ingredients and
    a Gaussian density over ``log1p(x)`` for present ingredients.
    """

    def __init__(self, alpha: float = 1.0, var_smoothing: float = 1e-3):
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.sort(np.unique(y))
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.prior_ = np.array([np.log(np.mean(y == c)) for c in self.classes_])
        self.p0_ = np.zeros((n_classes, n_features))
        self.mu_ = np.zeros((n_classes, n_features))
        self.var_ = np.ones((n_classes, n_features))

        for class_index, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            alpha = self.alpha
            self.p0_[class_index] = ((X_class <= 0).sum(axis=0) + alpha) / (
                len(X_class) + 2.0 * alpha
            )

            for j in range(n_features):
                observed = np.log1p(X_class[X_class[:, j] > 0, j])
                if len(observed) > 1:
                    self.mu_[class_index, j] = observed.mean()
                    self.var_[class_index, j] = observed.var() + self.var_smoothing

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_log = np.log1p(X)
        is_zero = X <= 0

        log_likelihood = np.zeros((len(X), len(self.classes_)))
        for class_index in range(len(self.classes_)):
            p0 = np.clip(self.p0_[class_index], 1e-9, 1.0 - 1e-9)
            mu = self.mu_[class_index]
            var = np.maximum(self.var_[class_index], self.var_smoothing)

            gaussian_log_density = -0.5 * (
                np.log(2.0 * np.pi * var) + ((X_log - mu) ** 2) / var
            )
            log_likelihood[:, class_index] = (
                self.prior_[class_index]
                + (is_zero * np.log(p0)).sum(axis=1)
                + (~is_zero * (np.log(1.0 - p0) + gaussian_log_density)).sum(axis=1)
            )

        normalized = log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True)
        return np.exp(normalized)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)].astype(int)


def make_et():
    """ExactMatch + ExtraTrees fallback."""
    return ExactMatchClassifier(
        Pipeline(
            [
                ("feat", SupervisedRecipeFeatureAdder()),
                (
                    "m",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    )


def make_svm():
    """ExactMatch + RBF SVM fallback."""
    return ExactMatchClassifier(
        Pipeline(
            [
                ("feat", SupervisedRecipeFeatureAdder()),
                ("sc", StandardScaler()),
                (
                    "m",
                    SVC(
                        C=1.5,
                        gamma=0.01,
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    )


def make_zinb():
    """ExactMatch + zero-inflated Gaussian Naive Bayes fallback."""
    return ExactMatchClassifier(ZeroInflatedGaussianNB(alpha=1.0))


def make_lda():
    """ExactMatch + shrinkage LDA fallback."""
    return ExactMatchClassifier(
        Pipeline(
            [
                ("feat", SupervisedRecipeFeatureAdder()),
                ("sc", StandardScaler()),
                ("m", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    )


def make_bnb():
    """ExactMatch + Bernoulli Naive Bayes fallback."""
    return ExactMatchClassifier(
        Pipeline(
            [
                ("bin", PresenceTransformer()),
                ("m", BernoulliNB(alpha=0.2)),
            ]
        )
    )


def make_mlp():
    """ExactMatch + one-hidden-layer MLP fallback."""
    return ExactMatchClassifier(
        Pipeline(
            [
                ("feat", SupervisedRecipeFeatureAdder()),
                ("sc", StandardScaler()),
                (
                    "m",
                    MLPClassifier(
                        hidden_layer_sizes=(24,),
                        alpha=0.05,
                        learning_rate_init=0.004,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    )


BUILDERS = [make_et, make_svm, make_zinb, make_lda, make_bnb, make_mlp]
MODEL_NAMES = ["ExtraTrees", "SVM_RBF", "ZeroInflatedGNB", "LDA", "BernoulliNB", "MLP"]
