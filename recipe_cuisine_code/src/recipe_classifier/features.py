"""Feature engineering for the recipe cuisine classification project.

The original challenge used 40 non-negative TF-IDF ingredient features.
These transformers add presence indicators, row-level summary statistics,
and fold-learned supervised ingredient statistics.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Raw TF-IDF + 40 binary presence indicators + 5 row summaries.

    For an input matrix with 40 TF-IDF columns, this returns 85 columns:
    - 40 original TF-IDF features
    - 40 binary ingredient-presence indicators
    - ingredient count, TF-IDF sum, TF-IDF max, nonzero mean, max/sum ratio
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        presence = (X > 0).astype(float)

        ingredient_count = presence.sum(axis=1, keepdims=True)
        tfidf_sum = X.sum(axis=1, keepdims=True)
        tfidf_max = X.max(axis=1, keepdims=True)

        nonzero_mean = np.divide(
            tfidf_sum,
            ingredient_count,
            out=np.zeros_like(tfidf_sum),
            where=ingredient_count > 0,
        )
        concentration = np.divide(
            tfidf_max,
            tfidf_sum,
            out=np.zeros_like(tfidf_sum),
            where=tfidf_sum > 0,
        )

        return np.hstack(
            [
                X,
                presence,
                ingredient_count,
                tfidf_sum,
                tfidf_max,
                nonzero_mean,
                concentration,
            ]
        )


class SupervisedRecipeFeatureAdder(BaseEstimator, TransformerMixin):
    """FeatureAdder plus supervised rarity and ingredient log-odds features.

    The supervised statistics are learned during ``fit``. When this transformer
    is used inside a scikit-learn Pipeline during cross-validation, those
    statistics are computed from the training fold only, preventing label
    leakage.
    """

    def __init__(self, smoothing: float = 2.0):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        presence = (X > 0).astype(float)
        n_rows = X.shape[0]

        self.freq_ = (presence.sum(axis=0) + 1.0) / (n_rows + 2.0)
        self.rarity_ = -np.log(self.freq_)

        if y is not None:
            y = np.asarray(y, dtype=int)
            alpha = self.smoothing

            italian_presence = presence[y == 2]
            american_presence = presence[y == 1]

            p_italian = (italian_presence.sum(axis=0) + alpha) / (
                len(italian_presence) + 2.0 * alpha
            )
            p_american = (american_presence.sum(axis=0) + alpha) / (
                len(american_presence) + 2.0 * alpha
            )
            self.log_odds_ = np.log(p_italian) - np.log(p_american)
        else:
            self.log_odds_ = np.zeros(X.shape[1])

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        presence = (X > 0).astype(float)

        base_features = FeatureAdder().fit_transform(X)
        ingredient_count = presence.sum(axis=1, keepdims=True)

        rarity_by_ingredient = presence * self.rarity_
        avg_rarity = np.divide(
            rarity_by_ingredient.sum(axis=1, keepdims=True),
            ingredient_count,
            out=np.zeros_like(ingredient_count),
            where=ingredient_count > 0,
        )
        max_rarity = rarity_by_ingredient.max(axis=1, keepdims=True)

        log_odds_by_ingredient = presence * self.log_odds_
        log_odds_sum = log_odds_by_ingredient.sum(axis=1, keepdims=True)
        positive_log_odds_sum = np.maximum(log_odds_by_ingredient, 0).sum(
            axis=1, keepdims=True
        )
        negative_log_odds_sum = np.minimum(log_odds_by_ingredient, 0).sum(
            axis=1, keepdims=True
        )

        return np.hstack(
            [
                base_features,
                avg_rarity,
                max_rarity,
                log_odds_sum,
                positive_log_odds_sum,
                negative_log_odds_sum,
            ]
        )


class PresenceTransformer(BaseEstimator, TransformerMixin):
    """Binarize TF-IDF features: 1 if ingredient is present, else 0."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) > 0).astype(float)
