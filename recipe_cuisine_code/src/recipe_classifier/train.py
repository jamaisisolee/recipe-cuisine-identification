"""Training and blending pipeline for recipe cuisine classification.

Expected input format:
- train CSV: first column is the label, remaining columns are 40 TF-IDF features
- test CSV: 40 TF-IDF feature columns and no label column

The original private university data is intentionally not included.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .models import BUILDERS, MODEL_NAMES, RANDOM_STATE


@dataclass
class BlendResult:
    """A single candidate blend and threshold."""

    weight_index: int
    error: float
    threshold: float
    weights: np.ndarray


@dataclass
class SubmissionVariant:
    """Metadata for one written submission variant."""

    variant_number: int
    path: Path
    error: float
    threshold: float
    weights: np.ndarray


def load_recipe_data(train_csv: str | Path, test_csv: str | Path | None = None):
    """Load train/test CSV files in the challenge format."""
    train_df = pd.read_csv(train_csv)
    X = train_df.iloc[:, 1:].to_numpy(dtype=float)
    y = train_df.iloc[:, 0].to_numpy(dtype=int)

    X_test = None
    feature_names = list(train_df.columns[1:])
    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        X_test = test_df.to_numpy(dtype=float)

    return X, y, X_test, feature_names


def proba_for_class(model, X, target_class: int = 2):
    """Return predicted probability for a specific class label."""
    prob = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    class_index = np.where(classes == target_class)[0][0]
    return prob[:, class_index]


def compute_oof_and_test_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None,
    builders: Iterable[Callable] = BUILDERS,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    """Compute out-of-fold and test probabilities for each model builder."""
    builders = list(builders)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    n_models = len(builders)
    oof = np.zeros((len(X), n_models))
    test = None if X_test is None else np.zeros((len(X_test), n_models))

    for model_index, build in enumerate(builders):
        oof_prob_italian = np.zeros(len(X))

        if X_test is not None:
            test_prob_by_fold = np.zeros((len(X_test), len(folds)))

        for fold_index, (train_idx, valid_idx) in enumerate(folds):
            model = build()
            model.fit(X[train_idx], y[train_idx])

            oof_prob_italian[valid_idx] = proba_for_class(model, X[valid_idx], 2)
            if X_test is not None:
                test_prob_by_fold[:, fold_index] = proba_for_class(model, X_test, 2)

        oof[:, model_index] = oof_prob_italian
        model_error = (np.where(oof_prob_italian >= 0.5, 2, 1) != y).mean()
        model_name = MODEL_NAMES[model_index] if model_index < len(MODEL_NAMES) else model_index
        print(f"{model_name} OOF error: {model_error:.4f}")

        if X_test is not None:
            full_model = build()
            full_model.fit(X, y)
            full_prob = proba_for_class(full_model, X_test, 2)
            test[:, model_index] = 0.5 * test_prob_by_fold.mean(axis=1) + 0.5 * full_prob

    return oof, test


def search_blend_weights(
    oof: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
    n_random_candidates: int = 3494,
    thresholds: np.ndarray | None = None,
):
    """Search sparse Dirichlet model weights and probability thresholds."""
    rng = np.random.default_rng(random_state)
    n_models = oof.shape[1]

    if thresholds is None:
        thresholds = np.linspace(0.38, 0.62, 97)

    candidates = [np.eye(n_models)[i] for i in range(n_models)]
    candidates.append(np.ones(n_models) / n_models)

    for _ in range(n_random_candidates):
        subset_size = int(rng.choice([2, 3, 4, 5, 6], p=[0.20, 0.30, 0.25, 0.15, 0.10]))
        subset_size = min(subset_size, n_models)
        subset = rng.choice(n_models, subset_size, replace=False)
        weights = np.zeros(n_models)
        weights[subset] = rng.dirichlet(np.ones(subset_size) * 0.7)
        candidates.append(weights)

    rows = []
    for weight_index, weights in enumerate(candidates):
        blend = oof @ weights
        for threshold in thresholds:
            pred = np.where(blend >= threshold, 2, 1).astype(int)
            error = 1.0 - accuracy_score(y, pred)
            rows.append(
                BlendResult(
                    weight_index=weight_index,
                    error=float(error),
                    threshold=float(threshold),
                    weights=weights.copy(),
                )
            )

    rows.sort(key=lambda row: (row.error, row.threshold))
    return rows


def hamming_distance(a, b) -> int:
    """Number of differing labels between two prediction vectors."""
    return int((np.asarray(a) != np.asarray(b)).sum())


def write_diverse_submissions(
    blend_rows: list[BlendResult],
    test_probabilities: np.ndarray,
    output_dir: str | Path,
    min_hamming_distance: int = 4,
    max_variants: int = 8,
    prefix: str = "submission_30_unanchored_wide_blend",
):
    """Write diverse submission variants using the searched blend rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_predictions = []
    variants: list[SubmissionVariant] = []
    variant_number = 1

    for row in blend_rows:
        test_blend = test_probabilities @ row.weights
        labels = np.where(test_blend >= row.threshold, 2, 1).astype(int)

        if any(
            hamming_distance(labels, old) < min_hamming_distance
            for old in written_predictions
        ):
            continue

        path = output_dir / f"{prefix}_{variant_number:02d}.csv"
        pd.DataFrame({"y": labels}).to_csv(path, index=False)

        written_predictions.append(labels.copy())
        variants.append(
            SubmissionVariant(
                variant_number=variant_number,
                path=path,
                error=row.error,
                threshold=row.threshold,
                weights=row.weights.copy(),
            )
        )

        variant_number += 1
        if variant_number > max_variants:
            break

    return variants


def run_pipeline(
    train_csv: str | Path,
    test_csv: str | Path | None = None,
    output_dir: str | Path = "submissions",
):
    """Run the full model-pool, OOF, blend-search, and submission pipeline."""
    X, y, X_test, _ = load_recipe_data(train_csv, test_csv)
    oof, test = compute_oof_and_test_probabilities(X, y, X_test)
    blend_rows = search_blend_weights(oof, y)

    best = blend_rows[0]
    print(
        "Best OOF blend: "
        f"error={best.error:.4f}, accuracy={1.0 - best.error:.4f}, "
        f"threshold={best.threshold:.3f}, weights={np.round(best.weights, 4)}"
    )

    variants = []
    if test is not None:
        variants = write_diverse_submissions(blend_rows, test, output_dir)
        if variants:
            print(f"Wrote {len(variants)} submission variants to {Path(output_dir)}")
            print(f"Final variant path: {variants[-1].path}")

    return oof, test, blend_rows, variants


def parse_args():
    parser = argparse.ArgumentParser(description="Recipe cuisine classification pipeline")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", default=None, help="Path to test.csv")
    parser.add_argument("--out", default="submissions", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.train, args.test, args.out)


if __name__ == "__main__":
    main()
