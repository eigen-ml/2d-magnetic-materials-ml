#!/usr/bin/env python3
"""Train and evaluate a composition-only magnetic-ordering classifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "combined_dataset.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
REQUIRED_COLUMNS = {"formula_normalized", "is_magnetic"}
RANDOM_STATE = 42


class TrainingDataError(ValueError):
    """Raised when prepared training data are missing or invalid."""


def get_element_fractions(formula: object) -> dict[str, float]:
    """Convert a formula to elemental fractions plus reduced-cell atom count."""
    try:
        composition = Composition(str(formula))
    except (TypeError, ValueError):
        return {}

    amounts = composition.get_el_amt_dict()
    total = float(sum(amounts.values()))
    if total <= 0:
        return {}
    features = {f"elem_{element}": float(amount / total) for element, amount in amounts.items()}
    features["natoms"] = float(composition.num_atoms)
    return features


def featurize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Any]]:
    """Create a deterministic feature table and retain valid source indices."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise TrainingDataError(
            f"prepared data are missing required column(s): {', '.join(missing)}"
        )

    rows: list[dict[str, float | int]] = []
    valid_indices: list[Any] = []
    for index, row in df.iterrows():
        features = get_element_fractions(row["formula_normalized"])
        if not features:
            continue
        label = row["is_magnetic"]
        try:
            numeric_label = float(label)
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(
                f"row {index!r} has a non-binary is_magnetic value: {label!r}"
            ) from exc
        if numeric_label not in (0.0, 1.0):
            raise TrainingDataError(
                f"row {index!r} has a non-binary is_magnetic value: {label!r}"
            )
        features["is_magnetic"] = int(numeric_label)
        rows.append(features)
        valid_indices.append(index)

    if not rows:
        raise TrainingDataError("no valid chemical formulas were available for training")

    feature_df = pd.DataFrame(rows).fillna(0.0)
    element_columns = sorted(
        column for column in feature_df.columns if column.startswith("elem_")
    )
    feature_df = feature_df[element_columns + ["natoms", "is_magnetic"]]
    return feature_df, valid_indices


def split_dataset(
    feature_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified 64/16/20 train/validation/test partitions."""
    y = feature_df["is_magnetic"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise TrainingDataError("training requires both magnetic and non-magnetic classes")
    X = feature_df.drop(columns="is_magnetic")

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=random_state,
        stratify=y_development,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 300,
    verbose: int = 0,
) -> tuple[GradientBoostingClassifier, dict[str, float]]:
    """Fit the class-weighted Gradient Boosting classifier."""
    class_counts = Counter(int(value) for value in y_train)
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise TrainingDataError("the training split must contain both classes")

    class_weight_factor = 0.35
    imbalance_ratio = class_counts[0] / class_counts[1]
    magnetic_weight = imbalance_ratio * class_weight_factor
    sample_weights = np.where(y_train.to_numpy() == 1, magnetic_weight, 1.0)

    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.06,
        max_depth=8,
        subsample=0.85,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        verbose=verbose,
    )
    classifier.fit(X_train, y_train, sample_weight=sample_weights)
    return classifier, {
        "class_weight_factor": class_weight_factor,
        "magnetic_sample_weight": float(magnetic_weight),
    }


def find_operating_threshold(
    classifier: Any,
    X_validation: pd.DataFrame | np.ndarray,
    y_validation: pd.Series | np.ndarray,
    *,
    target_recall: float = 0.90,
    minimum_precision: float = 0.80,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Select a threshold on validation data, preferring feasible maximum F1."""
    scores = np.asarray(classifier.predict_proba(X_validation))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validation, scores)
    if thresholds.size == 0:
        raise TrainingDataError("validation scores do not define a usable threshold")

    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    denominator = precision_at_threshold + recall_at_threshold
    f1 = np.divide(
        2 * precision_at_threshold * recall_at_threshold,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    feasible = (recall_at_threshold >= target_recall) & (
        precision_at_threshold >= minimum_precision
    )
    if feasible.any():
        candidate_indices = np.flatnonzero(feasible)
        selected_index = int(candidate_indices[np.argmax(f1[candidate_indices])])
        constraint_status = "met"
    else:
        selected_index = int(np.argmax(f1))
        constraint_status = "not_met_fallback_to_max_f1"

    threshold = float(thresholds[selected_index])
    selection = {
        "selected_on": "validation_split",
        "target_recall": target_recall,
        "minimum_precision": minimum_precision,
        "constraint_status": constraint_status,
        "validation_precision": float(precision_at_threshold[selected_index]),
        "validation_recall": float(recall_at_threshold[selected_index]),
        "validation_f1": float(f1[selected_index]),
    }
    return threshold, scores, selection


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_predicted: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Return binary classification metrics for an untouched test split."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Capture the direct dependency versions used for training."""
    packages = ["numpy", "pandas", "pymatgen", "scikit-learn", "joblib"]
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def save_artifacts(
    classifier: GradientBoostingClassifier,
    threshold: float,
    feature_names: list[str],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist the trained model and human-readable companion artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "model.joblib")
    np.save(output_dir / "threshold.npy", threshold)
    pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True]).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run training and return a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.data.is_file():
        print(
            f"ERROR: prepared dataset not found: {args.data}. "
            "Run scripts/02_prepare_data.py first or pass --data.",
            file=sys.stderr,
        )
        return 2
    if args.n_estimators < 1:
        print("ERROR: --n-estimators must be at least 1", file=sys.stderr)
        return 2

    try:
        raw = pd.read_csv(args.data)
        feature_df, valid_indices = featurize_dataset(raw)
        X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(
            feature_df
        )
        classifier, weighting = train_model(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            verbose=int(args.verbose),
        )
        threshold, _, threshold_selection = find_operating_threshold(
            classifier, X_validation, y_validation
        )
        test_scores = classifier.predict_proba(X_test)[:, 1]
        test_predictions = (test_scores >= threshold).astype(int)
        metrics = evaluate_model(y_test, test_predictions, test_scores)
    except (OSError, TrainingDataError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    full_counts = Counter(int(value) for value in feature_df["is_magnetic"])
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_version": "v3.0_v2db_composition_only",
        "scientific_scope": {
            "input": "reduced chemical composition only",
            "target": "V2DB magnetic label",
            "structural_features_used": False,
            "scores_calibrated": False,
        },
        "data": {
            "sha256": sha256_file(args.data),
            "rows_read": int(len(raw)),
            "rows_featurized": int(len(valid_indices)),
            "class_distribution": {
                "non_magnetic": int(full_counts[0]),
                "magnetic": int(full_counts[1]),
            },
        },
        "split": {
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
        },
        "model": {
            "class": "sklearn.ensemble.GradientBoostingClassifier",
            "parameters": classifier.get_params(),
            "weighting": weighting,
        },
        "threshold": threshold,
        "threshold_selection": threshold_selection,
        "test_metrics": metrics,
        "n_features": int(len(X_train.columns)),
        "feature_names": X_train.columns.tolist(),
        "software_versions": package_versions(),
    }
    save_artifacts(
        classifier,
        threshold,
        X_train.columns.tolist(),
        metadata,
        args.model_dir,
    )

    print(
        f"Split: {len(X_train):,} train / {len(X_validation):,} validation / "
        f"{len(X_test):,} test"
    )
    print(f"Validation-selected threshold: {threshold:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test F1: {metrics['f1_score']:.4f}")
    print(f"Wrote model artefacts to: {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
