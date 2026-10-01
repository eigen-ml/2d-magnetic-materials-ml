#!/usr/bin/env python3
"""Compare random and chemistry-held-out cross-validation for the same model.

A random split lets compositions from the same chemical system (for example
CrI3 and CrI2, both "Cr-I") land on both sides of the split. This script
measures how much of the headline test score survives when that is prevented:

* random:  stratified 5-fold CV (the optimistic reference)
* chemsys: 5-fold CV grouped by chemical system, so every element combination
           is seen either only in training or only in testing
* leave-one-element-out: all compositions containing one element are held
           out; the model never sees that element during training

Every fold repeats the main pipeline: fit on an inner training split, choose
the operating threshold on an inner validation split, score the held-out fold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "combined_dataset.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_ELEMENTS = ("V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ti")
N_SPLITS = 5


def _load_train_module():
    path = REPO_ROOT / "scripts" / "03_train_model.py"
    spec = importlib.util.spec_from_file_location("train_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load_train_module()


def chemical_system(feature_df: pd.DataFrame) -> pd.Series:
    """Return the sorted element set of each row, e.g. "Cr-I"."""
    element_columns = [c for c in feature_df.columns if c.startswith("elem_")]
    present = feature_df[element_columns].to_numpy() > 0
    symbols = np.array([c.removeprefix("elem_") for c in element_columns])
    return pd.Series(
        ["-".join(sorted(symbols[row])) for row in present], index=feature_df.index
    )


def run_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    test_index: np.ndarray,
    n_estimators: int,
) -> dict[str, Any]:
    """Fit, select a threshold on inner validation data, and score one fold."""
    X_outer, y_outer = X.iloc[train_index], y.iloc[train_index]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_outer,
        y_outer,
        test_size=0.20,
        random_state=train.RANDOM_STATE,
        stratify=y_outer,
    )
    classifier, _ = train.train_model(X_fit, y_fit, n_estimators=n_estimators)
    threshold, _, _ = train.find_operating_threshold(classifier, X_val, y_val)
    scores = classifier.predict_proba(X.iloc[test_index])[:, 1]
    return {"test_index": test_index, "scores": scores, "threshold": threshold}


def summarize(y_true: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray) -> dict:
    """Threshold-free and thresholded metrics for one set of held-out predictions."""
    predicted = (scores >= threshold).astype(int)
    metrics = train.evaluate_model(y_true, predicted, scores)
    metrics["average_precision"] = float(average_precision_score(y_true, scores))
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["n"] = int(len(y_true))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--elements", nargs="+", default=list(DEFAULT_ELEMENTS))
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds (joblib)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data.is_file():
        print(
            f"ERROR: prepared dataset not found: {args.data}. "
            "Run scripts/02_prepare_data.py first.",
            file=sys.stderr,
        )
        return 2

    feature_df, valid_indices = train.featurize_dataset(pd.read_csv(args.data))
    raw = pd.read_csv(args.data).loc[valid_indices]
    y = feature_df["is_magnetic"].astype(int).reset_index(drop=True)
    X = feature_df.drop(columns="is_magnetic").reset_index(drop=True)
    groups = chemical_system(X)

    jobs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    random_cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=train.RANDOM_STATE)
    for k, (tr, te) in enumerate(random_cv.split(X, y)):
        jobs.append(("random", f"fold{k}", tr, te))
    grouped_cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=train.RANDOM_STATE)
    for k, (tr, te) in enumerate(grouped_cv.split(X, y, groups)):
        jobs.append(("chemsys", f"fold{k}", tr, te))
    for element in args.elements:
        column = f"elem_{element}"
        if column not in X.columns:
            print(f"WARNING: {element} not in dataset, skipped", file=sys.stderr)
            continue
        held_out = X[column].to_numpy() > 0
        jobs.append(("leave_element_out", element, np.flatnonzero(~held_out), np.flatnonzero(held_out)))

    print(f"Running {len(jobs)} fits on {len(X):,} compositions ...")
    outputs = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(X, y, tr, te, args.n_estimators) for _, _, tr, te in jobs
    )

    fold_rows: list[dict[str, Any]] = []
    oof = pd.DataFrame(
        {"formula": raw["formula_normalized"].to_numpy(), "chemsys": groups, "is_magnetic": y}
    )
    summary: dict[str, Any] = {
        "n_compositions": int(len(X)),
        "n_chemical_systems": int(groups.nunique()),
        "median_compositions_per_chemical_system": float(groups.value_counts().median()),
    }
    # Context baseline, no training: "contains a 3d element from V to Ni".
    rule = X[[f"elem_{e}" for e in ("V", "Cr", "Mn", "Fe", "Co", "Ni")]].to_numpy().sum(axis=1) > 0
    summary["rule_contains_V_to_Ni"] = train.evaluate_model(y, rule.astype(int), rule.astype(float))
    for scheme in ("random", "chemsys"):
        scores = np.full(len(X), np.nan)
        thresholds = np.full(len(X), np.nan)
        for (s, name, _, _), out in zip(jobs, outputs):
            if s != scheme:
                continue
            scores[out["test_index"]] = out["scores"]
            thresholds[out["test_index"]] = out["threshold"]
            fold_rows.append(
                {"scheme": s, "fold": name, "threshold": out["threshold"]}
                | summarize(y.iloc[out["test_index"]].to_numpy(), out["scores"], out["threshold"])
            )
        oof[f"{scheme}_score"] = scores
        oof[f"{scheme}_threshold"] = thresholds
        folds = [r for r in fold_rows if r["scheme"] == scheme]
        summary[scheme] = {
            "pooled": summarize(y.to_numpy(), scores, thresholds),
            "roc_auc_mean": float(np.mean([r["roc_auc"] for r in folds])),
            "roc_auc_std": float(np.std([r["roc_auc"] for r in folds], ddof=1)),
            "average_precision_mean": float(np.mean([r["average_precision"] for r in folds])),
            "average_precision_std": float(np.std([r["average_precision"] for r in folds], ddof=1)),
            "recall_mean": float(np.mean([r["recall"] for r in folds])),
            "precision_mean": float(np.mean([r["precision"] for r in folds])),
        }

    summary["leave_element_out"] = {}
    loeo_scores: list[pd.DataFrame] = []
    for (s, name, _, te), out in zip(jobs, outputs):
        if s != "leave_element_out":
            continue
        y_te = y.iloc[te].to_numpy()
        loeo_scores.append(
            pd.DataFrame({"held_out_element": name, "is_magnetic": y_te, "score": out["scores"]})
        )
        row = {"scheme": s, "fold": name, "threshold": out["threshold"]}
        if len(np.unique(y_te)) < 2:
            row |= {"n": int(len(y_te)), "note": "single class in held-out set"}
        else:
            row |= summarize(y_te, out["scores"], out["threshold"])
        fold_rows.append(row)
        summary["leave_element_out"][name] = {
            k: row[k] for k in ("n", "positive_rate", "roc_auc", "average_precision", "precision", "recall")
            if k in row
        }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "grouped_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(fold_rows).drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        args.results_dir / "grouped_validation_folds.csv", index=False
    )
    oof.to_csv(args.results_dir / "cv_out_of_fold_scores.csv.gz", index=False)
    if loeo_scores:
        pd.concat(loeo_scores).to_csv(
            args.results_dir / "leave_element_out_scores.csv.gz", index=False
        )

    rule_metrics = summary["rule_contains_V_to_Ni"]
    print(
        f"rule 'contains V-Ni': recall {rule_metrics['recall']:.3f} "
        f"precision {rule_metrics['precision']:.3f}"
    )
    for scheme in ("random", "chemsys"):
        s = summary[scheme]
        print(
            f"{scheme:8s} ROC-AUC {s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f} | "
            f"AP {s['average_precision_mean']:.3f} ± {s['average_precision_std']:.3f} | "
            f"recall {s['recall_mean']:.3f} | precision {s['precision_mean']:.3f}"
        )
    for element, s in summary["leave_element_out"].items():
        if "roc_auc" in s:
            print(
                f"without {element:2s}: n={s['n']:6,} pos={s['positive_rate']:.2f} "
                f"ROC-AUC {s['roc_auc']:.3f} AP {s['average_precision']:.3f} "
                f"recall {s['recall']:.3f} precision {s['precision']:.3f}"
            )
    print(f"Wrote results to: {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
