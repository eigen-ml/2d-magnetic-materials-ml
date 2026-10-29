#!/usr/bin/env python3
"""Compare trained models on the C2DB part of one test split.

Every model is scored on the same C2DB test formulas (the test partition of the
reference dataset) against the same labels, next to the metal rule. The run-to-run
spread of C2DB-only training is measured by permuting the training rows, because
the boosting subsample picks rows by position.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPO_ROOT / "data" / "v3" / "combined_net_moment.csv"
DEFAULT_LABELS = REPO_ROOT / "data" / "v4" / "c2db_lowest_hull.csv"
DEFAULT_OTHER_LABELS = REPO_ROOT / "data" / "v4" / "c2db_thesis_table.csv"
DEFAULT_MODELS = (
    REPO_ROOT / "models" / "v3",
    REPO_ROOT / "models" / "v4",
    REPO_ROOT / "models" / "v3_experiments" / "c2db_thesis_table",
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "v4" / "comparison.json"
RULE_METALS = ("Mn", "Cr", "Fe", "Co", "V", "Ni")


def load_script(filename: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"compare_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_script("03_train_model.py")
discover = load_script("04_discover.py")


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = train.evaluate_model(y, (scores >= threshold).astype(int), scores)
    top = train.band_precision(y, scores)[0]
    result.update(threshold=float(threshold), rows_at_0_98=top["rows"], precision_at_0_98=top["precision"])
    return result


def score(model_dir: Path, formulas: list[str]) -> tuple[np.ndarray, float]:
    model, threshold, metadata = discover.load_model(model_dir)
    names = metadata["feature_names"]
    rows = [discover.featurize(formula, names) for formula in formulas]
    if any(row is None for row in rows):
        raise ValueError(f"{model_dir.name} cannot featurize every test formula")
    return model.predict_proba(pd.DataFrame(rows, columns=names))[:, 1], threshold


def has_metal(formula: str, metals: tuple[str, ...]) -> bool:
    return bool(set(Composition(formula).as_dict()) & set(metals))


def order_spread(labels: pd.DataFrame, partition: dict, truth: pd.Series, permutations: int) -> dict:
    """Train C2DB-only models with permuted training rows; summarise test ROC-AUC and F1."""
    features, valid = train.featurize_dataset(labels)
    features = train.select_features(features, "elements")
    formulas = pd.Series(labels.loc[valid, "formula_normalized"].to_numpy(), index=features.index)
    X_train, X_val, X_test, y_train, y_val, _ = train.split_like(features, formulas, partition)
    y_truth = truth.loc[formulas.loc[X_test.index]].to_numpy()
    runs = []
    for seed in range(permutations):
        order = np.random.default_rng(seed).permutation(len(X_train))
        model, _ = train.train_model(X_train.iloc[order], y_train.iloc[order])
        calibrated = train.calibrate(model, X_val, y_val)
        threshold, _, _ = train.find_operating_threshold(calibrated, X_val, y_val)
        result = metrics(y_truth, calibrated.predict_proba(X_test)[:, 1], threshold)
        runs.append((result["roc_auc"], result["f1_score"]))
    values = np.array(runs)
    return {
        name: {
            "mean": round(float(column.mean()), 4),
            "sd": round(float(column.std(ddof=1)), 4),
            "min": round(float(column.min()), 4),
            "max": round(float(column.max()), 4),
        }
        for name, column in zip(("roc_auc", "f1_score"), values.T)
    } | {"permutations": permutations}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="dataset that defines the split")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="C2DB labels used as truth")
    parser.add_argument("--other-labels", type=Path, default=DEFAULT_OTHER_LABELS)
    parser.add_argument("--models", type=Path, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        partition = train.reference_partition(pd.read_csv(args.reference))
        labels = pd.read_csv(args.labels)
        other = pd.read_csv(args.other_labels)
        truth = labels.set_index("formula_normalized")["is_magnetic"].astype(int)
        alternative = other.set_index("formula_normalized")["is_magnetic"].astype(int)
        test = sorted(f for f in truth.index if partition.get(f, ("",))[0] == "test")
        y = truth.loc[test].to_numpy()
        y_other = alternative.reindex(test)

        results: dict = {"test_formulas": len(test), "magnetic_share": round(float(y.mean()), 4)}
        results["label_sets"] = {
            "truth": args.labels.name,
            "other": args.other_labels.name,
            "test_formulas_where_they_differ": int((y_other.to_numpy() != y).sum()),
        }
        for model_dir in args.models:
            scores, threshold = score(model_dir, test)
            results.setdefault("models", {})[model_dir.name] = {
                "truth": metrics(y, scores, threshold),
                "other": metrics(y_other.to_numpy(), scores, threshold),
            }
        rule = np.array([has_metal(f, RULE_METALS) for f in test], dtype=float)
        results["rule_contains_" + "_".join(RULE_METALS)] = train.evaluate_model(y, rule.astype(int), rule)
        results["magnetic_share_by_metal"] = {
            metal: {
                "formulas": int(mask.sum()),
                "magnetic_share": round(float(y[mask].mean()), 3),
            }
            for metal in RULE_METALS
            for mask in [np.array([has_metal(f, (metal,)) for f in test])]
        }
        if args.permutations > 1:
            results["training_order_spread"] = {
                path.name: order_spread(pd.read_csv(path), partition, truth, args.permutations)
                for path in (args.labels, args.other_labels)
            }
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"C2DB test formulas: {len(test):,} ({results['magnetic_share']:.1%} magnetic)")
    for name, result in results["models"].items():
        r = result["truth"]
        print(f"{name}: ROC-AUC {r['roc_auc']:.3f}, precision {r['precision']:.3f}, "
              f"recall {r['recall']:.3f}, F1 {r['f1_score']:.3f}")
    for name, spread in results.get("training_order_spread", {}).items():
        print(f"order spread {name}: ROC-AUC {spread['roc_auc']['mean']:.3f} ± {spread['roc_auc']['sd']:.3f}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
