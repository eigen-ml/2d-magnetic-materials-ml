#!/usr/bin/env python3
"""Score enumerated compositions that are absent from a reference formula set."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN = REPO_ROOT / "data" / "known_materials.txt"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

MAGNETIC_METALS = ("Mn", "Cr", "Fe", "Co", "Ni", "V", "Cu", "Ti")
ANIONS = ("S", "Se", "Te", "O", "Cl", "Br", "I", "F")
OTHER_METALS = ("Zr", "Hf", "Nb", "Ta", "Mo", "W", "Ru", "Pd", "Pt")


class ScreeningError(RuntimeError):
    """Raised when screening inputs or model artefacts are unusable."""


def load_known_materials(path: str | Path) -> set[str]:
    """Load the reduced-formula reference set."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"known-material reference not found: {source}. "
            "Run scripts/02_prepare_data.py first or pass --known-materials."
        )
    known: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        reduced = normalize_formula(line.strip())
        if reduced is None:
            raise ScreeningError(
                f"{source}: line {line_number} is not a valid chemical formula"
            )
        known.add(reduced)
    if not known:
        raise ScreeningError(f"known-material reference is empty: {source}")
    return known


def load_model(model_dir: str | Path) -> tuple[Any, float, dict[str, Any]]:
    """Load trusted local model artefacts and validate their basic metadata."""
    directory = Path(model_dir)
    required = {
        "model": directory / "model.joblib",
        "threshold": directory / "threshold.npy",
        "metadata": directory / "metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing model artefact(s): " + ", ".join(missing) + ". Run training first."
        )

    try:
        metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
        threshold = float(np.load(required["threshold"], allow_pickle=False))
        # joblib uses pickle internally. Only load artefacts produced locally or
        # obtained from a trusted source.
        classifier = joblib.load(required["model"])
    except Exception as exc:  # Convert version/pickle failures into a useful CLI error.
        raise ScreeningError(
            "model artefacts could not be loaded; verify that they are trusted and "
            "were created with compatible dependency versions"
        ) from exc

    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ScreeningError("metadata.json does not contain a non-empty feature_names list")
    if not 0.0 <= threshold <= 1.0:
        raise ScreeningError(f"invalid decision threshold: {threshold}")
    return classifier, threshold, metadata


def generate_candidates(
    magnetic_metals: Sequence[str],
    anions: Sequence[str],
    other_metals: Sequence[str] | None = None,
) -> list[str]:
    """Enumerate the project's rule-based composition search space deterministically.

    Templates with an unequal metal ratio (A2BX4, AB2X4) are generated for both
    orderings of the metals. The thesis used ordered pairs only, so it produced
    Mn2CrX4 but never Cr2MnX4, which biased the shortlist towards Mn.
    """
    candidates: set[str] = set()

    for metal, anion in product(magnetic_metals, anions):
        candidates.update(
            {
                f"{metal}{anion}",
                f"{metal}{anion}2",
                f"{metal}{anion}3",
                f"{metal}2{anion}3",
            }
        )

    for (first_metal, second_metal), anion in product(
        combinations(magnetic_metals, 2), anions
    ):
        candidates.update(
            {
                f"{first_metal}{second_metal}{anion}3",
                f"{first_metal}{second_metal}{anion}4",
            }
        )
    for (first_metal, second_metal), anion in product(
        permutations(magnetic_metals, 2), anions
    ):
        candidates.add(f"{first_metal}2{second_metal}{anion}4")

    if other_metals:
        for metal, other_metal, anion in product(
            magnetic_metals, other_metals, anions
        ):
            candidates.update(
                {
                    f"{metal}{other_metal}{anion}3",
                    f"{metal}{other_metal}2{anion}4",
                    f"{metal}2{other_metal}{anion}4",
                }
            )

    for metal, (first_anion, second_anion) in product(
        magnetic_metals, combinations(anions, 2)
    ):
        candidates.update(
            {
                f"{metal}{first_anion}2{second_anion}",
                f"{metal}{first_anion}{second_anion}2",
            }
        )
    return sorted(candidates)


def normalize_formula(formula: object) -> str | None:
    """Return a reduced chemical formula, or ``None`` when parsing fails."""
    try:
        return Composition(str(formula)).reduced_formula
    except (TypeError, ValueError):
        return None


def featurize(formula: str, feature_names: Sequence[str]) -> dict[str, float] | None:
    """Map one composition onto the exact feature schema used during training."""
    try:
        composition = Composition(formula)
    except (TypeError, ValueError):
        return None

    amounts = composition.get_el_amt_dict()
    total = float(sum(amounts.values()))
    if total <= 0:
        return None
    available_element_features = {f"elem_{element}" for element in amounts}
    if not available_element_features.issubset(feature_names):
        return None
    available = {
        **{f"elem_{element}": float(amount / total) for element, amount in amounts.items()},
        "natoms": float(composition.num_atoms),
    }
    return {feature: available.get(feature, 0.0) for feature in feature_names}


def score_band(score: float) -> str:
    """Return a descriptive score band (calibrated only if the model says so)."""
    if score >= 0.98:
        return "0.98-1.00"
    if score >= 0.95:
        return "0.95-0.98"
    if score >= 0.90:
        return "0.90-0.95"
    if score >= 0.80:
        return "0.80-0.90"
    return "threshold-0.80"


def positive_class_index(classifier: Any) -> int:
    """Locate the positive class in a scikit-learn compatible classifier."""
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        raise ScreeningError("the loaded classifier does not expose binary class 1")
    return classes.index(1)


def raw_scores(classifier: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the uncalibrated positive-class score.

    Isotonic calibration maps many top scores to the same value, so ties are
    broken by the score of the underlying boosted trees.
    """
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if not calibrated:
        return np.asarray(classifier.predict_proba(X))[:, positive_class_index(classifier)]
    base = calibrated[0].estimator
    base = getattr(base, "estimator", base)  # unwrap FrozenEstimator
    return np.asarray(base.predict_proba(X))[:, positive_class_index(base)]


def normalize_candidates(candidates: Iterable[str]) -> tuple[list[str], int]:
    """Normalize and deduplicate candidates, returning invalid-input count."""
    normalized: set[str] = set()
    invalid = 0
    for formula in candidates:
        reduced = normalize_formula(formula)
        if reduced is None:
            invalid += 1
        else:
            normalized.add(reduced)
    return sorted(normalized), invalid


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-materials", type=Path, default=DEFAULT_KNOWN)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--shortlist-size", type=int, default=50)
    parser.add_argument(
        "--all-scores",
        action="store_true",
        help="also write all_scores.csv with every scored composition, below the threshold too",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run composition screening and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.shortlist_size < 1:
        print("ERROR: --shortlist-size must be at least 1", file=sys.stderr)
        return 2

    try:
        classifier, threshold, metadata = load_model(args.model_dir)
        known_materials = load_known_materials(args.known_materials)
        generated = generate_candidates(MAGNETIC_METALS, ANIONS, OTHER_METALS)
        normalized, invalid_count = normalize_candidates(generated)
        reference_matches = set(normalized).intersection(known_materials)
        unseen = sorted(set(normalized).difference(known_materials))
        if not unseen:
            raise ScreeningError("no compositions remain after reference-set exclusion")

        feature_names = metadata["feature_names"]
        feature_rows: list[dict[str, float]] = []
        valid_formulas: list[str] = []
        for formula in unseen:
            features = featurize(formula, feature_names)
            if features is not None:
                feature_rows.append(features)
                valid_formulas.append(formula)
        if not feature_rows:
            raise ScreeningError("none of the unseen formulas could be featurized")

        X = pd.DataFrame(feature_rows, columns=feature_names)
        class_index = positive_class_index(classifier)
        scores = np.asarray(classifier.predict_proba(X))[:, class_index]
        uncalibrated = raw_scores(classifier, X)
        selected = [
            {
                "formula": formula,
                "magnetic_score": float(score),
                "raw_score": float(raw),
                "score_band": score_band(float(score)),
            }
            for formula, score, raw in zip(valid_formulas, scores, uncalibrated)
            if score >= threshold
        ]
        results = pd.DataFrame(
            selected, columns=["formula", "magnetic_score", "raw_score", "score_band"]
        ).sort_values(
            ["magnetic_score", "raw_score", "formula"],
            ascending=[False, False, True],
            ignore_index=True,
        )
        everything = pd.DataFrame(
            {"formula": valid_formulas, "magnetic_score": scores, "raw_score": uncalibrated}
        ).sort_values(
            ["magnetic_score", "raw_score", "formula"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        ScreeningError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "screened_candidates.csv"
    shortlist_path = args.results_dir / "candidate_shortlist.csv"
    summary_path = args.results_dir / "screening_summary.json"
    results.to_csv(results_path, index=False)
    if args.all_scores:
        everything.to_csv(args.results_dir / "all_scores.csv", index=False)

    shortlist = results.head(args.shortlist_size).copy()
    shortlist.insert(0, "rank", range(1, len(shortlist) + 1))
    shortlist["validation_status"] = (
        "composition proposal; structure and stability not evaluated"
    )
    shortlist.to_csv(shortlist_path, index=False)

    band_counts = {
        band: int(count)
        for band, count in results["score_band"].value_counts().sort_index().items()
    }
    summary = {
        "model_version": metadata.get("model_version", "unknown"),
        "decision_threshold": threshold,
        "scores_calibrated": bool(
            metadata.get("scientific_scope", {}).get("scores_calibrated", False)
        ),
        "generated_formula_strings": len(generated),
        "unique_reduced_compositions": len(normalized),
        "invalid_generated_formulas": invalid_count,
        "reference_set_size": len(known_materials),
        "reference_matches_excluded": len(reference_matches),
        "unsupported_compositions_skipped": len(unseen) - len(valid_formulas),
        "unseen_compositions_scored": len(valid_formulas),
        "selected_at_threshold": len(results),
        "score_band_counts": band_counts,
        "interpretation": (
            "Selected rows are composition proposals absent from the supplied "
            "reference set, not validated materials or globally novel compounds."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Generated formula strings: {len(generated):,}")
    print(f"Unique reduced compositions: {len(normalized):,}")
    print(f"Reference matches excluded: {len(reference_matches):,}")
    print(f"Unsupported compositions skipped: {len(unseen) - len(valid_formulas):,}")
    print(f"Unseen compositions scored: {len(valid_formulas):,}")
    print(f"Selected at threshold {threshold:.4f}: {len(results):,}")
    if summary["scores_calibrated"]:
        print("Scores are calibrated on the validation split; they are not evidence of stability.")
    else:
        print("Scores are not calibrated probabilities or evidence of physical stability.")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {shortlist_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
