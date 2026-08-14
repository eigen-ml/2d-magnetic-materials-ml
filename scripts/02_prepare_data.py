#!/usr/bin/env python3
"""Prepare composition-level magnetic labels from the V2DB CSV export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "v2db.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "combined_dataset.csv"
DEFAULT_KNOWN_OUTPUT = REPO_ROOT / "data" / "known_materials.txt"
REQUIRED_COLUMNS = {"Material", "Material_is_magnetic"}
TRUE_LABELS = {"1", "1.0", "true", "t", "yes", "y"}
FALSE_LABELS = {"0", "0.0", "false", "f", "no", "n"}


class DataValidationError(ValueError):
    """Raised when an input dataset does not satisfy the documented schema."""


def normalize_formula(formula: object) -> str | None:
    """Return a reduced chemical formula, or ``None`` for an invalid value."""
    if formula is None or (isinstance(formula, float) and np.isnan(formula)):
        return None

    try:
        return Composition(str(formula).strip()).reduced_formula
    except (TypeError, ValueError):
        return None


def parse_magnetic_label(value: object) -> int:
    """Parse a strict binary magnetic label.

    Unrecognised values are rejected instead of being silently converted to the
    non-magnetic class.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in TRUE_LABELS:
        return 1
    if normalized in FALSE_LABELS:
        return 0
    raise DataValidationError(f"unrecognised magnetic label: {value!r}")


def validate_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    """Raise a useful error when required columns are absent."""
    missing = sorted(required.difference(df.columns))
    if missing:
        raise DataValidationError(
            f"{source} is missing required column(s): {', '.join(missing)}"
        )


def process_v2db(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load V2DB and produce deterministic, composition-level consensus labels.

    V2DB contains multiple structural prototypes for some reduced formulas. A
    composition-only model cannot distinguish those structures, so formulas with
    conflicting magnetic labels are excluded rather than assigned an arbitrary
    first-row label.
    """
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(
            f"V2DB input not found: {source}. Supply it with --input."
        )

    df = pd.read_csv(source)
    validate_columns(df, REQUIRED_COLUMNS, source)

    normalized = df["Material"].map(normalize_formula)
    invalid_formula_count = int(normalized.isna().sum())
    working = df.loc[normalized.notna()].copy()
    working["formula_normalized"] = normalized.loc[normalized.notna()]

    labels: list[int] = []
    for row_number, value in zip(working.index, working["Material_is_magnetic"]):
        try:
            labels.append(parse_magnetic_label(value))
        except DataValidationError as exc:
            raise DataValidationError(
                f"{source}: CSV row {int(row_number) + 2}: {exc}"
            ) from exc
    working["is_magnetic"] = labels

    grouped = working.groupby("formula_normalized", sort=True, observed=True)
    label_counts = grouped["is_magnetic"].nunique()
    conflicting_formulas = set(label_counts[label_counts > 1].index)
    consensus = working.loc[
        ~working["formula_normalized"].isin(conflicting_formulas)
    ].copy()

    aggregate_spec: dict[str, tuple[str, str]] = {
        "is_magnetic": ("is_magnetic", "first"),
        "record_count": ("is_magnetic", "size"),
    }
    if "Prototype" in consensus.columns:
        aggregate_spec["prototype_count"] = ("Prototype", "nunique")

    prepared = (
        consensus.groupby("formula_normalized", sort=True, observed=True)
        .agg(**aggregate_spec)
        .reset_index()
    )
    if "prototype_count" not in prepared.columns:
        prepared["prototype_count"] = pd.NA

    prepared = prepared[
        ["formula_normalized", "is_magnetic", "record_count", "prototype_count"]
    ]
    stats = {
        "input_rows": int(len(df)),
        "invalid_formula_rows": invalid_formula_count,
        "conflicting_formulas_removed": int(len(conflicting_formulas)),
        "output_compositions": int(len(prepared)),
        "magnetic_compositions": int(prepared["is_magnetic"].sum()),
    }
    return prepared, stats


def create_known_materials(df: pd.DataFrame, output_file: str | Path) -> int:
    """Write a sorted reference set of reduced formulas and return its size."""
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    formulas = sorted(set(df["formula_normalized"].dropna().astype(str)))
    destination.write_text("".join(f"{formula}\n" for formula in formulas), encoding="utf-8")
    return len(formulas)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare consensus composition labels from a V2DB CSV export."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--known-output", type=Path, default=DEFAULT_KNOWN_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run data preparation and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        prepared, stats = process_v2db(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(args.output, index=False)
        known_count = create_known_materials(prepared, args.known_output)
    except (FileNotFoundError, DataValidationError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    magnetic = stats["magnetic_compositions"]
    total = stats["output_compositions"]
    print(f"Input rows: {stats['input_rows']:,}")
    print(f"Invalid formula rows removed: {stats['invalid_formula_rows']:,}")
    print(
        "Structurally conflicting formulas removed: "
        f"{stats['conflicting_formulas_removed']:,}"
    )
    print(f"Prepared compositions: {total:,}")
    print(f"Magnetic compositions: {magnetic:,} ({magnetic / total:.1%})")
    print(f"Known-material reference entries: {known_count:,}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.known_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
