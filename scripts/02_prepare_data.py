#!/usr/bin/env python3
"""Prepare composition-level magnetic labels from V2DB and, optionally, C2DB.

Two targets are supported. ``net_moment`` (default) marks a composition as
magnetic when it carries a net moment (V2DB ``FM``, C2DB total cell moment
above 0.1). ``any_magnetic`` is the thesis target (V2DB FM or AFM).

The two sources do not mean quite the same thing by a net moment. C2DB keeps
magnetic materials in their ferromagnetic state, so a C2DB row with a moment
means "spin-polarized when ordered ferromagnetically", not "ferromagnetic
ground state". Known antiferromagnets such as MnPS3 and MnBr2 carry their
FM-state moment and are labelled 1 (see scripts/09_c2db_audit.py).
"""

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
DEFAULT_C2DB = REPO_ROOT / "data" / "legacy_hybrid" / "combined_dataset.csv"
REQUIRED_COLUMNS = {"Material", "Material_is_magnetic"}
C2DB_REQUIRED_COLUMNS = {"formula_normalized", "magmom"}
TARGET_NET_MOMENT = "net_moment"
TARGET_ANY_MAGNETIC = "any_magnetic"
TARGETS = (TARGET_NET_MOMENT, TARGET_ANY_MAGNETIC)
# Which C2DB structures label a formula: every structure (formulas whose structures
# disagree are dropped), or only the lowest-hull structure that has a label.
C2DB_STRUCTURES = ("all", "lowest-hull")
# Same cut as the thesis C2DB labels: every magnetic C2DB row has |m| >= 0.104 and
# every non-magnetic row |m| <= 0.090 (total moment per cell, mu_B).
MAGMOM_THRESHOLD = 0.1
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


def v2db_labels(df: pd.DataFrame, target: str, source: Path) -> list[int]:
    """Return per-row V2DB labels for the requested target."""
    if target == TARGET_NET_MOMENT:
        validate_columns(df, {"Magnetic_state"}, source)
        states = df["Magnetic_state"].astype(str).str.strip().str.upper()
        unknown = sorted(set(states).difference({"FM", "AFM", "NM"}))
        if unknown:
            raise DataValidationError(
                f"{source}: unrecognised Magnetic_state value(s): {', '.join(unknown)}"
            )
        return (states == "FM").astype(int).tolist()

    labels: list[int] = []
    for row_number, value in zip(df.index, df["Material_is_magnetic"]):
        try:
            labels.append(parse_magnetic_label(value))
        except DataValidationError as exc:
            raise DataValidationError(
                f"{source}: CSV row {int(row_number) + 2}: {exc}"
            ) from exc
    return labels


def process_v2db(
    filepath: str | Path,
    target: str = TARGET_NET_MOMENT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load V2DB and produce deterministic, composition-level consensus labels.

    V2DB contains multiple structural prototypes for some reduced formulas. A
    composition-only model cannot distinguish those structures, so formulas with
    conflicting magnetic labels are excluded rather than assigned an arbitrary
    first-row label.
    """
    if target not in TARGETS:
        raise DataValidationError(f"unknown target {target!r}; use one of {TARGETS}")
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

    working["is_magnetic"] = v2db_labels(working, target, source)

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

    prepared["source"] = "V2DB"
    prepared = prepared[
        ["formula_normalized", "is_magnetic", "source", "record_count", "prototype_count"]
    ]
    stats = {
        "input_rows": int(len(df)),
        "invalid_formula_rows": invalid_formula_count,
        "conflicting_formulas_removed": int(len(conflicting_formulas)),
        "output_compositions": int(len(prepared)),
        "magnetic_compositions": int(prepared["is_magnetic"].sum()),
    }
    return prepared, stats


def process_c2db(
    filepath: str | Path,
    structures: str = "all",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load C2DB rows (formula + total moment) and label them by net moment.

    Accepts the thesis hybrid table, where C2DB rows are marked ``source=C2DB``,
    or any table with ``formula_normalized`` and ``magmom`` columns, such as the
    one ``09_c2db_audit.py`` writes from ``c2db.db``. The moment is that of the
    stored FM state, see the module docstring. ``structures="lowest-hull"`` needs
    an ``ehull`` column and labels each formula by its lowest-hull structure
    among those with a label (a moment, or C2DB's is_magnetic key).
    """
    if structures not in C2DB_STRUCTURES:
        raise DataValidationError(f"unknown C2DB structure choice {structures!r}")
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(f"C2DB input not found: {source}. Supply it with --c2db.")

    df = pd.read_csv(source)
    required = C2DB_REQUIRED_COLUMNS | ({"ehull"} if structures == "lowest-hull" else set())
    validate_columns(df, required, source)
    if "source" in df.columns:
        df = df.loc[df["source"].astype(str).str.upper() == "C2DB"]

    working = df.copy()
    working["formula_normalized"] = working["formula_normalized"].map(normalize_formula)
    working = working.loc[working["formula_normalized"].notna()]
    label = (working["magmom"].abs() > MAGMOM_THRESHOLD).astype(float).where(working["magmom"].notna())
    if "is_magnetic_key" in working.columns:
        # c2db.db stores no moment for non-spin-polarized calculations; C2DB's own
        # is_magnetic key is 0 for 3,060 of the 3,061 such structures.
        label = label.fillna(working["is_magnetic_key"])
    working["is_magnetic"] = label
    unlabelled = working.loc[working["is_magnetic"].isna(), "formula_normalized"]
    lowest_unlabelled = 0
    if structures == "lowest-hull":
        ordered = working.sort_values(["formula_normalized", "ehull"], kind="mergesort")
        lowest_unlabelled = int(ordered.drop_duplicates("formula_normalized")["is_magnetic"].isna().sum())
        working = ordered.loc[ordered["is_magnetic"].notna()].drop_duplicates("formula_normalized")
    working = working.loc[working["is_magnetic"].notna()]
    working["is_magnetic"] = working["is_magnetic"].astype(int)

    grouped = working.groupby("formula_normalized", sort=True, observed=True)
    label_counts = grouped["is_magnetic"].nunique()
    conflicting_formulas = set(label_counts[label_counts > 1].index)
    consensus = working.loc[~working["formula_normalized"].isin(conflicting_formulas)]
    prepared = (
        consensus.groupby("formula_normalized", sort=True, observed=True)
        .agg(is_magnetic=("is_magnetic", "first"), record_count=("is_magnetic", "size"))
        .reset_index()
    )
    prepared["source"] = "C2DB"
    prepared["prototype_count"] = pd.NA
    prepared = prepared[
        ["formula_normalized", "is_magnetic", "source", "record_count", "prototype_count"]
    ]
    stats = {
        "input_rows": int(len(working)),
        "conflicting_formulas_removed": int(len(conflicting_formulas)),
        "unlabelled_rows": int(len(unlabelled)),
        "lowest_hull_structure_unlabelled": lowest_unlabelled,
        "output_compositions": int(len(prepared)),
        "magnetic_compositions": int(prepared["is_magnetic"].sum()),
    }
    return prepared, stats


def combine_sources(
    v2db: pd.DataFrame,
    c2db: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Merge the sources; a C2DB (DFT) label replaces a V2DB (ML-predicted) one."""
    if c2db is None or c2db.empty:
        return v2db.reset_index(drop=True), {"overlap": 0, "overlap_disagreeing": 0}

    overlap = v2db.merge(c2db, on="formula_normalized", suffixes=("_v2db", "_c2db"))
    disagreeing = int((overlap["is_magnetic_v2db"] != overlap["is_magnetic_c2db"]).sum())
    kept_v2db = v2db.loc[~v2db["formula_normalized"].isin(c2db["formula_normalized"])]
    combined = (
        pd.concat([kept_v2db, c2db], ignore_index=True)
        .sort_values("formula_normalized", kind="mergesort")
        .reset_index(drop=True)
    )
    return combined, {"overlap": int(len(overlap)), "overlap_disagreeing": disagreeing}


def collect_formulas(filepath: str | Path, column: str, source_filter: str | None = None) -> set[str]:
    """Return every valid reduced formula in an input, before any row is removed."""
    df = pd.read_csv(filepath)
    if source_filter is not None and "source" in df.columns:
        df = df.loc[df["source"].astype(str).str.upper() == source_filter]
    unique = pd.Series(df[column].dropna().astype(str).unique())
    return set(unique.map(normalize_formula).dropna())


def create_known_materials(formulas: pd.DataFrame | set[str], output_file: str | Path) -> int:
    """Write a sorted reference set of reduced formulas and return its size."""
    if isinstance(formulas, pd.DataFrame):
        formulas = set(formulas["formula_normalized"].dropna().astype(str))
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(formulas)
    destination.write_text("".join(f"{formula}\n" for formula in ordered), encoding="utf-8")
    return len(ordered)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare consensus composition labels from V2DB and C2DB exports."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--c2db",
        type=Path,
        default=DEFAULT_C2DB,
        help="table with C2DB formula + magmom rows (default: the thesis hybrid table)",
    )
    parser.add_argument("--no-c2db", action="store_true", help="use V2DB only")
    parser.add_argument(
        "--c2db-only",
        action="store_true",
        help="train on C2DB labels only; V2DB formulas still go into the known-material list",
    )
    parser.add_argument("--c2db-structures", choices=C2DB_STRUCTURES, default="all")
    parser.add_argument("--target", choices=TARGETS, default=TARGET_NET_MOMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--known-output", type=Path, default=DEFAULT_KNOWN_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run data preparation and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.no_c2db and args.c2db_only:
        print("ERROR: --no-c2db and --c2db-only exclude each other", file=sys.stderr)
        return 2
    try:
        if not args.c2db_only:
            v2db, stats = process_v2db(args.input, target=args.target)
        elif not args.input.is_file():
            raise FileNotFoundError(f"V2DB input not found: {args.input} (needed for known materials)")
        known = collect_formulas(args.input, "Material")
        c2db = None
        if not args.no_c2db:
            c2db, c2db_stats = process_c2db(args.c2db, structures=args.c2db_structures)
            known |= collect_formulas(args.c2db, "formula_normalized", source_filter="C2DB")
        if args.c2db_only:
            prepared = c2db.reset_index(drop=True)
        else:
            prepared, merge_stats = combine_sources(v2db, c2db)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(args.output, index=False)
        known_count = create_known_materials(known, args.known_output)
    except (FileNotFoundError, DataValidationError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    magnetic = int(prepared["is_magnetic"].sum())
    total = int(len(prepared))
    print(f"Target: {'C2DB net moment' if args.c2db_only else args.target}")
    if not args.c2db_only:
        print(f"V2DB input rows: {stats['input_rows']:,}")
        print(f"V2DB invalid formula rows removed: {stats['invalid_formula_rows']:,}")
        print(
            "V2DB structurally conflicting formulas removed: "
            f"{stats['conflicting_formulas_removed']:,}"
        )
    if c2db is not None:
        if args.target == TARGET_ANY_MAGNETIC and not args.c2db_only:
            print(
                "WARNING: C2DB labels are FM-state moments, not ground states; a C2DB "
                "antiferromagnet usually counts as magnetic and a zero-moment row as "
                "non-magnetic under any target.",
                file=sys.stderr,
            )
        print(f"C2DB structures used: {args.c2db_structures}")
        print(f"C2DB rows without a moment or label: {c2db_stats['unlabelled_rows']:,}")
        if args.c2db_structures == "lowest-hull":
            print(
                "C2DB formulas whose lowest-hull structure has no label "
                "(the next labelled one is used, if any): "
                f"{c2db_stats['lowest_hull_structure_unlabelled']:,}"
            )
        print(f"C2DB compositions: {c2db_stats['output_compositions']:,}")
        if not args.c2db_only:
            print(
                f"Formulas in both sources: {merge_stats['overlap']:,} "
                f"({merge_stats['overlap_disagreeing']:,} disagree; C2DB label kept)"
            )
    print(f"Prepared compositions: {total:,}")
    print(f"Magnetic compositions: {magnetic:,} ({magnetic / total:.1%})")
    print(f"Known-material reference entries: {known_count:,}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.known_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
