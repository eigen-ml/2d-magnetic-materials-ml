#!/usr/bin/env python3
"""Compare the thesis C2DB table with a current C2DB release (ASE database file).

Three questions for dft/CRITERIA.md: which release the thesis table matches, what its
magnetic label means, and which formulas the current release adds to S5. Writes a
summary JSON (tracked) and a formula table for the S5 recheck (not tracked, the C2DB
terms apply to it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THESIS = REPO_ROOT / "data" / "legacy_hybrid" / "combined_dataset.csv"
DEFAULT_CANDIDATES = REPO_ROOT / "dft" / "candidates_final.csv"
DEFAULT_SELECTION = REPO_ROOT / "dft" / "selection.json"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "c2db_audit" / "summary.json"
DEFAULT_FORMULAS = REPO_ROOT / "data" / "external" / "c2db_formulas.csv"
# Same cut as 02_prepare_data.py (total moment per cell, mu_B).
MAGMOM_THRESHOLD = 0.1
# A site counts as a local moment above this; opposite signs mean AFM or ferrimagnetic order.
SITE_MOMENT = 0.5
MAGMOM_BINS = (0.0, 0.1, 0.5, 1.5, float("inf"))
# Monolayers whose stored magnetic state is worth reading by hand. The MPX3 family is
# Neel antiferromagnetic in experiment.
KNOWN_MAGNETS = (
    "MnPS3", "MnPSe3", "FePS3", "NiPS3", "MnBr2", "MnCl2", "FeCl2",
    "CrI3", "CrBr3", "CrCl3", "VCl3", "CrSBr", "Fe3GeTe2",
)
METALS = ("Mn", "Cr", "Fe", "Co", "V", "Ni")


def reduced(formula: object) -> str | None:
    """Return pymatgen's reduced formula, or ``None`` for an unparsable value."""
    try:
        return Composition(str(formula)).reduced_formula
    except (TypeError, ValueError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_db(path: Path) -> pd.DataFrame:
    """One row per C2DB structure: formula, origin, energetics and magnetic data."""
    from ase.db import connect

    records = []
    for row in connect(str(path)).select():
        moments = row.get("magmoms")
        antiparallel = None
        if moments is not None:
            antiparallel = bool((moments >= SITE_MOMENT).any() and (moments <= -SITE_MOMENT).any())
        is_magnetic = row.get("is_magnetic")
        records.append(
            {
                "uid": row.get("uid"),
                "formula_db": row.formula,
                "origin": row.get("label", "unknown"),
                "ehull": row.get("ehull"),
                "hform": row.get("hform"),
                "magmom": row.get("magmom"),
                "is_magnetic_key": None if is_magnetic is None else int(is_magnetic),
                "antiparallel": antiparallel,
                "dyn_stab": row.get("dyn_stab"),
            }
        )
    table = pd.DataFrame.from_records(records)
    table["formula_normalized"] = table["formula_db"].map(reduced)
    return table


def read_thesis(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.loc[table["source"].astype(str).str.upper() == "C2DB"].copy()
    table["formula_normalized"] = table["formula_normalized"].map(reduced)
    return table


def label(magmom: object) -> int | None:
    if magmom is None or pd.isna(magmom):
        return None
    return int(abs(float(magmom)) >= MAGMOM_THRESHOLD)


def magmom_bins(values: pd.Series) -> dict[str, int]:
    counts = pd.cut(values.dropna().abs(), MAGMOM_BINS, right=False).value_counts(sort=False)
    return {str(interval): int(count) for interval, count in counts.items()}


def confusion(thesis: pd.Series, other: pd.Series) -> dict[str, int]:
    pairs = Counter(zip(thesis.astype(int), other.astype(int)))
    return {f"thesis{a}_db{b}": int(pairs.get((a, b), 0)) for a in (0, 1) for b in (0, 1)}


def contains(formula: str, element: str) -> bool:
    return element in Composition(formula).as_dict()


def audit(db: pd.DataFrame, thesis: pd.DataFrame) -> dict:
    """Release fingerprint, label meaning and the formula difference."""
    db_formulas = set(db["formula_normalized"].dropna())
    thesis_formulas = set(thesis["formula_normalized"].dropna())
    added = db.loc[~db["formula_normalized"].isin(thesis_formulas)]

    # The thesis table has one row per formula; compare it with every polymorph and with
    # the lowest-hull one.
    merged = thesis.merge(db, on="formula_normalized", suffixes=("_thesis", "_db"))
    exact_ehull = (merged["ehull_thesis"] - merged["ehull_db"]).abs() < 1e-6
    exact_magmom = (merged["magmom_thesis"] - merged["magmom_db"]).abs() < 1e-6
    matched = merged.loc[exact_ehull]

    lowest = db.sort_values("ehull").drop_duplicates("formula_normalized")
    pair = thesis.merge(lowest, on="formula_normalized", suffixes=("_thesis", "_db"))
    pair["db_label"] = pair["magmom_db"].map(label)
    with_label = pair.loc[pair["db_label"].notna()]
    with_key = pair.loc[pair["is_magnetic_key"].notna()]

    # Which polymorph the thesis row came from, where a formula has several.
    hull = db.groupby("formula_normalized")["ehull"].agg(["size", "min", "max"])
    multi = pair.loc[pair["formula_normalized"].map(hull["size"]) > 1]
    above = multi["ehull_thesis"] - multi["formula_normalized"].map(hull["min"])
    kept = {
        "formulas": int(len(multi)),
        "lowest_hull": int((above.abs() < 1e-6).sum()),
        "highest_hull": int(((multi["ehull_thesis"] - multi["formula_normalized"].map(hull["max"])).abs() < 1e-6).sum()),
        "above_lowest_by_more_than_0.05": int((above > 0.05).sum()),
        "label_differs_from_lowest_hull": int((with_label["is_magnetic"] != with_label["db_label"]).sum()),
    }

    mn_db = lowest.loc[lowest["formula_normalized"].map(lambda f: contains(f, "Mn"))]
    mn_labels = mn_db["magmom"].map(label).dropna()
    metal_share = {}
    for metal in METALS:
        rows = lowest.loc[lowest["formula_normalized"].map(lambda f, m=metal: contains(f, m))]
        labels = rows["magmom"].map(label).dropna()
        metal_share[metal] = {"formulas": int(len(rows)), "magnetic_share": round(float(labels.mean()), 3)}

    known = {}
    for formula in KNOWN_MAGNETS:
        row = thesis.loc[thesis["formula_normalized"] == formula]
        known[formula] = {
            "thesis_magmom": None if row.empty else round(float(row["magmom"].iloc[0]), 3),
            "db": [
                {
                    "uid": r.uid,
                    "origin": r.origin,
                    "ehull": round(float(r.ehull), 3),
                    "magmom": None if pd.isna(r.magmom) else round(float(r.magmom), 3),
                    "is_magnetic_key": None if pd.isna(r.is_magnetic_key) else int(r.is_magnetic_key),
                    "antiparallel": None if pd.isna(r.antiparallel) else bool(r.antiparallel),
                }
                for r in db.loc[db["formula_normalized"] == formula].sort_values("ehull").itertuples()
            ],
        }

    return {
        "db": {
            "rows": int(len(db)),
            "unique_formulas": len(db_formulas),
            "origin_rows": {k: int(v) for k, v in db["origin"].value_counts().sort_index().items()},
            "rows_with_is_magnetic_key": int(db["is_magnetic_key"].notna().sum()),
            "rows_with_magmom": int(db["magmom"].notna().sum()),
            "abs_magmom_bins": magmom_bins(db["magmom"]),
            "rows_with_antiparallel_site_moments": int(db["antiparallel"].eq(True).sum()),
        },
        "thesis_table": {
            "rows": int(len(thesis)),
            "unique_formulas": len(thesis_formulas),
            "formulas_in_db": len(thesis_formulas & db_formulas),
            "formulas_not_in_db": len(thesis_formulas - db_formulas),
            "not_in_db_examples": sorted(thesis_formulas - db_formulas)[:25],
            "rows_with_exact_ehull_in_db": int(matched["formula_normalized"].nunique()),
            "rows_with_exact_ehull_and_magmom_in_db": int(
                merged.loc[exact_ehull & exact_magmom, "formula_normalized"].nunique()
            ),
            "exact_matches_by_origin": {
                k: int(v) for k, v in matched.drop_duplicates("formula_normalized")["origin"].value_counts().items()
            },
            "kept_polymorph": kept,
        },
        "db_formulas_not_in_thesis": {
            "formulas": int(added["formula_normalized"].nunique()),
            "rows_by_origin": {k: int(v) for k, v in added["origin"].value_counts().sort_index().items()},
        },
        "label_agreement_lowest_hull_polymorph": {
            "overlap": int(len(pair)),
            "with_db_magmom": int(len(with_label)),
            "thesis_vs_db_magmom": confusion(with_label["is_magnetic"], with_label["db_label"]),
            "with_is_magnetic_key": int(len(with_key)),
            "thesis_vs_is_magnetic_key": confusion(with_key["is_magnetic"], with_key["is_magnetic_key"]),
        },
        "mn_lowest_hull": {
            "formulas": int(len(mn_db)),
            "with_magmom": int(len(mn_labels)),
            "magnetic_share": round(float(mn_labels.mean()), 3),
            "antiparallel_rows": int(mn_db["antiparallel"].eq(True).sum()),
        },
        "metal_magnetic_share_lowest_hull": metal_share,
        "known_magnets": known,
    }


def s5_recheck(
    db: pd.DataFrame, candidates: pd.DataFrame, batch1: Iterable[str], extra: Iterable[str]
) -> dict:
    """Which filtered candidates the current release would mark as known 2D compounds."""
    found = db.sort_values("ehull").drop_duplicates("formula_normalized").set_index("formula_normalized")
    candidates = candidates.loc[candidates["formula_plain"].notna()]
    candidates = candidates.assign(rf=candidates["formula_plain"].map(reduced))
    passing = candidates.loc[candidates["first_failure"] == "pass"]
    known_before = candidates["in_c2db"].astype("boolean").fillna(False).astype(bool)
    newly = candidates.loc[candidates["rf"].isin(found.index) & ~known_before]

    def hit(formula: str) -> dict | None:
        rf = reduced(formula)
        if rf not in found.index:
            return None
        r = found.loc[rf]
        return {"uid": r["uid"], "origin": r["origin"], "ehull": round(float(r["ehull"]), 3)}

    return {
        "candidates": int(candidates["rf"].nunique()),
        "newly_in_c2db": int(newly["rf"].nunique()),
        "newly_in_c2db_passing": sorted(set(newly.loc[newly["first_failure"] == "pass", "formula_plain"])),
        "passing_before": int(passing["rf"].nunique()),
        "passing_after": int(passing.loc[~passing["rf"].isin(found.index), "rf"].nunique()),
        "batch1": {formula: hit(formula) for formula in batch1},
        "checked": {formula: hit(formula) for formula in extra},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="C2DB ASE database (c2db.db)")
    parser.add_argument("--thesis", type=Path, default=DEFAULT_THESIS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--check", nargs="*", default=[], help="further formulas to look up")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--formulas-output", type=Path, default=DEFAULT_FORMULAS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.db, args.thesis, args.candidates, args.selection):
        if not path.is_file():
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            return 2

    db = read_db(args.db)
    thesis = read_thesis(args.thesis)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    batch1 = [*selection["main"], *selection["reference"]]

    summary = {
        "db_file": {"name": args.db.name, "sha256": sha256(args.db)},
        **audit(db, thesis),
        "s5_recheck": s5_recheck(db, pd.read_csv(args.candidates), batch1, args.check),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Input for 06_filter_candidates.py --c2db: it reads formula_normalized where source is C2DB.
    args.formulas_output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["formula_normalized", "uid", "origin", "ehull", "hform", "magmom", "is_magnetic_key", "antiparallel"]
    db[columns].assign(source="C2DB").to_csv(args.formulas_output, index=False)

    print(f"C2DB rows: {summary['db']['rows']:,}, formulas: {summary['db']['unique_formulas']:,}")
    t = summary["thesis_table"]
    print(f"Thesis formulas in db: {t['formulas_in_db']:,} / {t['unique_formulas']:,}")
    print(f"New in db: {summary['db_formulas_not_in_thesis']['formulas']:,} formulas")
    print(f"S5 passing: {summary['s5_recheck']['passing_before']} -> {summary['s5_recheck']['passing_after']}")
    print(f"Wrote: {args.summary_output}")
    print(f"Wrote: {args.formulas_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
