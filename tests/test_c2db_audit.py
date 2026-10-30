from __future__ import annotations

import pandas as pd
import pytest


def write_db(path):
    """A small C2DB-like ASE database: two CrI3 polymorphs, an FM-state MnPS3, an AFM MnO."""
    ase = pytest.importorskip("ase")
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.db import connect

    rows = [
        ("Cr2I6", [3, 3, 0, 0, 0, 0, 0, 0], "original03-18", 0.0),
        ("Cr2I6", [3, 3, 0, 0, 0, 0, 0, 0], "Lyngby22_LDP", 0.197),
        ("Mn2P2S6", [5, 5] + [0] * 8, "exfoliated02-21", 0.031),
        ("Mn2O2", [4.5, -4.5, 0, 0], "original03-18", 0.05),
        ("MoS2", [0, 0, 0], "original03-18", 0.0),
        ("VSe2", [1, 0, 0], "Wang23", 0.02),
    ]
    with connect(str(path)) as db:
        for index, (formula, moments, origin, ehull) in enumerate(rows):
            atoms = ase.Atoms(formula, positions=[[0, 0, i] for i in range(len(moments))], cell=[5, 5, 20])
            atoms.calc = SinglePointCalculator(atoms, energy=0.0, magmom=sum(moments), magmoms=moments)
            db.write(atoms, uid=f"{formula}-{index}", label=origin, ehull=ehull, hform=-0.5)


@pytest.fixture()
def tables(audit_module, tmp_path):
    path = tmp_path / "c2db.db"
    write_db(path)
    db = audit_module.read_db(path)
    thesis = pd.DataFrame(
        {
            "formula_normalized": ["CrI3", "MnPS3", "MnO", "MoS2", "Fe2O3"],
            "magmom": [6.0, 10.0, 0.0, 0.0, 10.0],
            "ehull": [0.197, 0.031, 0.05, 0.0, 0.0],
            "is_magnetic": [1, 1, 0, 0, 1],
            "source": ["C2DB", "C2DB", "C2DB", "C2DB", "V2DB"],
        }
    )
    thesis_path = tmp_path / "thesis.csv"
    thesis.to_csv(thesis_path, index=False)
    return db, audit_module.read_thesis(thesis_path)


def test_read_db_reduces_formulas_and_flags_antiparallel_moments(tables):
    db, _ = tables
    rows = db.set_index("uid")
    assert rows.loc["Mn2P2S6-2", "formula_normalized"] == "MnPS3"
    assert not rows.loc["Mn2P2S6-2", "antiparallel"]
    assert rows.loc["Mn2O2-3", "antiparallel"]
    assert rows.loc["Mn2O2-3", "magmom"] == pytest.approx(0.0)


def test_audit_finds_the_release_and_the_kept_polymorph(audit_module, tables):
    db, thesis = tables
    summary = audit_module.audit(db, thesis)

    assert summary["thesis_table"]["rows"] == 4  # the V2DB row is dropped
    assert summary["thesis_table"]["formulas_in_db"] == 4
    assert summary["db_formulas_not_in_thesis"] == {"formulas": 1, "rows_by_origin": {"Wang23": 1}}
    kept = summary["thesis_table"]["kept_polymorph"]
    assert (kept["formulas"], kept["lowest_hull"], kept["highest_hull"]) == (1, 0, 1)
    assert summary["db"]["rows_with_antiparallel_site_moments"] == 1
    assert summary["label_agreement_lowest_hull_polymorph"]["thesis_vs_db_label"] == {
        "thesis0_db0": 2, "thesis0_db1": 0, "thesis1_db0": 0, "thesis1_db1": 2,
    }
    mnps3 = summary["known_magnets"]["MnPS3"]["db"]
    assert [(r["magmom"], r["antiparallel"]) for r in mnps3] == [(10.0, False)]


def test_s5_recheck_reports_newly_known_candidates(audit_module, tables):
    db, _ = tables
    candidates = pd.DataFrame(
        {
            "formula_plain": ["VSe2", "Mn2CrS4", "CrI3", None],
            "first_failure": ["pass", "pass", "S5", "S1"],
            "in_c2db": [False, False, True, False],
        }
    )
    result = audit_module.s5_recheck(db, candidates, ["CrI3", "Mn2CrS4"], ["MnO"])

    assert result["newly_in_c2db_passing"] == ["VSe2"]
    assert (result["passing_before"], result["passing_after"]) == (2, 1)
    assert result["batch1"]["Mn2CrS4"] is None
    assert result["batch1"]["CrI3"]["ehull"] == 0.0
    assert result["checked"]["MnO"]["origin"] == "original03-18"
