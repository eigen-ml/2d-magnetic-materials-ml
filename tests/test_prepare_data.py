from __future__ import annotations

import pandas as pd
import pytest


def test_normalize_formula_uses_reduced_composition(prepare_module):
    assert prepare_module.normalize_formula("Fe4O6") == "Fe2O3"
    assert prepare_module.normalize_formula("not-a-formula") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, 1), (False, 0), ("YES", 1), ("0.0", 0), (1, 1), (0.0, 0)],
)
def test_parse_magnetic_label_is_strict(prepare_module, value, expected):
    assert prepare_module.parse_magnetic_label(value) == expected


def test_parse_magnetic_label_rejects_unknown_values(prepare_module):
    with pytest.raises(prepare_module.DataValidationError, match="unrecognised"):
        prepare_module.parse_magnetic_label("maybe")


def test_process_v2db_drops_structurally_conflicting_labels(prepare_module, tmp_path):
    source = tmp_path / "v2db.csv"
    pd.DataFrame(
        {
            "Material": ["Fe2O3", "Fe4O6", "MnO", "Mn2O2"],
            "Prototype": ["A", "B", "A", "B"],
            "Material_is_magnetic": [1, 0, 1, 1],
        }
    ).to_csv(source, index=False)

    prepared, stats = prepare_module.process_v2db(source, target="any_magnetic")

    assert prepared["formula_normalized"].tolist() == ["MnO"]
    assert prepared["is_magnetic"].tolist() == [1]
    assert prepared["record_count"].tolist() == [2]
    assert prepared["prototype_count"].tolist() == [2]
    assert stats["conflicting_formulas_removed"] == 1


def test_net_moment_target_counts_only_fm(prepare_module, tmp_path):
    source = tmp_path / "v2db.csv"
    pd.DataFrame(
        {
            "Material": ["CrI3", "MnPS3", "MoS2"],
            "Magnetic_state": ["FM", "AFM", "NM"],
            "Material_is_magnetic": [1, 1, 0],
        }
    ).to_csv(source, index=False)

    prepared, _ = prepare_module.process_v2db(source, target="net_moment")

    labels = dict(zip(prepared["formula_normalized"], prepared["is_magnetic"]))
    assert labels == {"CrI3": 1, "MnPS3": 0, "MoS2": 0}
    assert set(prepared["source"]) == {"V2DB"}


def test_net_moment_target_needs_magnetic_state(prepare_module, tmp_path):
    source = tmp_path / "v2db.csv"
    pd.DataFrame({"Material": ["CrI3"], "Material_is_magnetic": [1]}).to_csv(
        source, index=False
    )
    with pytest.raises(prepare_module.DataValidationError, match="Magnetic_state"):
        prepare_module.process_v2db(source, target="net_moment")


def test_process_c2db_labels_by_total_moment(prepare_module, tmp_path):
    source = tmp_path / "hybrid.csv"
    pd.DataFrame(
        {
            "formula_normalized": ["CrI3", "MnPS3", "MoS2", "VSe2"],
            "magmom": [6.0, 0.0, 0.0, 0.5],
            "source": ["C2DB", "C2DB", "C2DB", "V2DB"],
        }
    ).to_csv(source, index=False)

    prepared, stats = prepare_module.process_c2db(source)

    labels = dict(zip(prepared["formula_normalized"], prepared["is_magnetic"]))
    assert labels == {"CrI3": 1, "MnPS3": 0, "MoS2": 0}
    assert stats["output_compositions"] == 3


def test_combine_sources_keeps_the_c2db_label(prepare_module):
    columns = ["formula_normalized", "is_magnetic", "source", "record_count", "prototype_count"]
    v2db = pd.DataFrame(
        [["CrI3", 0, "V2DB", 1, 1], ["FeCl2", 1, "V2DB", 1, 1]], columns=columns
    )
    c2db = pd.DataFrame([["CrI3", 1, "C2DB", 1, pd.NA]], columns=columns)

    combined, stats = prepare_module.combine_sources(v2db, c2db)

    rows = combined.set_index("formula_normalized")
    assert rows.loc["CrI3", "source"] == "C2DB"
    assert rows.loc["CrI3", "is_magnetic"] == 1
    assert rows.loc["FeCl2", "source"] == "V2DB"
    assert stats == {"overlap": 1, "overlap_disagreeing": 1}


def test_known_materials_include_formulas_removed_as_conflicts(prepare_module, tmp_path):
    source = tmp_path / "v2db.csv"
    pd.DataFrame(
        {
            "Material": ["Fe2O3", "Fe4O6", "MnO"],
            "Magnetic_state": ["FM", "NM", "FM"],
            "Material_is_magnetic": [1, 0, 1],
        }
    ).to_csv(source, index=False)

    prepared, _ = prepare_module.process_v2db(source)
    known = prepare_module.collect_formulas(source, "Material")

    assert "Fe2O3" not in set(prepared["formula_normalized"])
    assert known == {"Fe2O3", "MnO"}


def test_process_v2db_reports_missing_columns(prepare_module, tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"Material": ["FeO"]}).to_csv(source, index=False)
    with pytest.raises(prepare_module.DataValidationError, match="Material_is_magnetic"):
        prepare_module.process_v2db(source, target="any_magnetic")


def test_prepare_cli_reports_missing_input(prepare_module, tmp_path, capsys):
    exit_code = prepare_module.main(["--input", str(tmp_path / "missing.csv")])
    assert exit_code == 2
    assert "V2DB input not found" in capsys.readouterr().err
