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

    prepared, stats = prepare_module.process_v2db(source)

    assert prepared["formula_normalized"].tolist() == ["MnO"]
    assert prepared["is_magnetic"].tolist() == [1]
    assert prepared["record_count"].tolist() == [2]
    assert prepared["prototype_count"].tolist() == [2]
    assert stats["conflicting_formulas_removed"] == 1


def test_process_v2db_reports_missing_columns(prepare_module, tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"Material": ["FeO"]}).to_csv(source, index=False)
    with pytest.raises(prepare_module.DataValidationError, match="Material_is_magnetic"):
        prepare_module.process_v2db(source)


def test_prepare_cli_reports_missing_input(prepare_module, tmp_path, capsys):
    exit_code = prepare_module.main(["--input", str(tmp_path / "missing.csv")])
    assert exit_code == 2
    assert "V2DB input not found" in capsys.readouterr().err
