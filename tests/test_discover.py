from __future__ import annotations


def test_candidate_generation_is_sorted_and_unique(discover_module):
    candidates = discover_module.generate_candidates(["Fe", "Mn"], ["O", "S"])
    assert candidates == sorted(set(candidates))
    assert "FeO2" in candidates
    assert "FeMnO3" in candidates


def test_formula_normalization_and_feature_schema(discover_module):
    assert discover_module.normalize_formula("Fe4O6") == "Fe2O3"
    features = discover_module.featurize(
        "Fe2O3", ["elem_O", "elem_Fe", "elem_Mn", "natoms"]
    )
    assert features == {
        "elem_O": 0.6,
        "elem_Fe": 0.4,
        "elem_Mn": 0.0,
        "natoms": 5.0,
    }
    assert discover_module.featurize("FeO", ["elem_Fe", "natoms"]) is None


def test_score_bands_are_descriptive(discover_module):
    assert discover_module.score_band(0.96) == "0.95-1.00"
    assert discover_module.score_band(0.91) == "0.90-0.95"
    assert discover_module.score_band(0.85) == "0.80-0.90"
    assert discover_module.score_band(0.60) == "threshold-0.80"


def test_discovery_cli_reports_missing_model(discover_module, tmp_path, capsys):
    known = tmp_path / "known.txt"
    known.write_text("FeO\n", encoding="utf-8")
    exit_code = discover_module.main(
        [
            "--known-materials",
            str(known),
            "--model-dir",
            str(tmp_path / "models"),
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert exit_code == 2
    assert "missing model artefact" in capsys.readouterr().err
