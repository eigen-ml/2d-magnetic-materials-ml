from __future__ import annotations


def test_candidate_generation_is_sorted_and_unique(discover_module):
    candidates = discover_module.generate_candidates(["Fe", "Mn"], ["O", "S"])
    assert candidates == sorted(set(candidates))
    assert "FeO2" in candidates
    assert "FeMnO3" in candidates


def test_unequal_metal_ratios_are_generated_both_ways(discover_module):
    candidates = discover_module.generate_candidates(["Mn", "Cr"], ["Br"], ["Mo"])
    assert {"Mn2CrBr4", "Cr2MnBr4"} <= set(candidates)
    assert {"MnMo2Br4", "Mn2MoBr4"} <= set(candidates)


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


def test_discovery_parser_has_an_all_scores_switch(discover_module):
    assert discover_module.build_parser().parse_args(["--all-scores"]).all_scores is True
    assert discover_module.build_parser().parse_args([]).all_scores is False


def test_score_bands_are_descriptive(discover_module):
    assert discover_module.score_band(0.99) == "0.98-1.00"
    assert discover_module.score_band(0.96) == "0.95-0.98"
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


def test_raw_scores_unwrap_a_calibrated_model(discover_module, train_module):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=300)})
    y = pd.Series((X["x"] > 0).astype(int))
    classifier, _ = train_module.train_model(X, y, n_estimators=10)
    calibrated = train_module.calibrate(classifier, X, y)

    raw = discover_module.raw_scores(calibrated, X)

    assert np.allclose(raw, classifier.predict_proba(X)[:, 1])
    assert np.allclose(discover_module.raw_scores(classifier, X), raw)
