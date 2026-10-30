from __future__ import annotations

import numpy as np
import pandas as pd


class FixedScoreClassifier:
    def __init__(self, positive_scores):
        self.positive_scores = np.asarray(positive_scores, dtype=float)

    def predict_proba(self, _features):
        return np.column_stack([1.0 - self.positive_scores, self.positive_scores])


def test_element_fraction_features(train_module):
    features = train_module.get_element_fractions("Fe2O3")
    assert features["elem_Fe"] == 0.4
    assert features["elem_O"] == 0.6
    assert features["natoms"] == 5.0


def test_featurization_has_deterministic_column_order(train_module):
    frame = pd.DataFrame(
        {
            "formula_normalized": ["MnO", "Fe2O3"],
            "is_magnetic": [1, 0],
        }
    )
    features, indices = train_module.featurize_dataset(frame)
    assert features.columns.tolist() == [
        "elem_Fe",
        "elem_Mn",
        "elem_O",
        "natoms",
        "is_magnetic",
    ]
    assert indices == [0, 1]


def test_split_like_follows_the_reference_partition_and_order(train_module):
    import pytest

    formulas = [f"Fe{n}O{n + 1}" for n in range(1, 41)]
    reference = pd.DataFrame({"formula_normalized": formulas, "is_magnetic": [n % 2 for n in range(40)]})
    partition = train_module.reference_partition(reference)
    features, _ = train_module.featurize_dataset(reference)
    expected = train_module.split_dataset(features)

    # A subset in another order, with flipped labels: partition and order still follow the reference.
    subset = reference.iloc[::-2].reset_index(drop=True).assign(is_magnetic=lambda d: 1 - d["is_magnetic"])
    sub_features, valid = train_module.featurize_dataset(subset)
    names = pd.Series(subset.loc[valid, "formula_normalized"].to_numpy(), index=sub_features.index)
    parts = train_module.split_like(sub_features, names, partition)

    for got, want in zip(parts[:3], expected[:3]):
        want_formulas = [formulas[i] for i in want.index if formulas[i] in set(names)]
        assert names.loc[got.index].tolist() == want_formulas
    with pytest.raises(train_module.TrainingDataError, match="not in the reference split"):
        train_module.split_like(sub_features, names.replace(names.iloc[0], "MnO"), partition)


def test_threshold_is_selected_on_validation_constraints(train_module):
    classifier = FixedScoreClassifier([0.10, 0.40, 0.80, 0.90])
    threshold, scores, selection = train_module.find_operating_threshold(
        classifier,
        np.zeros((4, 1)),
        np.array([0, 0, 1, 1]),
        target_recall=1.0,
        minimum_precision=1.0,
    )
    assert threshold == 0.8
    assert scores.tolist() == [0.10, 0.40, 0.80, 0.90]
    assert selection["constraint_status"] == "met"


def test_train_cli_reports_missing_prepared_data(train_module, tmp_path, capsys):
    exit_code = train_module.main(["--data", str(tmp_path / "missing.csv")])
    assert exit_code == 2
    assert "prepared dataset not found" in capsys.readouterr().err


def test_elements_feature_set_drops_natoms(train_module):
    frame = pd.DataFrame({"formula_normalized": ["MnO", "Fe2O3"], "is_magnetic": [1, 0]})
    features, _ = train_module.featurize_dataset(frame)
    assert "natoms" not in train_module.select_features(features, "elements").columns
    assert "natoms" in train_module.select_features(features, "elements+natoms").columns


def test_band_precision_counts_each_band_once(train_module):
    bands = train_module.band_precision(
        np.array([1, 1, 0, 1, 0]),
        np.array([0.99, 0.985, 0.96, 0.91, 0.20]),
        edges=(0.98, 0.95, 0.90),
    )
    assert [band["rows"] for band in bands] == [2, 1, 1]
    assert [band["precision"] for band in bands] == [1.0, 0.0, 1.0]


def test_evaluate_by_source_skips_single_class_sources(train_module):
    results = train_module.evaluate_by_source(
        pd.Series([1, 0, 1, 1]),
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.1, 0.8, 0.4]),
        pd.Series(["V2DB", "V2DB", "C2DB", "C2DB"]),
    )
    assert results["V2DB"]["roc_auc"] == 1.0
    assert "note" in results["C2DB"]


def test_calibrated_scores_stay_in_unit_interval(train_module):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400)})
    y = pd.Series((X["x"] + rng.normal(scale=0.5, size=400) > 0).astype(int))
    classifier, _ = train_module.train_model(X, y, n_estimators=10)
    calibrated = train_module.calibrate(classifier, X, y)
    scores = calibrated.predict_proba(X)[:, 1]
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert list(calibrated.classes_) == [0, 1]
