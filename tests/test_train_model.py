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
