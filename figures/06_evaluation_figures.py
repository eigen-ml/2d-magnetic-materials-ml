#!/usr/bin/env python3
"""ROC, confusion-matrix and leave-element-out figures for the current model.

Needs models/ from scripts/03_train_model.py and results/ from
scripts/05_grouped_validation.py. The held-out test split is rebuilt with the
same seed as training, so the confusion matrix matches models/metadata.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "combined_dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _load_train_module():
    path = REPO_ROOT / "scripts" / "03_train_model.py"
    spec = importlib.util.spec_from_file_location("train_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def held_out_test_scores(train) -> tuple[np.ndarray, np.ndarray, float]:
    """Rebuild the untouched test split and score it with the saved model."""
    feature_df, _ = train.featurize_dataset(pd.read_csv(DATA))
    _, _, X_test, _, _, y_test = train.split_dataset(feature_df)
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    X_test = X_test.reindex(columns=metadata["feature_names"], fill_value=0.0)
    # joblib uses pickle: only load the model this repository produced.
    classifier = joblib.load(MODEL_DIR / "model.joblib")
    scores = classifier.predict_proba(X_test)[:, 1]
    return y_test.to_numpy(), scores, float(metadata["threshold"])


def fig_roc(y_test, test_scores, oof: pd.DataFrame, loeo_scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    curves = [
        ("Held-out test split", y_test, test_scores, "#1f4e79", "-"),
        ("Chemical-system grouped CV", oof["is_magnetic"], oof["chemsys_score"], "#7f7f7f", "--"),
    ]
    palette = ["#c55a11", "#e3a21a", "#8e44ad"]
    worst = (
        loeo_scores.groupby("held_out_element")
        .apply(lambda g: roc_auc_score(g["is_magnetic"], g["score"]), include_groups=False)
        .sort_values()
        .index[:3]
    )
    for element, color in zip(worst, palette):
        part = loeo_scores[loeo_scores["held_out_element"] == element]
        curves.append((f"No {element} in training", part["is_magnetic"], part["score"], color, ":"))
    for label, y, s, color, style in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, style, color=color, lw=1.6, label=f"{label} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: random-style splits vs unseen elements")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(FIGURES_DIR / "eval_roc.png")
    plt.close(fig)


def _draw_cm(ax, cm: np.ndarray, title: str) -> None:
    normalized = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}\n({normalized[i, j]:.1%})",
                ha="center", va="center", fontsize=9,
                color="white" if normalized[i, j] > 0.6 else "black",
            )
    ax.set_xticks([0, 1], ["non-magnetic", "magnetic"])
    ax.set_yticks([0, 1], ["non-magnetic", "magnetic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("V2DB label")
    ax.set_title(title, fontsize=10)


def fig_confusion(y_test, test_scores, threshold, oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cm_test = confusion_matrix(y_test, (test_scores >= threshold).astype(int), labels=[0, 1])
    _draw_cm(axes[0], cm_test, f"Held-out test split\n(threshold {threshold:.3f} from validation)")
    pred = (oof["chemsys_score"] >= oof["chemsys_threshold"]).astype(int)
    cm_group = confusion_matrix(oof["is_magnetic"], pred, labels=[0, 1])
    _draw_cm(axes[1], cm_group, "Chemical-system grouped CV\n(pooled out-of-fold, per-fold thresholds)")
    for ax in axes:
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_confusion_matrix.png")
    plt.close(fig)


def fig_leave_element_out(loeo: pd.DataFrame, random_recall: float) -> None:
    loeo = loeo.dropna(subset=["roc_auc"])
    x = np.arange(len(loeo))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.38
    ax.bar(x - width / 2, loeo["roc_auc"], width, color="#1f4e79", label="ROC-AUC")
    ax.bar(x + width / 2, loeo["recall"], width, color="#c55a11", label="Recall at validation threshold")
    ax.axhline(random_recall, color="#c55a11", lw=0.8, ls="--")
    ax.text(-0.5, random_recall + 0.02, f"recall under random CV ({random_recall:.2f})",
            ha="left", fontsize=8, color="#c55a11")
    ax.set_xticks(x, [f"no {e}\n(n={n:,})" for e, n in zip(loeo["fold"], loeo["n"])], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on held-out compositions")
    ax.set_title("Leave-one-element-out: every composition containing the element is held out")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_leave_element_out.png")
    plt.close(fig)


def fig_feature_importance() -> None:
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv").head(15)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.barh(importance["feature"].str.removeprefix("elem_"), importance["importance"], color="#1f4e79")
    ax.invert_yaxis()
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Top 15 features (current model)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eval_feature_importance.png")
    plt.close(fig)


def main() -> int:
    required = [
        DATA,
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "feature_importance.csv",
        RESULTS_DIR / "cv_out_of_fold_scores.csv.gz",
        RESULTS_DIR / "grouped_validation_folds.csv",
        RESULTS_DIR / "grouped_validation.json",
        RESULTS_DIR / "leave_element_out_scores.csv.gz",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("ERROR: missing " + ", ".join(missing) + ". Run scripts 02, 03 and 05 first.", file=sys.stderr)
        return 2

    train = _load_train_module()
    y_test, test_scores, threshold = held_out_test_scores(train)
    oof = pd.read_csv(RESULTS_DIR / "cv_out_of_fold_scores.csv.gz")
    folds = pd.read_csv(RESULTS_DIR / "grouped_validation_folds.csv")
    loeo = folds[folds["scheme"] == "leave_element_out"].reset_index(drop=True)
    summary = json.loads((RESULTS_DIR / "grouped_validation.json").read_text(encoding="utf-8"))

    loeo_scores = pd.read_csv(RESULTS_DIR / "leave_element_out_scores.csv.gz")
    fig_roc(y_test, test_scores, oof, loeo_scores)
    fig_confusion(y_test, test_scores, threshold, oof)
    fig_leave_element_out(loeo, summary["random"]["recall_mean"])
    fig_feature_importance()
    print(f"Wrote eval_*.png to: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
