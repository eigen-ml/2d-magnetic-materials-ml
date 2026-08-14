#!/usr/bin/env python3
"""Generate figures from artefacts produced by the current V2DB-only pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymatgen.core import Composition


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

MAGNETIC_ELEMENTS = {"Mn", "Cr", "Fe", "Co", "Ni", "V", "Cu", "Ti"}
COLORS = {
    "magnetic": "#e74c3c",
    "non_magnetic": "#3498db",
    "other": "#7f8c8d",
    "highlight": "#f39c12",
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def require_files(paths: list[Path]) -> None:
    """Raise one actionable error listing all absent pipeline artefacts."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required artefact(s): "
            + ", ".join(missing)
            + ". Run data preparation, training, and screening first."
        )


def fig1_dataset_overview() -> None:
    """Plot composition-level class counts and proportions."""
    df = pd.read_csv(DATA_DIR / "combined_dataset.csv")
    counts = df["is_magnetic"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Non-magnetic", "Magnetic"]
    colors = [COLORS["non_magnetic"], COLORS["magnetic"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(labels, counts.values, color=colors)
    axes[0].set_ylabel("Number of reduced compositions")
    axes[0].set_title("(a) Composition-level labels", fontweight="bold")
    axes[0].bar_label(bars, labels=[f"{value:,}" for value in counts.values])

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[1].set_title("(b) Class proportions", fontweight="bold")
    fig.suptitle("Prepared V2DB Composition Dataset")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_dataset_overview.png")
    plt.close(fig)


def fig2_feature_importance() -> None:
    """Plot the twenty largest impurity-based feature importances."""
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv").head(20)
    colors = [
        COLORS["magnetic"]
        if feature.removeprefix("elem_") in MAGNETIC_ELEMENTS
        else COLORS["non_magnetic"]
        for feature in importance["feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        importance["feature"].str.removeprefix("elem_"),
        importance["importance"],
        color=colors,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Impurity-based feature importance")
    ax.set_title("Top Composition Features", fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_feature_importance.png")
    plt.close(fig)


def fig3_screening_results() -> None:
    """Plot score distribution and descriptive score-band counts."""
    candidates = pd.read_csv(RESULTS_DIR / "screened_candidates.csv")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        candidates["magnetic_score"],
        bins=30,
        color=COLORS["magnetic"],
        edgecolor="white",
    )
    axes[0].axvline(threshold, color="#2c3e50", linestyle="--", label=f"Threshold {threshold:.3f}")
    axes[0].set_xlabel("Model magnetic score (uncalibrated)")
    axes[0].set_ylabel("Candidate compositions")
    axes[0].set_title("(a) Selected score distribution", fontweight="bold")
    axes[0].legend()

    order = ["0.95-1.00", "0.90-0.95", "0.80-0.90", "threshold-0.80"]
    counts = candidates["score_band"].value_counts().reindex(order, fill_value=0)
    bars = axes[1].bar(order, counts.values, color=["#27ae60", "#2ecc71", "#f39c12", "#95a5a6"])
    axes[1].set_ylabel("Candidate compositions")
    axes[1].set_title("(b) Descriptive score bands", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].bar_label(bars)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_discovery_results.png")
    plt.close(fig)


def fig4_top_candidates() -> None:
    """Plot elemental fractions for the top thirty scored compositions."""
    candidates = pd.read_csv(RESULTS_DIR / "screened_candidates.csv").head(30)
    all_elements = sorted(
        {
            element.symbol
            for formula in candidates["formula"]
            for element in Composition(formula).elements
        }
    )
    matrix = np.zeros((len(candidates), len(all_elements)))
    for row_index, formula in enumerate(candidates["formula"]):
        amounts = Composition(formula).get_el_amt_dict()
        total = sum(amounts.values())
        for element, amount in amounts.items():
            matrix[row_index, all_elements.index(element)] = amount / total

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        matrix,
        xticklabels=all_elements,
        yticklabels=candidates["formula"],
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Element fraction"},
    )
    ax.set_title("Top Scored Composition Proposals", fontweight="bold")
    ax.set_xlabel("Element")
    ax.set_ylabel("Reduced formula")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_top_candidates.png")
    plt.close(fig)


def fig5_pipeline_summary() -> None:
    """Render a summary table using only current metadata and output values."""
    dataset = pd.read_csv(DATA_DIR / "combined_dataset.csv")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS_DIR / "screening_summary.json").read_text(encoding="utf-8"))
    metrics = metadata["test_metrics"]
    rows = [
        ["Prepared compositions", f"{len(dataset):,}"],
        ["Magnetic labels", f"{int(dataset['is_magnetic'].sum()):,}"],
        ["Features", str(metadata["n_features"])],
        ["Test ROC-AUC", f"{metrics['roc_auc']:.4f}"],
        ["Test precision", f"{metrics['precision']:.4f}"],
        ["Test recall", f"{metrics['recall']:.4f}"],
        ["Decision threshold", f"{metadata['threshold']:.4f}"],
        ["Unseen compositions scored", f"{summary['unseen_compositions_scored']:,}"],
        ["Selected at threshold", f"{summary['selected_at_threshold']:,}"],
        ["Scores calibrated", str(summary["scores_calibrated"])],
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Audited metric", "Value"],
        cellLoc="left",
        colWidths=[0.65, 0.25],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)
    ax.set_title("Current Pipeline Summary", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_pipeline_summary.png")
    plt.close(fig)


def main() -> int:
    """Generate all current figures and return a process exit code."""
    required = [
        DATA_DIR / "combined_dataset.csv",
        MODEL_DIR / "feature_importance.csv",
        MODEL_DIR / "metadata.json",
        RESULTS_DIR / "screened_candidates.csv",
        RESULTS_DIR / "screening_summary.json",
    ]
    try:
        require_files(required)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig1_dataset_overview()
        fig2_feature_importance()
        fig3_screening_results()
        fig4_top_candidates()
        fig5_pipeline_summary()
    except (FileNotFoundError, KeyError, ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote figures to: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
