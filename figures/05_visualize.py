#!/usr/bin/env python3
"""
05_visualize.py
Thesis-ready figures for ML-Driven Discovery of 2D Magnetic Materials.
Author: Ataberk Özturk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

# Color palette
COLORS = {
    'magnetic': '#e74c3c',
    'non_magnetic': '#3498db', 
    'c2db': '#2ecc71',
    'v2db': '#9b59b6',
    'highlight': '#f39c12'
}

MAGNETIC_ELEMENTS = {'Mn', 'Cr', 'Fe', 'Co', 'Ni', 'V', 'Cu', 'Ti'}

# ============================================================
# FIGURE 1: Dataset Overview
# ============================================================
def fig1_dataset_overview():
    """Dataset composition and class distribution."""
    df = pd.read_csv(DATA_DIR / "combined_dataset.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) Source distribution
    source_counts = df['source'].value_counts()
    wedges_a, texts_a, autotexts_a = axes[0].pie(
        source_counts, 
        autopct='%1.1f%%',
        colors=[COLORS['c2db'], COLORS['v2db']], 
        explode=[0.02, 0],
        textprops={'fontsize': 11}
    )
    axes[0].legend(wedges_a, source_counts.index, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.12), ncol=2)
    axes[0].set_title('(a) Data Source Distribution', fontweight='bold')
    
    # (b) Class distribution
    class_counts = df['is_magnetic'].value_counts()
    labels = ['Non-magnetic', 'Magnetic']
    wedges_b, texts_b, autotexts_b = axes[1].pie(
        class_counts, 
        autopct='%1.1f%%',
        colors=[COLORS['non_magnetic'], COLORS['magnetic']], 
        explode=[0, 0.05],
        textprops={'fontsize': 11}
    )
    axes[1].legend(wedges_b, labels, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.12), ncol=2)
    axes[1].set_title('(b) Class Distribution', fontweight='bold')
    
    # (c) Class by source - FIXED: moved legend outside
    cross = pd.crosstab(df['source'], df['is_magnetic'], normalize='index') * 100
    cross.columns = ['Non-magnetic', 'Magnetic']
    cross.plot(kind='bar', ax=axes[2], 
               color=[COLORS['non_magnetic'], COLORS['magnetic']], 
               rot=0, width=0.7, edgecolor='white', linewidth=1)
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_xlabel('')
    axes[2].set_title('(c) Class Distribution by Source', fontweight='bold')
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    axes[2].set_ylim(0, 110)
    
    # Add percentage labels on bars - positioned inside bars
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt='%.1f%%', fontsize=9, label_type='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_dataset_overview.png")
    plt.close()
    print("✓ fig1_dataset_overview.png")
# ============================================================
# FIGURE 2: Feature Importance
# ============================================================
def fig2_feature_importance():
    """Top 20 most important features from the model."""
    feat_imp = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    top20 = feat_imp.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by magnetic element
    colors = []
    for feat in top20['feature']:
        elem = feat.replace('elem_', '')
        colors.append(COLORS['magnetic'] if elem in MAGNETIC_ELEMENTS else COLORS['non_magnetic'])
    
    bars = ax.barh(range(len(top20)), top20['importance'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f.replace('elem_', '') for f in top20['feature']])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title('Top 20 Most Important Features', fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top20['importance'])):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['magnetic'], label='Magnetic Transition Metal'),
        Patch(facecolor=COLORS['non_magnetic'], label='Other Element')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(0, max(top20['importance']) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_feature_importance.png")
    plt.close()
    print("✓ fig2_feature_importance.png")

# ============================================================
# FIGURE 3: Discovery Results
# ============================================================
def fig3_discovery_results():
    """Distribution of predictions and confidence levels."""
    df = pd.read_csv(RESULTS_DIR / "novel_candidates.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Probability distribution
    axes[0].hist(df['probability'], bins=30, color=COLORS['magnetic'], 
                 edgecolor='white', alpha=0.85, linewidth=0.5)
    axes[0].axvline(x=0.5, color='#2c3e50', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[0].axvline(x=0.9, color=COLORS['c2db'], linestyle='--', linewidth=2, label='High Confidence (0.9)')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Number of Candidates')
    axes[0].set_title('(a) Distribution of Magnetic Probability', fontweight='bold')
    axes[0].legend(loc='upper left')
    
    # (b) Confidence breakdown
    conf_order = ['Very High', 'High', 'Medium', 'Low']
    conf_counts = df['confidence'].value_counts().reindex(conf_order)
    colors_conf = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c']
    bars = axes[1].bar(conf_order, conf_counts, color=colors_conf, edgecolor='white', linewidth=1)
    axes[1].set_xlabel('Confidence Level')
    axes[1].set_ylabel('Number of Candidates')
    axes[1].set_title('(b) Candidates by Confidence Level', fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, conf_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                     str(int(count)), ha='center', fontweight='bold', fontsize=11)
    
    axes[1].set_ylim(0, max(conf_counts) * 1.12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_discovery_results.png")
    plt.close()
    print("✓ fig3_discovery_results.png")

# ============================================================
# FIGURE 4: Top Candidates Heatmap
# ============================================================
def fig4_top_candidates():
    """Composition heatmap of top 30 novel candidates."""
    df = pd.read_csv(RESULTS_DIR / "novel_candidates.csv").head(30)
    
    from pymatgen.core import Composition
    
    # Build element matrix
    all_elements = set()
    for formula in df['formula']:
        try:
            comp = Composition(formula)
            all_elements.update([el.symbol for el in comp.elements])
        except:
            pass
    
    all_elements = sorted(all_elements)
    matrix = np.zeros((len(df), len(all_elements)))
    
    for i, formula in enumerate(df['formula']):
        try:
            comp = Composition(formula)
            frac = comp.get_el_amt_dict()
            total = sum(frac.values())
            for el, amt in frac.items():
                j = all_elements.index(el)
                matrix[i, j] = amt / total
        except:
            pass
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(matrix, xticklabels=all_elements, yticklabels=df['formula'],
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Element Fraction'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Top 30 Novel Magnetic Candidates - Composition Heatmap', fontweight='bold')
    ax.set_xlabel('Element')
    ax.set_ylabel('Formula')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_top_candidates.png")
    plt.close()
    print("✓ fig4_top_candidates.png")

# ============================================================
# FIGURE 5: Pipeline Summary Table
# ============================================================
def fig5_pipeline_summary():
    """Professional summary table of the entire pipeline."""
    with open(RESULTS_DIR / "discovery_summary.json", "r") as f:
        summary = json.load(f)
    
    with open(MODEL_DIR / "metadata.json", "r") as f:
        model_meta = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Table data with sections
    table_data = [
        # Dataset section
        ['DATASET', '', ''],
        ['Total Materials', '141,171', 'C2DB + V2DB merged'],
        ['Magnetic Materials', '14,601', '10.3%'],
        ['Non-magnetic Materials', '126,570', '89.7%'],
        ['Class Imbalance Ratio', '8.7:1', 'Handled with hybrid weighting'],
        # Model section
        ['MODEL PERFORMANCE', '', ''],
        ['Algorithm', 'Gradient Boosting', 'n_estimators=200, max_depth=8'],
        ['ROC-AUC Score', f"{model_meta['metrics']['roc_auc']:.4f}", 'Excellent discrimination'],
        ['Recall (Sensitivity)', f"{model_meta['metrics']['recall']:.4f}", '90.6% magnetic detected'],
        ['Precision', f"{model_meta['metrics']['precision']:.4f}", 'Acceptable false positive rate'],
        ['F1-Score', f"{model_meta['metrics']['f1_score']:.4f}", 'Balanced performance'],
        # Discovery section
        ['DISCOVERY RESULTS', '', ''],
        ['Candidates Generated', f"{summary['total_generated']:,}", 'Systematic enumeration'],
        ['Known Materials Filtered', f"{summary['known_filtered']}", 'Novelty filter applied'],
        ['Novel Candidates Screened', f"{summary['novel_screened']:,}", 'Unique compositions'],
        ['Predicted Magnetic', f"{summary['magnetic_predicted']:,}", f"{summary['magnetic_predicted']/summary['novel_screened']*100:.1f}% hit rate"],
        ['Very High Confidence (≥0.95)', f"{summary['very_high_confidence']}", 'Top priority for DFT'],
        ['High Confidence (0.9-0.95)', f"{summary['high_confidence']}", 'Secondary priority'],
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Value', 'Description'],
        cellLoc='left',
        loc='center',
        colWidths=[0.32, 0.18, 0.40]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#1a252f')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style section headers
    section_rows = [1, 6, 12]
    for row in section_rows:
        for j in range(3):
            table[(row, j)].set_facecolor('#34495e')
            table[(row, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors for data
    data_rows = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
    for i, row in enumerate(data_rows):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(3):
            table[(row, j)].set_facecolor(color)
    
    # Title
    ax.set_title('Pipeline Summary: ML-Driven Discovery of 2D Magnetic Materials\n', 
                 fontsize=15, fontweight='bold', pad=10)
    
    plt.savefig(FIGURES_DIR / "fig5_pipeline_summary.png", facecolor='white', edgecolor='none')
    plt.close()
    print("✓ fig5_pipeline_summary.png")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("📊 THESIS FIGURES GENERATION")
    print("   ML-Driven Discovery of 2D Magnetic Materials")
    print("=" * 60 + "\n")
    
    fig1_dataset_overview()
    fig2_feature_importance()
    fig3_discovery_results()
    fig4_top_candidates()
    fig5_pipeline_summary()
    
    print(f"\n{'=' * 60}")
    print(f"✅ All figures saved to: {FIGURES_DIR.absolute()}/")
    print("=" * 60)
    print("\n📁 Generated Files:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"   • {f.name}")
    print()

if __name__ == "__main__":
    main()
