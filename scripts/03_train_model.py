#!/usr/bin/env python3
"""
03_train_model.py
Gradient Boosting model egitimi (Sadece V2DB icin).
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             roc_auc_score, confusion_matrix, 
                             precision_recall_curve)
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import joblib
import json
from pathlib import Path

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def get_element_fractions(formula: str) -> dict:
    """Formulu element fraksiyonlarina cevir"""
    try:
        comp = Composition(formula)
        frac = comp.get_el_amt_dict()
        total = sum(frac.values())
        result = {f"elem_{k}": v/total for k, v in frac.items()}
        result["natoms"] = comp.num_atoms
        return result
    except:
        return {}

def featurize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Dataset'i featurize et"""
    print("Featurization yapiliyor...")
    
    feature_rows = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        features = get_element_fractions(row["formula_normalized"])
        if features:
            features["is_magnetic"] = row["is_magnetic"]
            feature_rows.append(features)
            valid_indices.append(idx)
    
    feature_df = pd.DataFrame(feature_rows).fillna(0)
    print(f"  > {len(feature_df):,} malzeme featurize edildi")
    
    return feature_df, valid_indices

def train_model(X_train, y_train, class_counts):
    """Model egit (Class weighting ile)"""
    print("\nModel egitiliyor...")
    
    ratio = class_counts[0] / class_counts[1]
    
    # Sadece class weighting, source weight (hybrid) kaldirildi
    CLASS_WEIGHT_FACTOR = 0.35
    
    sample_weights = np.where(y_train == 1, ratio * CLASS_WEIGHT_FACTOR, 1.0)
    
    print(f"  Class weight: Magnetic x{ratio * CLASS_WEIGHT_FACTOR:.2f}")
    
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=8,
        subsample=0.85,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        verbose=1
    )
    
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    return clf, {"class_weight_factor": CLASS_WEIGHT_FACTOR}

def find_optimal_threshold(clf, X_test, y_test):
    """Optimal threshold bul (recall oncelikli)"""
    print("\nOptimal threshold bulunuyor...")
    
    y_proba = clf.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    target_recall = 0.90
    min_precision = 0.80
    best_threshold = 0.5
    best_f1 = 0
    
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= target_recall and p >= min_precision:
            f1 = 2 * (p * r) / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
    
    print(f"  > Optimal threshold: {best_threshold:.4f}")
    
    return best_threshold, y_proba

def evaluate_model(y_test, y_pred, y_proba):
    """Model performansini degerlendir"""
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision_mag = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_mag = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_mag = 2 * (precision_mag * recall_mag) / (precision_mag + recall_mag) if (precision_mag + recall_mag) > 0 else 0
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANSI")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Magnetic Class:")
    print(f"    Precision: {precision_mag:.4f}")
    print(f"    Recall:    {recall_mag:.4f}")
    print(f"    F1-Score:  {f1_mag:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn:,}  |  FP: {fp:,}")
    print(f"    FN: {fn:,}  |  TP: {tp:,}")
    
    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": float(precision_mag),
        "recall": float(recall_mag),
        "f1_score": float(f1_mag),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def save_artifacts(clf, threshold, feature_names, metrics, weighting_params, class_counts):
    """Model ve metadata kaydet"""
    print("\nKaydediliyor...")
    
    # Model
    joblib.dump(clf, MODEL_DIR / "model.joblib")
    np.save(MODEL_DIR / "threshold.npy", threshold)
    
    # Feature importance
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    
    # Metadata
    metadata = {
        "model_version": "v2.0_v2db_only",
        "weighting": weighting_params,
        "threshold": float(threshold),
        "metrics": metrics,
        "class_distribution": {
            "non_magnetic": int(class_counts[0]),
            "magnetic": int(class_counts[1])
        },
        "n_features": len(feature_names),
        "feature_names": list(feature_names)
    }
    
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  > {MODEL_DIR}/model.joblib")
    print(f"  > {MODEL_DIR}/threshold.npy")
    print(f"  > {MODEL_DIR}/feature_importance.csv")
    print(f"  > {MODEL_DIR}/metadata.json")

def main():
    print("=" * 60)
    print("MODEL EGITIMI (V2DB)")
    print("=" * 60)
    
    # 1. Veri yukle
    print("\nVeri yukleniyor...")
    df = pd.read_csv(DATA_DIR / "combined_dataset.csv")
    print(f"  > {len(df):,} malzeme")
    
    # 2. Featurize
    feature_df, _ = featurize_dataset(df)
    
    # 3. Train/test split
    y = feature_df["is_magnetic"]
    X = feature_df.drop(columns=["is_magnetic"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSplit: {len(X_train):,} train | {len(X_test):,} test")
    
    class_counts = Counter(y)
    print(f"  Non-magnetic: {class_counts[0]:,}")
    print(f"  Magnetic:     {class_counts[1]:,}")
    
    # 4. Model egit
    clf, weighting_params = train_model(X_train, y_train, class_counts)
    
    # 5. Threshold optimizasyonu
    threshold, y_proba = find_optimal_threshold(clf, X_test, y_test)
    y_pred = (y_proba >= threshold).astype(int)
    
    # 6. Degerlendirme
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    # 7. Kaydet
    save_artifacts(clf, threshold, X.columns, metrics, weighting_params, class_counts)
    
    # 8. Top features
    print("\nTOP 15 ONEMLI ELEMENTLER:")
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    magnetic_elements = {'Mn', 'Cr', 'Fe', 'Co', 'Ni', 'V', 'Cu'}
    
    for feat, imp in feat_imp.items():
        elem = str(feat).replace('elem_', '')
        marker = '[M]' if elem in magnetic_elements else '   '
        bar = '#' * int(imp * 100)
        print(f"  {marker} {elem:4s}: {bar} {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    main()
