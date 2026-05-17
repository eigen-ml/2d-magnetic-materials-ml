#!/usr/bin/env python3
"""
04_discover.py
Novelty filter ile yeni manyetik 2D malzeme kesfi.
"""

import joblib
import pandas as pd
import numpy as np
import json
from pymatgen.core import Composition
from itertools import combinations, product
from pathlib import Path

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_known_materials() -> set:
    """Bilinen malzemeleri yukle"""
    known = set()
    with open(DATA_DIR / "known_materials.txt", "r") as f:
        for line in f:
            known.add(line.strip())
    return known

def load_model():
    """Model ve metadata yukle"""
    clf = joblib.load(MODEL_DIR / "model.joblib")
    threshold = np.load(MODEL_DIR / "threshold.npy")
    
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return clf, float(threshold), metadata

def generate_candidates(magnetic_metals, anions, other_metals=None) -> list:
    """Sistematik aday uret"""
    candidates = set()
    
    # Binary: MX, MX2, MX3
    for m, x in product(magnetic_metals, anions):
        candidates.add(f"{m}{x}")
        candidates.add(f"{m}{x}2")
        candidates.add(f"{m}{x}3")
        candidates.add(f"{m}2{x}3")
    
    # Ternary: M1M2X3, M1M2X4
    for (m1, m2), x in product(combinations(magnetic_metals, 2), anions):
        candidates.add(f"{m1}{m2}{x}3")
        candidates.add(f"{m1}{m2}{x}4")
        candidates.add(f"{m1}2{m2}{x}4")
    
    # Mixed metal: MM'X3
    if other_metals:
        for m, m2, x in product(magnetic_metals, other_metals, anions):
            candidates.add(f"{m}{m2}{x}3")
            candidates.add(f"{m}{m2}2{x}4")
    
    # Mixed anion: MX2Y
    for m, (x, y) in product(magnetic_metals, combinations(anions, 2)):
        candidates.add(f"{m}{x}2{y}")
        candidates.add(f"{m}{x}{y}2")
    
    return list(candidates)

def normalize_formula(formula: str) -> str | None:
    """Formulu normalize et"""
    try:
        return Composition(formula).reduced_formula
    except:
        return None

def featurize(formula: str, feature_names: list) -> dict | None:
    """Tek formulu featurize et"""
    try:
        comp = Composition(formula)
        frac = comp.get_el_amt_dict()
        total = sum(frac.values())
        
        feats = {f"elem_{k}": v/total for k, v in frac.items()}
        feats["natoms"] = comp.num_atoms
        
        # Template'e uyumlu hale getir
        return {col: feats.get(col, 0) for col in feature_names}
    except:
        return None

def main():
    print("=" * 60)
    print("* NOVEL MAGNETIC MATERIALS DISCOVERY")
    print("=" * 60)
    
    # 1. Model yukle
    print("\nModel yukleniyor...")
    clf, threshold, metadata = load_model()
    feature_names = metadata["feature_names"]
    print(f"  > Threshold: {threshold:.4f}")
    print(f"  > Features: {len(feature_names)}")
    
    # 2. Known materials yukle
    print("\nBilinen malzemeler yukleniyor...")
    known_materials = load_known_materials()
    print(f"  > {len(known_materials):,} bilinen formul")
    
    # 3. Adaylar uret
    print("\nAdaylar uretiliyor...")
    MAGNETIC_METALS = ["Mn", "Cr", "Fe", "Co", "Ni", "V", "Cu", "Ti"]
    ANIONS = ["S", "Se", "Te", "O", "Cl", "Br", "I", "F"]
    OTHER_METALS = ["Zr", "Hf", "Nb", "Ta", "Mo", "W", "Ru", "Pd", "Pt"]
    
    raw_candidates = generate_candidates(MAGNETIC_METALS, ANIONS, OTHER_METALS)
    print(f"  > {len(raw_candidates):,} ham aday")
    
    # 4. Normalize ve novelty filter
    print("\nNovelty filter uygulaniyor...")
    novel_candidates = []
    known_count = 0
    invalid_count = 0
    
    for formula in raw_candidates:
        normalized = normalize_formula(formula)
        if normalized is None:
            invalid_count += 1
            continue
        
        if normalized in known_materials:
            known_count += 1
            continue
        
        novel_candidates.append(normalized)
    
    # Duplicate'leri kaldir
    novel_candidates = list(set(novel_candidates))
    
    print(f"  > {known_count:,} zaten biliniyor (atlandi)")
    print(f"  > {invalid_count:,} gecersiz formul")
    print(f"  > {len(novel_candidates):,} NOVEL aday")
    
    # 5. Featurize
    print("\n* Featurization...")
    X_list = []
    valid_formulas = []
    
    for formula in novel_candidates:
        feats = featurize(formula, feature_names)
        if feats:
            X_list.append(feats)
            valid_formulas.append(formula)
    
    X = pd.DataFrame(X_list)
    print(f"  > {len(X):,} featurize edildi")
    
    # 6. Prediction
    print("\nTahminler yapiliyor...")
    probs = clf.predict_proba(X)[:, 1]
    
    results = []
    for formula, prob in zip(valid_formulas, probs):
        if prob >= threshold:
            if prob >= 0.95:
                conf = "Very High"
            elif prob >= 0.90:
                conf = "High"
            elif prob >= 0.80:
                conf = "Medium"
            else:
                conf = "Low"
            
            results.append({
                "formula": formula,
                "probability": prob,
                "confidence": conf
            })
    
    df_results = pd.DataFrame(results).sort_values("probability", ascending=False)
    
    print(f"\n    {len(df_results):,} NOVEL manyetik aday bulundu!")
    
    # 7. Istatistikler
    print("\n" + "=" * 60)
    print("DISCOVERY ISTATISTIKLERI")
    print("=" * 60)
    print(f"  Taranan novel aday:  {len(valid_formulas):,}")
    print(f"  Manyetik tahmin:     {len(df_results):,} ({len(df_results)/len(valid_formulas)*100:.1f}%)")
    print(f"\n  Confidence dagilimi:")
    for conf in ["Very High", "High", "Medium", "Low"]:
        count = len(df_results[df_results["confidence"] == conf])
        print(f"     {conf:10s}: {count:,}")
    
    # 8. Top 20
    print("\nTOP 20 NOVEL MANYETIK ADAYLAR:")
    print("-" * 50)
    
    for i, row in df_results.head(20).iterrows():
        rank = list(df_results.head(20).index).index(i) + 1
        bar = "#" * int(row["probability"] * 30)
        print(f"{rank:2d}. {row['formula']:15s} {bar} {row['probability']:.4f} [{row['confidence']}]")
    
    # 9. Element analizi
    print("\n* TOP 20'DEKI ELEMENT DAGILIMI:")
    elem_counts = {}
    for formula in df_results.head(20)["formula"]:
        try:
            comp = Composition(formula)
            for el in comp.elements:
                elem_counts[el.symbol] = elem_counts.get(el.symbol, 0) + 1
        except:
            pass
    
    for elem, count in sorted(elem_counts.items(), key=lambda x: -x[1])[:10]:
        marker = "[M]" if elem in MAGNETIC_METALS else "   "
        print(f"  {marker} {elem:3s}: {'#' * count} {count}")
    
    # 10. Kaydet
    print("\nSonuclar kaydediliyor...")
    
    df_results.to_csv(RESULTS_DIR / "novel_candidates.csv", index=False)
    
    # DFT-ready format
    df_dft = df_results.head(50).copy()
    df_dft["priority"] = range(1, len(df_dft) + 1)
    df_dft["suggested_method"] = "PBE+U"
    df_dft["notes"] = "Novel - not in V2DB"
    df_dft.to_csv(RESULTS_DIR / "dft_candidates.csv", index=False)
    
    # Summary
    summary = {
        "total_generated": len(raw_candidates),
        "known_filtered": known_count,
        "novel_screened": len(valid_formulas),
        "magnetic_predicted": len(df_results),
        "very_high_confidence": len(df_results[df_results["confidence"] == "Very High"]),
        "high_confidence": len(df_results[df_results["confidence"] == "High"]),
        "top_10": df_results.head(10)[["formula", "probability"]].to_dict("records")
    }
    
    with open(RESULTS_DIR / "discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  > {RESULTS_DIR}/novel_candidates.csv")
    print(f"  > {RESULTS_DIR}/dft_candidates.csv")
    print(f"  > {RESULTS_DIR}/discovery_summary.json")
    
    print("\n" + "=" * 60)
    print("DISCOVERY TAMAMLANDI!")
    print("=" * 60)
    print(f"\n{len(df_results[df_results['confidence'].isin(['Very High', 'High'])])} yuksek guvenilirlikli NOVEL aday DFT'ye hazir!")

if __name__ == "__main__":
    main()
