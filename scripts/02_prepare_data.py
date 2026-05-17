#!/usr/bin/env python3
"""
02_prepare_data.py
V2DB verisini isler ve known_materials listesi olusturur.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition
import re
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def normalize_formula(formula: str) -> str | None:
    """Formulu standart hale getir (alfabetik sirali)"""
    try:
        comp = Composition(formula)
        return comp.reduced_formula
    except:
        return None

def process_v2db(filepath: str) -> pd.DataFrame:
    """V2DB verisini isle"""
    print("\nV2DB yukleniyor...")
    df = pd.read_csv(filepath)
    
    # Material sutununu normalize et
    df["formula_normalized"] = df["Material"].apply(normalize_formula)
    df = df.dropna(subset=["formula_normalized"])
    
    # is_magnetic olustur
    df["is_magnetic"] = df["Material_is_magnetic"].apply(
        lambda x: 1 if str(x).lower() in ['true', '1', 't', 'yes', '1.0'] else 0
    )
    
    # V2DB'de magmom yok, ehull ve hform farkli isimde
    df["magmom"] = np.nan  
    df["ehull"] = df.get("Energy_above_convex_hull", np.nan)
    df["hform"] = df.get("Heat_of_formation", np.nan)
    
    print(f"  > {len(df):,} malzeme")
    print(f"  > Manyetik: {df['is_magnetic'].sum():,} ({df['is_magnetic'].mean()*100:.1f}%)")
    
    return df[["formula_normalized", "magmom", "ehull", "hform", "is_magnetic"]]

def create_known_materials(df: pd.DataFrame, output_file: str):
    """Bilinen malzemelerin listesini olustur"""
    known = set(df["formula_normalized"].unique())
    
    with open(output_file, "w") as f:
        for formula in sorted(known):
            f.write(f"{formula}\n")
    
    print(f"\nKnown materials kaydedildi: {output_file}")
    print(f"  > {len(known):,} unique formul")
    
    return known

def main():
    print("=" * 60)
    print("VERI HAZIRLAMA (V2DB)")
    print("=" * 60)
    
    input_file = DATA_DIR / "v2db.csv"
    if not input_file.exists():
        print(f"HATA: {input_file} bulunamadi!")
        return
        
    # 1. Verileri yukle ve isle
    df_combined = process_v2db(input_file)
    
    # 2. Duplicate kontrolu (ayni formul varsa)
    print("Duplicate kontrolu...")
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["formula_normalized"], keep="first")
    after = len(df_combined)
    print(f"  > {before - after:,} duplicate kaldirildi")
    
    # 3. Istatistikler
    print("\n" + "=" * 60)
    print("VERISETI ISTATISTIKLERI")
    print("=" * 60)
    print(f"  Toplam malzeme:  {len(df_combined):,}")
    
    print(f"\n  Sinif dagilimi:")
    mag_count = df_combined["is_magnetic"].sum()
    nonmag_count = len(df_combined) - mag_count
    print(f"    Manyetik:     {mag_count:,} ({mag_count/len(df_combined)*100:.1f}%)")
    print(f"    Non-manyetik: {nonmag_count:,} ({nonmag_count/len(df_combined)*100:.1f}%)")
    if mag_count > 0:
        print(f"    Imbalance:    {nonmag_count/mag_count:.1f}:1")
    
    # 4. Known materials listesi
    known_out = DATA_DIR / "known_materials.txt"
    known = create_known_materials(df_combined, known_out)
    
    # 5. Kaydet
    out_file = DATA_DIR / "combined_dataset.csv"
    df_combined.to_csv(out_file, index=False)
    print(f"\nDataset kaydedildi: {out_file}")
    
    print("\n" + "=" * 60)
    print("HAZIRLAMA TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    main()
