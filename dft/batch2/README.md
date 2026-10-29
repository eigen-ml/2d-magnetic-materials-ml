# Batch 2 and control group

Fixed on 2026-10-30, before any DFT calculation (`dft/CRITERIA.md`, items 6, 7 and 9). The first run, with a
cap of 3 per metal and a looser control rule, is kept unchanged as checkpoint A in `checkpoint-A/`.

```bash
python scripts/04_discover.py --model-dir models/v4 --known-materials data/v4/known_materials.txt \
    --results-dir results/v4 --all-scores
python scripts/06_filter_candidates.py --candidates results/v4/screened_candidates.csv \
    --training data/v4/c2db_lowest_hull.csv --output-dir dft/batch2 --metal-cap 4 \
    --exclude-selection dft/selection.json --control-candidates results/v4/all_scores.csv
```

## Batch 2 (v4)

Funnel: 2,172 above the v4 threshold, 564 pass S1, 323 pass S1-S4, 188 pass S1-S9. All 10 of batch 1 are among
the 188, so v3 and v4 agree on batch 1. They are skipped here.

| Rank | Formula | Oxidation states | Alexandria bulk hull (eV/atom) |
| --- | --- | --- | --- |
| 1 | MnCrBr4 | Mn+2 Cr+2 Br-1 | 0.056 |
| 2 | Mn2CrSe4 | Mn+3 Cr+2 Se-2 | 0.018 |
| 3 | Mn2VTe4 | Mn+2 V+4 Te-2 | 0.027 |
| 4 | MnRuO3 | Mn+2 Ru+4 O-2 | 0.079 |
| 5 | CrNiI4 | Cr+2 Ni+2 I-1 | no entry |
| 6 | CrFeO3 | Cr+3 Fe+3 O-2 | 0.005 |
| 7 | FeCoS3 | Fe+3 Co+3 S-2 | 0.077 |
| 8 | VFe2Se4 | V+4 Fe+2 Se-2 | 0.011 |
| 9 | VFeSe3 | V+3 Fe+3 Se-2 | 0.080 |
| 10 | VNiI4 | V+2 Ni+2 I-1 | no entry |

Notes:
- CrNiI4 and VNiI4 have no Alexandria entry, so S9 passes them by default.
- CrFeO3 is in Materials Project as a bulk oxide (Fe2O3 and Cr2O3 are corundum oxides), and MnRuO3 exists only as
  a computed bulk in Alexandria.
  Neither comes from layered (van der Waals) chemistry. S8 found 2D prototypes for both, and stage P shows
  whether a monolayer holds.

## Control group

| Formula | v4 score | Oxidation states | ICSD likelihood |
| --- | --- | --- | --- |
| ZrNi2Te4 | 0.020 | Zr+4 Ni+2 Te-2 | 0.43 |
| CuNi2Se4 | 0.034 | Cu+2 Ni+3 Se-2 | 0.06 |
| HfNi2S4 | 0.109 | Hf+4 Ni+2 S-2 | 0.50 |
| NiPdSe3 | 0.109 | Ni+2 Pd+4 Se-2 | 0.10 |
| MnBr2 | fixed | Mn+2 Br-1 | – |

Pool: 98 compositions below 0.3 with a magnetic metal, 76 pass S2-S4, 16 pass everything including the
likelihood floor. All four picks contain Ni: the model gives low scores almost only to Ni compounds. MnBr2 checks
whether the protocol finds a known antiferromagnetic ground state. The control group gets relax and the NM/FM/AFM
comparison, no phonons.
