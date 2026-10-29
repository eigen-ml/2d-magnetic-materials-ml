# Batch 2 and control group: first run (proposal, not fixed yet)

Run on 2026-10-30 with v4 (`dft/CRITERIA.md`, items 6-7). Ata still has to confirm the rules below. Until then,
nothing in this folder is fixed and no DFT calculation uses it.

```bash
python scripts/04_discover.py --model-dir models/v4 --known-materials data/v4/known_materials.txt \
    --results-dir results/v4 --all-scores
python scripts/06_filter_candidates.py --candidates results/v4/screened_candidates.csv \
    --training data/v4/c2db_lowest_hull.csv --output-dir dft/batch2 --metal-cap 3 \
    --exclude-selection dft/selection.json --control-candidates results/v4/all_scores.csv
```

S2 counts templates among the v4 training compositions (C2DB only). The batch 1 formulas are skipped.

## Batch 2

Funnel: 2,172 above the v4 threshold, 564 pass S1, 323 pass S1-S4, 188 pass S1-S9. All 10 of batch 1 are among
the 188, so v3 and v4 agree on batch 1.

With at most 3 per magnetic metal, only **9** fit: every candidate holds two magnetic metals, and 6 metals × 3
slots fill after 9 picks.

| Rank | Formula | Oxidation states | Alexandria hull (eV/atom) |
| --- | --- | --- | --- |
| 1 | MnCrBr4 | Mn+2 Cr+2 Br-1 | 0.056 |
| 2 | Mn2CrSe4 | Mn+3 Cr+2 Se-2 | 0.018 |
| 3 | Mn2VTe4 | Mn+2 V+4 Te-2 | 0.027 |
| 4 | CrNiI4 | Cr+2 Ni+2 I-1 | no entry |
| 5 | FeCoO3 | Fe+3 Co+3 O-2 | 0.071 |
| 6 | FeCoS3 | Fe+3 Co+3 S-2 | 0.077 |
| 7 | VFe2Se4 | V+4 Fe+2 Se-2 | 0.011 |
| 8 | VNiI4 | V+2 Ni+2 I-1 | no entry |
| 9 | CoNi2I4 | Co+2 Ni+1 I-1 | no entry |

Points to decide:
- CoNi2I4 needs Ni+1. It passes S3 (Ni+1 is above 1% of nickel's ICSD entries), but it is the least plausible.
- Three iodides have no Alexandria entry, so S9 passes them by default.
- A cap of 4 gives 10: MnCrBr4, Mn2CrSe4, Mn2VTe4, MnRuO3, CrNiI4, CrFeO3, FeCoS3, VFe2Se4, VFeSe3, VNiI4. It drops
  CoNi2I4 and FeCoO3, but lets Mn, Cr, Fe and V appear 4 times.

## Control group

Rule as run: v4 score below 0.3, contains V, Cr, Mn, Fe, Co or Ni, passes S2, S3, S4, S8 and S9 (S5 and S6 not
required), lowest score first, one per family, cation set and magnetic metal. Of 98 such compositions, 76 pass
S2-S4 and 53 pass all five.

Picked: **TaNi2Te4** (0.009), **VW2O4** (0.143), **Zr2FeSe4** (0.143), and **MnBr2** fixed. Only 3 because the
one-per-family and one-per-metal rules run out (two per family gives TaNi2Te4, VW2O4, Zr2FeTe4, CrW2Se4). Two of them need rare oxidation states (W+2, Zr+3; ICSD likelihood 0.003 and
0.006), so a DFT result on them says little about the model.

The model gives low scores mostly to Ni compounds. With a plausibility floor (likelihood ≥ 0.05) the pool is
almost all Ni:

| Variant | Picks |
| --- | --- |
| A, as run | TaNi2Te4, VW2O4, Zr2FeSe4 |
| B, likelihood ≥ 0.05, one per family and cation set | ZrNi2Te4, CuNi2Se4, HfNi2S4, NiPdSe3 |
| C, likelihood ≥ 0.1, one per family and cation set | ZrNi2Te4, Ni2PtSe4, HfNi2S4, Ni2S3 |
| D, likelihood ≥ 0.05, two per family, one per cation set and magnetic metal | ZrNi2Te4, Co2CuTe4 |

The likelihood is the ICSD-based product that S3 already computes (`oxidation_likelihood` in
`candidates_final.csv`). MnBr2 stays in every variant.
