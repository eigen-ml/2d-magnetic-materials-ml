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

## Stage P (30 Oct)

```bash
docker compose --profile research run --rm research python scripts/07_build_structures.py \
    --selection dft/batch2/selection.json --out dft/batch2/structures --groups main,control
.venv-mlip/bin/python scripts/08_mlip_rank.py --structures dft/batch2/structures
```

CrI3 and VCl3 are not rebuilt; batch 1 has them. Three prototypes per composition, 45 MACE relaxations, all
converged. A relaxed prototype counts as a held monolayer when it is one bonded piece (covalent radii x 1.2) and its
thickness stays within 0.7-1.3 of the start; only held prototypes go to VASP (`layer_holds`, `to_vasp` in
`structures/summary_mlip.csv`). The 0.7-1.3 band is a working choice, not a physical criterion.

| Formula | Group | To VASP (prototype, space group at start) | ΔE second, meV/atom | Rejected |
| --- | --- | --- | --- | --- |
| MnCrBr4 | main | CaI4Pb P2/m; CaI4Pb P2_1/m | 1.0 | P2_1/c flattened (0.66) |
| Mn2CrSe4 | main | CrGa2Se4 P-3m1 | – | Pmna split in two |
| Mn2VTe4 | main | Fe2Se4Ti P-1; Te4TiV2 C2/m | 4.7 | – |
| MnRuO3 | main | O3SnZn P-31m | – | both O3PdSe flattened (0.51, 0.62) |
| CrNiI4 | main | CaI4Pb P2/m; P2_1/m | 6.3 | – |
| CrFeO3 | main | AlO3Sb P31m | – | MnO3Tl P-31m flattened (0.67) |
| FeCoS3 | main | GaInS3 Pmn2_1 | – | – |
| VFe2Se4 | main | Te4TiV2 C2/m; Fe2Se4Ti P-1 | 10.8 | – |
| VFeSe3 | main | InS3Sb P2_1/m | – | – |
| VNiI4 | main | CaI4Pb P2/m; P2_1/m | 10.7 | – |
| ZrNi2Te4 | control | Fe2Se4Ti P-1 | – | – |
| CuNi2Se4 | control | Ga2S4Zn P3m1; In2MgSe4 P-1 | 8.2 | – |
| HfNi2S4 | control | Fe2Se4Ti P-1 | – | – |
| NiPdSe3 | control | NiPSe3 P-31m | – | lowest, CaS3Sn Pmmm, thickened (1.38) |
| MnBr2 | control | Br2Mn P-3m1 (its own structure); Br2Ti P2_1/m | 0.2 | – |

Notes:
- CrFeO3 holds in one prototype (P31m, 15 atoms, thickness unchanged); the other two either flatten or lie
  256 meV/atom higher. MnRuO3 holds only in P-31m, and its thickness grows by 25 %, close to the band edge. Both stay
  on the list (criteria item 9); a monolayer that fails in VASP is reported, not replaced.
- MnBr2 lands on its known CdI2-type layer first, a sanity check on the ranking. The P2_1/m runner-up relaxes to the
  same thickness and is probably the same layer; VESTA will show it.
- MnCrBr4, CrNiI4 and VNiI4 have only the CaI4Pb family as prototype, with the largest substitution mismatch of
  the batch (0.55-0.56).
- MACE ignores spin order; it only picks prototypes. No MACE energy is a result.
