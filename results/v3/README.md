# v3 revision (work in progress)

Before sending candidates to DFT, I went back over the thesis pipeline. This folder holds the revised model's
screening output and a comparison with the thesis ranking. The thesis results stay frozen under the
`thesis-2026` tag and `results/legacy/`.

## What changed

| Issue in the thesis pipeline | Change |
| --- | --- |
| "Magnetic" meant different things per source. C2DB rows were labelled by total moment (\|m\| > 0.1); V2DB counted FM and AFM as 1 | Default target is now **net moment** (V2DB `FM`, C2DB \|m\| > 0.1). The old target is still available (`--target any_magnetic`). Correction (2026-10-30): this is still not one definition, because C2DB stores the FM state of antiferromagnets too. See "C2DB label check" below |
| Conflicting V2DB formulas were dropped from the reference set, so they came back as "novel" | `known_materials` is built from every input formula |
| `natoms` (atom count of the reduced formula) partly encodes the source: 5- and 7-atom formulas exist only in C2DB | Off by default (`--feature-set elements`) |
| Scores were not calibrated, and nobody had measured what a 0.98 score means | Isotonic calibration on the validation split, precision per score band, metrics per source |
| A2BX4 used ordered metal pairs: Mn2CrX4 was generated, Cr2MnX4 never | Both orderings |

## Three models on the same split

All three are evaluated on the same 20% test split. C2DB is the only part with DFT labels; V2DB labels are
themselves ML predictions.

| Model | Test ROC-AUC (all) | C2DB ROC-AUC | C2DB recall | C2DB precision at score ≥ 0.98 |
| --- | --- | --- | --- | --- |
| A, thesis-like (any_magnetic, natoms, C2DB ×6, raw) | 0.989 | 0.922 | 0.760 | 69/70 = 0.986 |
| B, net moment + natoms, calibrated | 0.989 | 0.905 | 0.719 | 179/182 = 0.984 |
| **C, net moment, elements only, calibrated (`models/v3`)** | 0.987 | 0.893 | 0.596 | 187/193 = 0.969 |

Full numbers: `revision_metrics.json`.

## What I take from it

- The overall AUC near 0.99 is carried by V2DB. On real DFT labels (C2DB) the model is around 0.89-0.92.
- Dropping `natoms` barely changes the candidates. 215 of the 219 thesis candidates at ≥ 0.98 are still at
  ≥ 0.98 in v3 (Spearman 0.83 on the thesis list). The high scores come from compositions with two magnetic
  metals, and the A2BX4 and ABX3 templates are exactly those. So `natoms` was not the main driver.
- On known C2DB materials, a calibrated score ≥ 0.98 means a measured hit rate of about 97%. That does not
  carry over directly to hypothetical compositions with no known structure.
- So the model alone does not pick good DFT candidates. Charge balance, novelty against external databases and
  literature, and whether a 2D structure can be built at all have to do that. That is the next step.

## C2DB label check (2026-10-30)

`scripts/09_c2db_audit.py` compares the thesis C2DB table with the C2DB file `c2db.db` (2025-07-08, 16,905
structures). Output: `results/c2db_audit/summary.json`.

- The thesis table is this release. All 13,251 formulas are in the file, the file has no formula the table lacks,
  and every table row matches a structure in the file exactly in `ehull`.
- C2DB keeps magnetic materials in their FM state. Known antiferromagnets carry the FM-state moment per cell:
  MnPS3 9.97, MnPSe3 9.75, FePS3 7.94, NiPS3 3.91 and MnBr2 5.00 μB, with no antiparallel site moments. So a C2DB 1
  means "spin-polarized when ordered ferromagnetically", not "ferromagnetic ground state". V2DB keeps AFM as its
  own class, which the net-moment target counts as 0. FM versus AFM has to come from DFT.
- The table keeps one structure per formula, but not the lowest-hull one. Of 1,892 formulas with several
  structures, it kept the lowest in 321 and the highest in 1,246 (CrI3: the 0.197 eV/atom structure, not the
  0.000 one). The label differs from the lowest-hull structure for 184 of 13,251 formulas (1.4%). `ehull` and
  `hform` from this table are not used anywhere in the pipeline.
- Structures without a stored moment are non-spin-polarized calculations: C2DB's own `is_magnetic` key is 0 for
  3,060 of the 3,061. The table writes a moment of 0 for them, which gives the right label.

## Mn diagnosis (2026-10-29)

Nine of the first main 10 contain Mn. The numbers below come from a one-off diagnosis on the C2DB part of the v3
test split (2,604 compositions, DFT labels from the thesis table). The C2DB-only model becomes v4. Reproducible numbers on the same test
formulas, from the pipeline itself: `results/v4/README.md`.

- **The Mn preference is in the C2DB label.** 95.4% of the 175 Mn compositions are magnetic, against 61.4% of the
  533 with Cr, Fe, Co, V or Ni but no Mn, and 7.7% of the 1,678 with none of these metals, Cu or Ti. In the whole release
  (lowest-hull structure per formula) 97.5% of 865 Mn formulas are magnetic.
- **V2DB makes the model worse on DFT labels.** Same split and hyperparameters:

  | Model | ROC-AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- |
  | v3 (C2DB + V2DB, threshold 0.667) | 0.893 | 0.853 | 0.596 | 0.702 |
  | C2DB only (threshold 0.41; run outside the Docker image, 0.936 inside it) | 0.939 | 0.822 | 0.788 | 0.805 |
  | Rule: contains Mn, Cr, Fe, Co, V or Ni | 0.819 | 0.698 | 0.747 | 0.722 |

- **The top of the ranking is flat.** 203 of the 208 candidates that pass S1-S9 sit on three calibrated score
  steps between 0.986 and 0.989. Swapping the metals does not move the score either: Mn2CrS4, Fe2CrS4 and V2CrS4 all
  get 0.989. Within the top, the order carries little information.

## Files

- `screened_candidates.csv`, `candidate_shortlist.csv`, `screening_summary.json`: v3 screening. Sorted by
  calibrated score, with ties broken by the raw score (isotonic calibration flattens the top).
- `compare_thesis.csv`: every screened formula, with thesis, A, B and v3 scores and ranks.
- `revision_metrics.json`: test metrics of A, B and C.

Reproduce (inside `docker-compose --profile research run --rm research`):

```bash
python scripts/02_prepare_data.py --output data/v3/combined_net_moment.csv --known-output data/v3/known_materials.txt
python scripts/03_train_model.py --data data/v3/combined_net_moment.csv --model-dir models/v3
python scripts/04_discover.py --model-dir models/v3 --known-materials data/v3/known_materials.txt --results-dir results/v3
```
