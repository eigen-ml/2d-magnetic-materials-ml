# v3 revision (work in progress)

Before sending candidates to DFT, I went back over the thesis pipeline. This folder holds the revised model's
screening output and a comparison with the thesis ranking. The thesis results stay frozen under the
`thesis-2026` tag and `results/legacy/`.

## What changed

| Issue in the thesis pipeline | Change |
| --- | --- |
| "Magnetic" meant different things per source. C2DB rows were labelled by total moment (\|m\| > 0.1), so a zero-moment antiferromagnet was 0; V2DB counted FM and AFM as 1 | Default target is now **net moment** (V2DB `FM`, C2DB \|m\| > 0.1), the only definition both sources share. The old target is still available (`--target any_magnetic`) |
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
