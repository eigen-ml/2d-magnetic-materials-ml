# v4: C2DB-only model

C2DB labels come from DFT. V2DB labels are ML predictions of C2DB-style labels, and adding them made the model
worse on DFT labels (`results/v3/README.md`, Mn diagnosis). So from 2026-10-30, v4 is the main model
(`dft/CRITERIA.md`, item 7). v3 stays as the comparison.

## Data

- `c2db.db` (2025-07-08), one label per formula from its lowest-hull structure: 13,251 compositions, 3,321
  magnetic (25.1%). Label: total moment of the stored FM state above 0.1 μB, as before.
- C2DB stores no moment for a non-spin-polarized calculation. Its own `is_magnetic` key is 0 for 3,060 of the
  3,061 such structures, so they are labelled 0. The thesis table did the same by writing a moment of 0.
- Against the thesis table, the label differs for 184 formulas (1.4%), 28 of them in the test split.
- Split: the v3 split (`--split-like data/v3/combined_net_moment.csv`), in the same row order. All models below
  are tested on the same 2,604 C2DB formulas. 8,502 train, 2,145 validation.

## Same test formulas, same labels

Labels: lowest-hull structure. Each model uses its own validation-selected threshold. Full numbers, also against
the thesis-table labels: `comparison.json` (`scripts/10_compare_c2db_models.py`).

| Model | Training data | Threshold | ROC-AUC | Precision | Recall | F1 | Score ≥ 0.98 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 | C2DB + V2DB | 0.667 | 0.900 | 0.864 | 0.611 | 0.716 | 187/193 = 0.969 |
| C2DB only | thesis table | 0.423 | 0.941 | 0.827 | 0.786 | 0.806 | 121/124 = 0.976 |
| **v4** | C2DB only, lowest hull | 0.411 | 0.946 | 0.804 | 0.822 | 0.813 | 155/158 = 0.981 |
| Rule: contains Mn, Cr, Fe, Co, V or Ni | – | – | 0.824 | 0.699 | 0.758 | 0.727 | – |

The rule's ROC-AUC comes from its 0/1 score.

**Run-to-run spread.** Training the two C2DB-only models 10 times with permuted training rows gives ROC-AUC
0.945 ± 0.001 (lowest hull) and 0.942 ± 0.001 (thesis table), F1 0.805 ± 0.007 and 0.808 ± 0.004. So
dropping V2DB gains about 0.045 in ROC-AUC and 0.09 in F1. The lowest-hull labels add about 0.003 in ROC-AUC, and
F1 stays the same within the spread.

**Reproducing 2026-10-29.** The one-off diagnosis reported ROC-AUC 0.939 for the C2DB-only model on thesis-table
labels. The same code in the research Docker image gives 0.936, and so does this pipeline
(`models/v3_experiments/`, not tracked). The first run was outside the image. Gradient boosting fits differ slightly
between platforms. ROC-AUC moves by a few thousandths. Recall and F1 move by up to 0.02, because the
validation-selected threshold shifts too (0.41 there, 0.423 here). So small differences between these models mean
nothing.

## Screening

`04_discover.py` with v4 scores the same 2,815 unseen compositions as v3. 2,172 pass the threshold and 563 are
tied at a calibrated score of 1.000. Mn is in 47 of the top 50 and Cr in 28. On the C2DB test split, 96% of the Mn
compositions are magnetic (Cr 86%, Fe 85%, V 64%, Co 55%, Ni 42%), so a model that fits the C2DB label better pushes
Mn further up. The top of the ranking carries little information, and batch 2 relies on the filters and the
per-metal cap (`dft/CRITERIA.md`, item 7).

## Files

- `models/v4/`: model, threshold, metadata (input checksum, split reference, test metrics).
- `screened_candidates.csv`, `candidate_shortlist.csv`, `screening_summary.json`: v4 screening.
- `comparison.json`: the table above, the label sets and the run-to-run spread.

Reproduce (in the research image; `c2db.db` from `DATASET_NOTES.md`):

```bash
python scripts/09_c2db_audit.py --db /path/to/c2db.db
python scripts/02_prepare_data.py --c2db-only --c2db data/external/c2db_formulas.csv --c2db-structures lowest-hull \
    --output data/v4/c2db_lowest_hull.csv --known-output data/v4/known_materials.txt
python scripts/02_prepare_data.py --c2db-only --output data/v4/c2db_thesis_table.csv \
    --known-output data/v4/known_thesis_table.txt
python scripts/03_train_model.py --data data/v4/c2db_lowest_hull.csv --split-like data/v3/combined_net_moment.csv \
    --model-dir models/v4 --model-version v4.0_c2db_lowest_hull
python scripts/03_train_model.py --data data/v4/c2db_thesis_table.csv --split-like data/v3/combined_net_moment.csv \
    --model-dir models/v3_experiments/c2db_thesis_table --model-version v4_repro_c2db_thesis_table
python scripts/04_discover.py --model-dir models/v4 --known-materials data/v4/known_materials.txt --results-dir results/v4
python scripts/10_compare_c2db_models.py
```
