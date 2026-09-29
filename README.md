# Composition-only screening of magnetic 2D materials

[![tests](https://github.com/eigen-ml/2d-magnetic-materials-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/eigen-ml/2d-magnetic-materials-ml/actions/workflows/tests.yml)

A gradient boosting classifier that predicts the magnetic label of 2D materials in the Virtual 2D Materials Database (V2DB) from chemical composition alone (elemental fractions and atom count of the reduced formula). It started as my undergraduate thesis at Ankara University; this is a revised version (see [Thesis version and this revision](#thesis-version-and-this-revision)). The model is a cheap first filter; it does not replace structure-aware or DFT calculations, and the labels it learns are themselves predictions made by the V2DB workflow.

## Results

127,024 unique compositions after cleaning (9,772 labelled magnetic, 7.7%). All numbers below come from files in `results/` and `models/metadata.json` and can be regenerated with the commands in [Reproducing](#reproducing).

| Evaluation | ROC-AUC | Average precision | Recall | Precision |
| --- | --- | --- | --- | --- |
| Held-out test split (20%, threshold chosen on validation) | 0.999 | – | 0.943 | 0.949 |
| Random stratified 5-fold CV | 0.999 ± 0.000 | 0.990 ± 0.001 | 0.939 | 0.959 |
| 5-fold CV grouped by chemical system | 0.999 ± 0.000 | 0.989 ± 0.001 | 0.939 | 0.955 |
| Rule baseline: "contains V, Cr, Mn, Fe, Co or Ni" | – | – | 1.000 | 0.137 |

Leave-one-element-out (every composition containing the element is removed from training and used as the test set):

| Held-out element | Test compositions | Magnetic share | ROC-AUC | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| V | 6,190 | 15% | 0.987 | 0.231 | 1.000 |
| Cr | 8,673 | 18% | 0.942 | 0.137 | 1.000 |
| Mn | 18,644 | 41% | 0.878 | 0.014 | 1.000 |
| Fe | 16,093 | 10% | 0.984 | 0.192 | 1.000 |
| Co | 12,960 | 12% | 0.986 | 0.199 | 1.000 |
| Ni | 16,445 | 6% | 0.997 | 0.333 | 1.000 |
| Cu | 6,066 | 3% | 0.999 | 0.965 | 0.734 |
| Ti | 10,845 | 2% | 0.998 | 0.913 | 0.752 |

What this means:

- Within the chemistry it has seen, the model reproduces the V2DB label very closely, and it is much better than the simple "has a magnetic 3d metal" rule (precision 0.95 vs 0.14).
- Grouping by chemical system barely changes the score. V2DB has 76,925 distinct element sets and the median set holds a single composition, so this grouping ends up close to a random split. It is reported, but it is not a hard test.
- Leave-one-element-out is the hard test, and the model mostly fails it. When V, Cr, Mn, Fe, Co or Ni is missing from training, recall at the chosen threshold drops to 1–33%. The ranking (ROC-AUC) stays fairly high, so the scores still carry some order, but the model cannot flag magnetism that comes from an element it has not seen. Mn alone accounts for about half of the feature importance.
- In short: the classifier is useful for ranking compositions made of elements already in the training data. It should not be used to predict magnetism for chemistries built on new magnetic elements.

<p>
  <img src="figures/eval_roc.png" width="48%" alt="ROC curves for the held-out test split, grouped CV and leave-element-out cases">
  <img src="figures/eval_leave_element_out.png" width="50%" alt="ROC-AUC and recall for each held-out element">
</p>
<p>
  <img src="figures/eval_confusion_matrix.png" width="62%" alt="Confusion matrices for the held-out test split and the grouped CV">
  <img src="figures/eval_feature_importance.png" width="36%" alt="Top 15 feature importances">
</p>

Feature importance is impurity based. It shows what the model uses, not a physical mechanism.

### Thesis version and this revision

The undergraduate thesis (Ankara University, 2026) used a merged C2DB + V2DB dataset of 141,171 compositions and reported ROC-AUC 0.986 and recall 0.906 for magnetic ordering. That version is tagged [`thesis-2026`](../../tree/thesis-2026), and its model, data, figures and tables are kept unchanged under `models/legacy_hybrid/`, `data/legacy_hybrid/`, `figures/legacy/` and `results/legacy/`.

This revision, done after the thesis, restricts the data to V2DB only, selects the decision threshold on a separate validation split and adds grouped and leave-one-element-out validation. The two sets of numbers use different data and evaluation protocols, so they are not directly comparable.

## Pipeline

```text
V2DB CSV (formula, prototype, magnetic label)
  -> schema and label checks, reduce formulas with pymatgen
  -> drop formulas whose prototypes disagree on the label (2,231 removed)
  -> features: elemental fractions + reduced-formula atom count (52 features)
  -> stratified split 64 / 16 / 20 (train / validation / test)
  -> class-weighted GradientBoostingClassifier (scikit-learn)
  -> choose threshold on validation, evaluate once on test
  -> random, chemical-system grouped and leave-one-element-out CV
  -> score a small enumerated set of compositions not in V2DB
```

## Repository layout

```text
scripts/02_prepare_data.py        clean V2DB, build consensus labels
scripts/03_train_model.py         train, pick threshold, evaluate on test
scripts/04_discover.py            score enumerated compositions
scripts/05_grouped_validation.py  random / grouped / leave-element-out CV
figures/05_visualize.py           dataset and screening figures (fig1-fig5)
figures/06_evaluation_figures.py  ROC, confusion matrix, leave-element-out (eval_*)
api/main.py                       FastAPI service for the trained model
models/                           current model, threshold and metadata
results/                          CV summaries and screening outputs
tests/                            pytest suite (runs in CI)
*/legacy*                         artefacts of the old hybrid run
```

## Reproducing

Python 3.10+.

```bash
git clone https://github.com/eigen-ml/2d-magnetic-materials-ml.git
cd 2d-magnetic-materials-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_prepare_data.py        # ~10 s
python scripts/03_train_model.py         # ~30 s
python scripts/05_grouped_validation.py  # 18 model fits, ~1.5 min on 12 cores
python scripts/04_discover.py
python figures/05_visualize.py
python figures/06_evaluation_figures.py

python -m pytest -q
```

The committed `models/model.joblib` was trained with scikit-learn 1.7.2. joblib files are pickles; only load models you trained yourself or trust.

## REST API

The trained model is served with FastAPI. The API loads the same artefacts as `scripts/04_discover.py`.

```bash
docker compose up --build
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"formulas": ["CrI3", "MoS2"]}'
```

Endpoints: `GET /health`, `GET /model` (version, threshold, test metrics), `POST /predict` (up to 1000 formulas per request). Interactive docs at `http://localhost:8000/docs`. Formulas with elements outside the training set are returned with an error instead of a score. Scores are uncalibrated model outputs, not probabilities that a real material is magnetic.

For a public HTTPS deployment (API behind Caddy) see [deploy/README.md](deploy/README.md).

<!-- Public demo URL goes here once it is deployed. -->

## Screening output

`scripts/04_discover.py` builds 2,528 binary and ternary formulas from fixed element lists, removes the 212 that already exist in V2DB and scores the remaining 2,316. 432 pass the validation threshold (`results/candidate_shortlist.csv`). These are compositions to look at with a structure-aware workflow, not new materials: no crystal structure, stability or charge balance is checked, and given the leave-element-out results the scores are only as good as the training chemistry around them.

## Limitations

- The target is the V2DB magnetic label, which is itself an ML prediction trained on C2DB data, not a DFT calculation done here.
- Composition-only features ignore structure, coordination, oxidation state, magnetic configuration and spin–orbit coupling.
- Removing formulas with conflicting labels across prototypes gives a cleaner target but narrows the question.
- Scores are not calibrated, and the model does not generalise to unseen magnetic elements (see the leave-element-out table).

## Data and license

Code is MIT licensed. The license does not cover V2DB or files derived from it; see [DATASET_NOTES.md](DATASET_NOTES.md) for provenance and checksums. V2DB is described by its authors as CC BY 4.0. If you use the data, cite:

- M. C. Sorkun, S. Astruc, J. M. V. A. Koelman, S. Er, *V2DB: Virtual 2D Materials Database*, Harvard Dataverse (2020), [doi:10.7910/DVN/SNCZF4](https://doi.org/10.7910/DVN/SNCZF4).
- M. C. Sorkun et al., "An artificial intelligence-aided virtual screening recipe for two-dimensional materials discovery", *npj Computational Materials* 6, 106 (2020), [doi:10.1038/s41524-020-00375-7](https://doi.org/10.1038/s41524-020-00375-7).

## Contact

Ata Berk Öztürk · ataberkozturk5a@gmail.com
