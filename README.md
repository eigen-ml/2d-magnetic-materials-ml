# Project V2 - 2D Magnetic Material Discovery Pipeline

This repository provides a lightweight, fully reproducible machine learning pipeline for predicting magnetic ordering in two-dimensional (2D) materials using only chemical composition as input. The project integrates PAW-based DFT data (V2DB) with a Gradient Boosting classifier and performs high-throughput screening to identify novel magnetic candidates.

> Note: The first version of this project used data from the Computational 2D Materials Database (C2DB). However, C2DB has a strict CC-BY-NC-SA license. Because of this, I removed all C2DB datasets, related scripts, and hybrid weighting functions from this repository. This is the license-free version of the project. You just need to bring your own V2DB-style data to run the codes.

---

## Key Features
- Composition-based feature engineering (Pymatgen)
- Gradient Boosting model optimized for class imbalance
- High-throughput candidate screening for novel magnetic materials
- Reproducible pipeline with modular Python scripts
- Fully customizable: use your own dataset

---

## Project Structure
```text
Project_V2/
│
├── data/                    # User-provided datasets
│   └── example_v2db.csv
│
├── scripts/                 # Core pipeline scripts
│   ├── 02_prepare_data.py
│   ├── 03_train_model.py
│   ├── 04_discover.py
│   └── utils.py
│
├── models/                  # (Optional) Saved model artifacts
│   └── .gitkeep
│
├── figures/                 # Generated plots (optional)
│
├── results/                 # Prediction outputs
│   └── .gitkeep
│
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
```

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/eigen-ml/2d-magnetic-materials-ml.git
cd 2d-magnetic-materials-ml
```

### 2. Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Input Data
This project requires a dataset formatted similarly to:

### V2DB-style dataset (`your_v2db.csv`)
Must contain:
```text
Material, Material_is_magnetic
```

Place your dataset in the `data/` folder (named as `v2db.csv` for the scripts to find it):
```text
data/
 └── v2db.csv
```

---

## Running the Pipeline
### 1. Data Preparation
```bash
python scripts/02_prepare_data.py
```
This produces a cleaned dataset named `combined_dataset.csv`.

### 2. Train the Model
```bash
python scripts/03_train_model.py
```
Outputs:
- `models/model.joblib`
- `models/feature_importance.csv`
- `models/threshold.npy`

### 3. Run High-Throughput Screening
```bash
python scripts/04_discover.py
```
Outputs go to `results/`:
- `novel_candidates.csv`
- `discovery_summary.json`

---

## Example Results
- ROC-AUC: ~0.986
- Recall (Magnetic class): ~90%
- Model correctly identifies Mn, Cr, Fe as dominant magnetic drivers
- ~377 high-confidence magnetic candidates can typically be generated

---

## Method Summary
The pipeline uses:
- Pymatgen Composition Parser -> elemental fractions + atom count
- Gradient Boosting Classifier
- Stratified train/test split (handles class imbalance)
- Novelty filtering -> removes known compositions before discovery

---

## License
This project is released under the MIT License, which allows unrestricted academic, personal, or commercial use.

---

## Citation
If you use this repository in research, please cite:
```text
Öztürk, A. B. (2025). Machine learning framework for predicting
magnetic ordering in 2D materials using composition-only features.
```

---

## Contact
For questions or collaboration inquiries:
```text
Author: Ata Berk Öztürk
Email: ataberkozturk5a@gmail.com
```

---

## Contribution
Pull requests are welcome. You may:
- add new descriptor methods
- extend structural features
- contribute alternative ML models
- improve stability filtering (e_hull prediction)

---

Thank you for checking out Project V2!
If this repository helped you, please consider giving a star on GitHub.
