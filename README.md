# Customer Churn Prediction — MLOps End-to-End

**Tech Stack:** Python, Pandas, scikit-learn, FastAPI, DuckDB, pytest, pre-commit, GitHub Actions

This repository demonstrates a complete **end-to-end MLOps pipeline** for customer churn prediction:

- **Data preparation:** Cleaning, deduplication, and feature engineering.
- **Exploratory Data Analysis (EDA):** (planned, to be added in notebooks/)
- **Modeling:** Training and evaluating a baseline classifier.
- **Serving:** (planned) Serving predictions via FastAPI.
- **CI/CD:** (planned) Automated linting and testing with GitHub Actions.

## Project Structure
```bash
├── data/               # raw and processed data (gitignored)
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── src/churn_mlops/    # main code (data.py, features.py, train.py)
├── tests/              # unit tests (to be added)
├── models/             # trained models + metrics.json
├── configs/            # training configs (future)
└── README.md
```

## Quickstart
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSV(s) in data/raw/ and run:
python src/churn_mlops/data.py
python src/churn_mlops/features.py

# 4. Train model
python src/churn_mlops/train.py
cat models/metrics.json

## Results (tiny demo)
This example was run on a small 5-row dataset (purely illustrative, will overfit by design):
| Metric   | Value |
|---------:|------:|
| Accuracy | 1.00  |
| ROC-AUC  | 1.00  |

*Note*: Results will change as the dataset grows and more robust evaluation is added.

