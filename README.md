# Customer Churn Prediction — MLOps End-to-End

**Stack**: Python, Pandas, scikit-learn, FastAPI, DuckDB, pytest, pre-commit, GitHub Actions

Este proyecto demuestra un pipeline completo para **predicción de churn**:
- Limpieza y exploración de datos (EDA)
- Ingeniería de características
- Entrenamiento y evaluación de modelos
- Servir predicciones vía API (FastAPI)
- CI/CD con GitHub Actions (lint + tests automáticos)

## Estructura
```bash
├── data/
├── notebooks/
├── src/churn_mlops/
├── tests/
├── models/
├── configs/
└── README.md
```

## Instalación 
python3 -m venv .venv
source .venv/bin/activate  # en macOS/Linux
pip install -r requirements.txt

## Results (tiny demo)
Small illustrative run on a 5-row sample (not representative; overfits by design).

| Metric   | Value |
|---------:|------:|
| Accuracy | 1.00  |
| ROC-AUC  | 1.00  |

> Reproduce:
> ```bash
> python src/churn_mlops/train.py
> cat models/metrics.json
> ```