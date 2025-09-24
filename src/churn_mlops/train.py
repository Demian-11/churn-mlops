# src/churn_mlops/train.py
"""
Entrena un modelo binario de churn usando TU dataset real:
- Lee data/processed/dataset_features.csv
- Usa solo columnas numéricas como X (ignora textos e IDs)
- Soporta datasets pequeños (evalúa en train si no alcanza para split)
Guarda:
- models/model.joblib
- models/metrics.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
INPUT = PROC / "dataset_features.csv"


def load_df() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"No encuentro {INPUT}. Corre data.py y features.py primero.")
    df = pd.read_csv(INPUT)
    return df


def build_xy(df: pd.DataFrame, target: str = "churn"):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no está en las columnas: {df.columns.tolist()}")

    # Quita columnas no numéricas y columnas que no aportan (ej. IDs)
    X_df = df.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
    # Si existe customer_id numérico, quítalo explícitamente
    for col in ["customer_id", "id", "customerid"]:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    if X_df.shape[1] == 0:
        raise ValueError("No quedaron columnas numéricas para entrenar (X está vacío).") # por si falla

    y = df[target].astype(int).values
    X = X_df.values
    return X, y, X_df.columns.tolist()


def safe_auc(y_true, y_proba):
    # Para AUC necesitamos al menos dos clases presentes
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None


def train_and_eval(X, y, seed: int = 42):
    # Si el dataset es muy pequeño, o hay muy pocas instancias por clase,
    # entrenamos y evaluamos en el MISMO conjunto (solo para probar el flujo),
    # como en este ejemplo. En un caso real, siempre hacer holdout.
    n = len(y)
    unique, counts = np.unique(y, return_counts=True)
    min_class = counts.min()

    use_holdout = (n >= 10) and (min_class >= 2)
    clf = LogisticRegression(max_iter=1000)

    if not use_holdout:
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = float(accuracy_score(y, y_pred))
        # Para AUC necesitamos predict_proba y dos clases
        auc = None
        if hasattr(clf, "predict_proba") and len(np.unique(y)) >= 2:
            y_proba = clf.predict_proba(X)[:, 1]
            auc = safe_auc(y, y_proba)
        return clf, {"mode": "train_only", "accuracy": acc, "roc_auc": auc}
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed)
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, y_pred))
        auc = None
        if hasattr(clf, "predict_proba") and len(np.unique(yte)) >= 2:
            y_proba = clf.predict_proba(Xte)[:, 1]
            auc = safe_auc(yte, y_proba)
        return clf, {"mode": "holdout", "accuracy": acc, "roc_auc": auc}


def main():
    df = load_df()
    X, y, cols = build_xy(df, target="churn")
    clf, metrics = train_and_eval(X, y)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS / "model.joblib")
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(
            {
                "features_used": cols,
                **metrics,
                "n_samples": int(len(y)),
            },
            f,
            indent=2,
        )

    print(f"Modelo guardado en: {MODELS / 'model.joblib'}")
    print(f"Métricas: {metrics}")
    print(f"Features usadas: {cols}")


if __name__ == "__main__":
    main()