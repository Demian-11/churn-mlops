"""
Toma data/processed/dataset.csv y crea nuevas columnas (features) útiles para modelado.
Guarda el resultado en data/processed/dataset_features.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
INPUT = PROC / "dataset.csv"
OUTPUT = PROC / "dataset_features.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Ejemplo 1
    if "contract" in df.columns:
        df["contract_len"] = df["contract"].str.len()

    # Ejemplo 2: one-hot encoding 
    if "contract" in df.columns:
        dummies = pd.get_dummies(df["contract"], prefix="contract", dtype=int)
        dummies.columns = (
            dummies.columns.str.lower()
                           .str.replace(r"[^a-z0-9]+", "_", regex=True)
                           .str.strip("_")
        )
        df = pd.concat([df.drop(columns=["contract"]), dummies], axis=1)
    
    return df

def main():
    print(f"Dataset limpio: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"   - Filas: {len(df):,} | Columnas: {len(df.columns)}")
    
    feat_df = build_features(df)
    print(f"Nuevas columnas creadas: {len(feat_df.columns) - len(df.columns)}")
    
    feat_df.to_csv(OUTPUT, index=False)
    print(f"Guardado como: {OUTPUT}")

if __name__ == "__main__":
    main()