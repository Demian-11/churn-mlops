# src/churn_mlops/data.py
"""
Carga CSVs desde data/raw/, hace una limpieza MUY básica y guarda data/processed/dataset.csv.
Mantiene nombres de columnas en snake_case y mapea 'Churn' Yes/No -> 1/0 si existe.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_FILE = DATA_PROCESSED / "dataset.csv"


def _snake_case_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def load_raw_csvs() -> pd.DataFrame:
    files = sorted(DATA_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron CSVs en {DATA_RAW}. "
            "Verifica que existan al menos un .csv en data/raw/ y vuelve a intentar."
        )
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _snake_case_cols(df)
    # Duplicados idénticos
    df = df.drop_duplicates()

    # Normaliza columna 'churn' si existe con valores Yes/No
    if "churn" in df.columns:
        if df["churn"].dtype == "object":
            df["churn"] = (
                df["churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
            )

        df["churn"] = pd.to_numeric(df["churn"], errors="coerce")

    
    df = df.dropna(axis=1, how="all")

    return df


def save_processed(df: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    print(f"Leyendo CSV(s) en: {DATA_RAW}")
    df = load_raw_csvs()
    print(f"   - Filas crudas: {len(df):,} | Columnas: {len(df.columns)}")

    df = basic_clean(df)
    print(f"Después de la limpieza: {len(df):,} filas | {len(df.columns)} columnas")
    if "churn" in df.columns:
        print("   - Columna 'churn' detectada")

    save_processed(df)
    print(f"Guardado como: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()