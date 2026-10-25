"""Clean data and create reproducible train/test CSVs."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
OUT = ROOT / "artifacts"
TARGET = "ProdTaken"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    # Remove generated index and identifier columns: they do not generalize as predictors.
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No valid binary target rows found.")
    # Normalize placeholder strings to missing values; model pipeline imputes them.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
        df[col] = df[col].replace(["Unknown", "unknown", "NA", "N/A"], pd.NA)
    train, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"Cleaned dataset: {len(df):,} rows; features: {df.shape[1]-1}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Train positive rate: {train[TARGET].mean():.3f}")
    print(f"Test positive rate: {test[TARGET].mean():.3f}")
    print(f"Saved splits to {OUT}")

if __name__ == "__main__":
    main()
