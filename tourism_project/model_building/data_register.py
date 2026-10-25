"""Validate and summarize the tourism dataset before downstream jobs run."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "tourism_project" / "data" / "tourism.csv"
EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    if "ProdTaken" not in df:
        raise ValueError("Target column ProdTaken is required.")
    print(f"Dataset validation passed: {DATA_PATH.relative_to(ROOT)}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print("Target distribution (count):")
    print(df["ProdTaken"].value_counts(dropna=False).sort_index().to_string())
    print("Column summary:")
    print(df.describe(include="all").transpose().to_string())

if __name__ == "__main__":
    main()
