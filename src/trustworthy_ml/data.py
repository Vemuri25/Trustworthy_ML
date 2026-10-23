from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "adult.csv"


def load_dataset() -> pd.DataFrame:
    """Load the cached dataset or download it securely from OpenML."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print(f"Loading cached data: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("Downloading Adult Census Income dataset from OpenML...")

    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    df = dataset.frame.copy()
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved to: {DATA_PATH}")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== FIRST FIVE ROWS ===")
    print(df.head().to_string())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    print("\n=== DUPLICATE ROWS ===")
    print(f"Duplicate count: {df.duplicated().sum():,}")

    target = next(
        (c for c in df.columns if c.lower() in {"class", "income"}),
        None,
    )

    print("\n=== TARGET DISTRIBUTION ===")
    if target:
        print(df[target].value_counts(dropna=False).to_string())
    else:
        print("Target not identified. Review the column names above.")


if __name__ == "__main__":
    dataframe = load_dataset()
    inspect_dataset(dataframe)
