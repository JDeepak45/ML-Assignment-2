"""
Load the real UCI Wine Quality dataset (Cortez et al., 2009) from the raw
red/white CSVs, combine them, add a wine_type feature, and bucket the
numeric `quality` score (0-10) into a 3-class categorical target for
classification.

Source files (semicolon-delimited, as distributed by UCI):
    data_raw/winequality-red.csv
    data_raw/winequality-white.csv

Bucketing rule (based on the observed score distribution, 3-9):
    quality <= 5            -> "low"
    quality == 6             -> "medium"
    quality >= 7             -> "high"

Run:
    python model/build_dataset.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def bucket_quality(q: int) -> str:
    if q <= 5:
        return "low"
    elif q == 6:
        return "medium"
    else:
        return "high"


def main():
    red = pd.read_csv(RAW_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(RAW_DIR / "winequality-white.csv", sep=";")

    red["wine_type"] = 0   # red
    white["wine_type"] = 1  # white

    df = pd.concat([red, white], ignore_index=True)

    # Clean up column names (drop quotes/spaces -> snake_case)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    df["quality_class"] = df["quality"].apply(bucket_quality)
    df = df.drop(columns=["quality"])  # replaced by quality_class

    out_path = MODEL_DIR / "wine_quality_full.csv"
    df.to_csv(out_path, index=False)

    print(f"Combined dataset shape: {df.shape}")
    print("Class distribution:\n", df["quality_class"].value_counts())
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
