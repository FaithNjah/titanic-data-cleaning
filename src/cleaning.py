"""
Data-cleaning pipeline for the Titanic passenger dataset.

Usage:
    python src/cleaning.py                       # downloads the data if missing
    python src/cleaning.py --input data/titanic.csv --output data/titanic_clean.csv

Steps:
    1. Load the raw data (downloads a public copy if no local file exists)
    2. Extract passenger titles from names
    3. Handle missing values (Age by title median, Embarked by mode, drop Cabin)
    4. Remove duplicates
    5. Handle Fare outliers with a log transform
    6. Fix data types (Survived, Pclass, Sex, Embarked -> category)
    7. Add simple engineered features (FamilySize, IsAlone, HasCabin)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "the Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV, downloading it first if it does not exist locally."""
    path = Path(path)
    if not path.exists():
        print(f"{path} not found, downloading from {DATA_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the honorific (Mr, Mrs, Miss, Master, ...) out of the Name column."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace(TITLE_MAP)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Age: median of passengers with the same title.
    Embarked: mode (only two values missing).
    Cabin: ~77% missing, so keep only a HasCabin flag and drop the column."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df.drop(columns=["Cabin"])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and repeated passenger IDs."""
    df = df.drop_duplicates()
    return df.drop_duplicates(subset=["PassengerId"]).reset_index(drop=True)


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """High fares are real (first-class suites), not errors, so they are kept.
    A log transform reduces their skew for modelling."""
    df = df.copy()
    df["FareLog"] = np.log1p(df["Fare"])
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Store categorical columns as categories instead of numbers or strings."""
    df = df.copy()
    df["Survived"] = df["Survived"].astype("category")
    df["Pclass"] = pd.Categorical(df["Pclass"], categories=[1, 2, 3], ordered=True)
    for col in ["Sex", "Embarked", "Title"]:
        df[col] = df[col].astype("category")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """FamilySize = siblings/spouses + parents/children + the passenger."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_title(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = fix_types(df)
    df = add_features(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the Titanic dataset.")
    parser.add_argument("--input", default="data/titanic.csv")
    parser.add_argument("--output", default="data/titanic_clean.csv")
    args = parser.parse_args()

    raw = load_data(args.input)
    print(f"Raw data:   {raw.shape[0]} rows x {raw.shape[1]} columns, "
          f"{int(raw.isnull().sum().sum())} missing values")
    clean = run_pipeline(raw)
    print(f"Clean data: {clean.shape[0]} rows x {clean.shape[1]} columns, "
          f"{int(clean.isnull().sum().sum())} missing values")
    clean.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
