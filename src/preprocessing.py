"""Data loading, validation, target normalization, and leakage-safe preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
TARGET_CANDIDATES = ("target", "num", "goal")
# UCI documents these numeric codes as categories rather than measurements.
KNOWN_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def load_dataset(path: str | Path = DATA_PATH, *, verbose: bool = True) -> pd.DataFrame:
    """Load a CSV and print a factual inspection summary before modeling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Add UCI data as data/heart.csv.")
    frame = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])
    if "thalch" in frame.columns and "thalach" not in frame.columns:
        frame = frame.rename(columns={"thalch": "thalach"})
    if verbose:
        print(f"Dataset loaded: {path}\nShape: {frame.shape}\n\nFirst five rows:\n{frame.head()}")
        print(f"\nColumns: {list(frame.columns)}\n\nData types:\n{frame.dtypes}")
        print(f"\nMissing values:\n{frame.isna().sum()}\n\nDuplicate rows: {frame.duplicated().sum()}")
        print(f"\nUnique values:\n{frame.nunique(dropna=False)}")
        for column in frame.columns:
            values = frame[column].dropna().unique().tolist()
            preview = values[:20]
            suffix = " ..." if len(values) > 20 else ""
            print(f"  {column}: {preview}{suffix}")
    return frame


def normalize_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return a copy with a binary target; UCI num severity values > 0 mean disease."""
    target = next((name for name in TARGET_CANDIDATES if name in frame.columns), None)
    if target is None:
        raise ValueError(f"Expected one target column from {TARGET_CANDIDATES}; found {list(frame.columns)}")
    cleaned = frame.copy()
    # The newer UCI export uses thalch; use the familiar Cleveland feature name.
    if "thalch" in cleaned.columns and "thalach" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"thalch": "thalach"})
    raw = pd.to_numeric(cleaned[target], errors="coerce")
    if raw.isna().any():
        cleaned[target] = raw
    values = set(raw.dropna().unique())
    if values <= {0, 1}:
        cleaned[target] = raw
    elif values and all(float(value).is_integer() and value >= 0 for value in values):
        cleaned[target] = (raw > 0).astype("Int64")
    else:
        raise ValueError(f"Target {target!r} is not a recognized binary/severity target: {sorted(values, key=str)}")
    cleaned = cleaned.dropna(subset=[target])
    cleaned[target] = cleaned[target].astype(int)
    if cleaned[target].nunique() != 2:
        raise ValueError("Binary target must contain both classes 0 and 1.")
    return cleaned, target


def validate_and_split(frame: pd.DataFrame, target: str):
    """Remove exact duplicate rows, exclude invalid feature rows and make stratified split."""
    from sklearn.model_selection import train_test_split

    cleaned = frame.drop_duplicates().copy()
    # Exclude row identifiers and the source-center label: neither is a patient measure.
    ignored = [column for column in ("id", "dataset") if column in cleaned.columns]
    X = cleaned.drop(columns=[target, *ignored]).replace([np.inf, -np.inf], np.nan)
    y = cleaned[target]
    # UCI uses '?' for unknown categorical/number fields. Imputation is performed in-fold.
    # Reject columns that are entirely empty and obvious sentinel values in bounded fields.
    empty_columns = [name for name in X if X[name].isna().all()]
    if empty_columns:
        X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after validation.")
    for name in X:
        # Convert object columns only when every nonmissing value is genuinely numeric.
        if X[name].dtype == "object":
            parsed = pd.to_numeric(X[name], errors="coerce")
            if parsed.notna().sum() == X[name].notna().sum():
                X[name] = parsed
    # Flag domain violations in conventional UCI columns as missing so the
    # training-only imputer can handle them without learning from the test set.
    valid_ranges = {
        "age": (1, 120), "trestbps": (50, 300), "chol": (50, 800),
        "thalach": (40, 300), "oldpeak": (-5, 20),
        "sex": (0, 1), "fbs": (0, 1), "exang": (0, 1),
        "cp": (1, 4), "restecg": (0, 2), "slope": (1, 3),
        "ca": (0, 3), "thal": (3, 7),
    }
    for name, (minimum, maximum) in valid_ranges.items():
        if name in X and pd.api.types.is_numeric_dtype(X[name]):
            invalid = X[name].notna() & ~X[name].between(minimum, maximum)
            if invalid.any():
                print(f"Warning: {int(invalid.sum())} out-of-range value(s) in {name} will be imputed.")
                X.loc[invalid, name] = np.nan
    X = X.dropna(axis=1, how="all")
    if not X.columns.size:
        raise ValueError("No numeric, non-empty features were found.")
    print(f"\nClean shape: {X.shape}; target distribution:\n{y.value_counts().sort_index()}")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Infer numeric/categorical columns from training data and fit only in a Pipeline."""
    categorical = [column for column in X.columns if column in KNOWN_CATEGORICAL]
    numeric = [column for column in X.columns if column not in categorical and
               pd.api.types.is_numeric_dtype(X[column])]
    categorical.extend(column for column in X.columns if column not in categorical and column not in numeric)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")

