"""Validate inputs and normalize values before model fitting or scoring."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sale_price_usd"
NUMERIC_BOUNDS = {
    "living_area_sqft": (1, None), "lot_area_sqft": (1, None),
    "overall_quality": (1, 10), "year_built": (1800, 2025),
    "bedrooms": (0, None), "bathrooms": (0, None), "garage_spaces": (0, None),
}
CATEGORICAL = ["renovated", "neighborhood", "house_style"]
FEATURES = [*NUMERIC_BOUNDS, *CATEGORICAL]


def clean_features(frame):
    missing = set(FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    result = frame[FEATURES].copy()
    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        values = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.ge(lower)
        if upper is not None:
            valid &= values.le(upper)
        result[column] = values.where(valid)
    for column in CATEGORICAL:
        values = result[column].astype("string").str.strip().str.title()
        result[column] = values.replace("", pd.NA).to_numpy(dtype=object, na_value=np.nan)
    result["renovated"] = result["renovated"].where(result["renovated"].isin(["Yes", "No"]))
    return result


def load_training(path=ROOT / "data/housing_raw.csv"):
    raw = pd.read_csv(path)
    unique = raw.drop_duplicates()
    if unique["property_id"].duplicated().any():
        raise ValueError("Conflicting records share a property ID.")
    target = pd.to_numeric(unique[TARGET], errors="coerce")
    valid = np.isfinite(target) & target.gt(0)
    features = clean_features(unique.loc[valid])
    return features, target.loc[valid], {
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(unique),
        "invalid_targets_removed": int((~valid).sum()), "clean_rows": int(valid.sum()),
    }
