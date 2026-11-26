"""Feature engineering and preprocessing shared by training and inference."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.clean import CATEGORICAL, NUMERIC_BOUNDS

NUMERIC = [*NUMERIC_BOUNDS, "home_age", "lot_to_living_ratio"]


def engineer_features(frame):
    result = frame.copy()
    result["home_age"] = 2025 - result.pop("year_built")
    result["lot_to_living_ratio"] = result["lot_area_sqft"] / result["living_area_sqft"]
    return result


NUMERIC.remove("year_built")


def build_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features)),
        ("preprocess", ColumnTransformer([
            ("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL),
        ])),
        ("model", model),
    ])
