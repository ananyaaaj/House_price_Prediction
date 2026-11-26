"""Select the model with training-fold MAE and evaluate on untouched rows."""
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.clean import ROOT, load_training
from src.transform import build_pipeline

SEED = 42


def main():
    features, target, audit = load_training()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED
    )
    candidates = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Ridge regression": Ridge(alpha=10),
        "Random forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, n_jobs=2, random_state=SEED
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=220, max_depth=2, random_state=SEED
        ),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(
            build_pipeline(model), x_train, y_train, cv=folds,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                     "r2": "r2"}, error_score="raise"
        )
        rows.append({
            "model": name, "cv_mae_usd": -scores["test_mae"].mean(),
            "cv_mae_sd_usd": scores["test_mae"].std(),
            "cv_rmse_usd": -scores["test_rmse"].mean(), "cv_r2": scores["test_r2"].mean(),
        })
    comparison = pd.DataFrame(rows).sort_values("cv_mae_usd")
    winner = comparison.iloc[0]["model"]
    pipeline = build_pipeline(candidates[winner]).fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    baseline = DummyRegressor(strategy="median").fit(x_train, y_train).predict(x_test)
    mae = mean_absolute_error(y_test, prediction)
    baseline_mae = mean_absolute_error(y_test, baseline)
    metrics = {
        **audit, "training_rows": len(x_train), "test_rows": len(x_test),
        "selected_model": winner, "test_mae_usd": mae,
        "test_rmse_usd": root_mean_squared_error(y_test, prediction),
        "test_r2": r2_score(y_test, prediction), "baseline_test_mae_usd": baseline_mae,
        "mae_reduction_pct": 100 * (1 - mae / baseline_mae),
    }
    importance = permutation_importance(
        pipeline, x_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=8, random_state=SEED,
    )
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    pd.DataFrame({
        "actual_usd": y_test, "predicted_usd": prediction,
        "residual_usd": y_test - prediction,
    }).to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame({
        "feature": x_test.columns, "mae_increase_usd": importance.importances_mean,
        "sd_usd": importance.importances_std,
    }).sort_values("mae_increase_usd", ascending=False).to_csv(
        reports / "feature_importance.csv", index=False
    )
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump(pipeline, models / "house_price_pipeline.joblib", compress=3)
    print(comparison.round(3).to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
