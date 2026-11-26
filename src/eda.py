"""Export exploratory and held-out diagnostic charts for the README and app."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.clean import ROOT, load_training


def main():
    x, y, _ = load_training()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    axes[0].hist(y / 1000, bins=30, color="#2563a7")
    axes[0].set(title="Sale price distribution", xlabel="Sale price (USD thousands)", ylabel="Homes")
    axes[1].scatter(x["living_area_sqft"], y / 1000, s=8, alpha=0.25, color="#2563a7")
    axes[1].set(title="Living area and sale price", xlabel="Living area (sq ft)", ylabel="USD thousands")
    medians = y.groupby(x["neighborhood"]).median().sort_values() / 1000
    medians.plot.barh(ax=axes[2], color="#117c73")
    axes[2].set(title="Median price by neighborhood", xlabel="USD thousands", ylabel="")
    fig.savefig(reports / "eda.png", dpi=160)
    plt.close(fig)

    predictions = pd.read_csv(reports / "test_predictions.csv") / 1000
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    actual, predicted = predictions["actual_usd"], predictions["predicted_usd"]
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axes[0].scatter(actual, predicted, s=15, alpha=0.48, color="#2563a7")
    axes[0].plot([low, high], [low, high], "--", color="#64748b")
    axes[0].set(title="Actual vs. predicted", xlabel="Actual (USD thousands)", ylabel="Predicted (USD thousands)")
    axes[1].scatter(predicted, predictions["residual_usd"], s=15, alpha=0.48, color="#117c73")
    axes[1].axhline(0, linestyle="--", color="#64748b")
    axes[1].set(title="Residuals", xlabel="Predicted (USD thousands)", ylabel="Actual minus predicted (USD thousands)")
    fig.savefig(reports / "diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
