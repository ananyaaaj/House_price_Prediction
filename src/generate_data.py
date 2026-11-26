"""Generate a reproducible fictional housing market for the portfolio demo."""
import numpy as np
import pandas as pd

from src.clean import ROOT


def main():
    rng = np.random.default_rng(42)
    n = 2500
    neighborhood = rng.choice(["Brookside", "Cedar Park", "Old Town", "Westfield"], n)
    premium = pd.Series(neighborhood).map({
        "Brookside": 45000, "Cedar Park": 72000, "Old Town": 0, "Westfield": 22000,
    }).to_numpy()
    area = np.clip(rng.lognormal(np.log(1650), 0.34, n), 620, 4200).round()
    lot = np.clip(area * rng.uniform(2.2, 7.8, n), 1700, 25000).round()
    quality = np.clip(rng.normal(6.2, 1.5, n).round(), 2, 10)
    built = rng.integers(1940, 2026, n)
    renovated = rng.random(n) < 0.26
    bedrooms = np.clip((area / 580 + rng.normal(0.5, 0.55, n)).round(), 1, 6)
    bathrooms = np.clip((area / 900 + rng.normal(0, 0.55, n)).round(), 1, 5)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.09, 0.26, 0.57, 0.08])
    style = rng.choice(["Ranch", "Two Story", "Split Level"], n)
    price = (
        27000 + 84 * area + 1.5 * lot + 7800 * quality
        + premium + 650 * (built - 1940) + 16000 * renovated
        + 5800 * bathrooms + 6500 * garage + 9000 * (style == "Two Story")
        + 12500 * (quality >= 8) * (area >= 1850)
        + rng.normal(0, 14000 + 2.5 * area, n)
    )
    frame = pd.DataFrame({
        "property_id": np.arange(1, n + 1), "living_area_sqft": area,
        "lot_area_sqft": lot, "overall_quality": quality, "year_built": built,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "garage_spaces": garage,
        "renovated": np.where(renovated, "Yes", "No"),
        "neighborhood": neighborhood, "house_style": style,
        "sale_price_usd": price.round().astype(int),
    })
    for column, count in {"lot_area_sqft": 110, "garage_spaces": 70, "house_style": 45}.items():
        frame.loc[rng.choice(n, count, replace=False), column] = np.nan
    frame.loc[0, "neighborhood"] = " brookside "
    frame.loc[1, "living_area_sqft"] = -50
    frame.loc[2, "sale_price_usd"] = 0
    raw = pd.concat([frame, frame.iloc[10:30]], ignore_index=True)
    raw.to_csv(ROOT / "data/housing_raw.csv", index=False)
    frame.iloc[35:41].drop(columns="sale_price_usd").to_csv(
        ROOT / "data/example_scoring.csv", index=False
    )


if __name__ == "__main__":
    main()
