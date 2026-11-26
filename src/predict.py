"""Score a CSV and preserve its original columns and row order."""
import argparse

import joblib
import pandas as pd

from src.clean import ROOT, clean_features

MODEL = ROOT / "models/house_price_pipeline.joblib"


def score(frame, pipeline):
    if frame.empty:
        raise ValueError("Provide at least one property.")
    clean = clean_features(frame)
    if clean.isna().all(axis=1).any():
        raise ValueError("At least one property has no valid feature values.")
    return frame.assign(predicted_price_usd=pipeline.predict(clean).round(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    score(pd.read_csv(args.input_csv), joblib.load(MODEL)).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
