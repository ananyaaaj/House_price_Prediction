"""Launch with python -m streamlit run app.py."""
import json

import joblib
import pandas as pd
import streamlit as st

from src.clean import ROOT
from src.predict import MODEL, score

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")
st.caption("Classic machine learning • Synthetic home sales • USD")
reports = ROOT / "reports"

if not MODEL.exists():
    st.error("Run python -m src.train, then reload this page.")
    st.stop()


@st.cache_resource
def load_model(modified_at):
    return joblib.load(MODEL)


pipeline = load_model(MODEL.stat().st_mtime_ns)
metrics = json.loads((reports / "metrics.json").read_text())
overview, single, batch = st.tabs(["Model results", "Single property", "Score a CSV"])
with overview:
    st.subheader(metrics["selected_model"])
    cards = st.columns(4)
    cards[0].metric("Test MAE", f"${metrics['test_mae_usd']:,.0f}")
    cards[1].metric("Test RMSE", f"${metrics['test_rmse_usd']:,.0f}")
    cards[2].metric("Test R²", f"{metrics['test_r2']:.3f}")
    cards[3].metric("Median baseline MAE", f"${metrics['baseline_test_mae_usd']:,.0f}")
    st.dataframe(pd.read_csv(reports / "model_comparison.csv").round(3), hide_index=True)
    for chart in ["eda.png", "diagnostics.png"]:
        if (reports / chart).exists():
            st.image(str(reports / chart))
    st.subheader("Which inputs helped the model?")
    st.caption("Each bar shows the increase in test MAE after shuffling one input.")
    importance = pd.read_csv(reports / "feature_importance.csv")
    st.bar_chart(importance, x="feature", y="mae_increase_usd", horizontal=True)

with single:
    with st.form("property"):
        left, right = st.columns(2)
        with left:
            area = st.number_input("Living area (sq ft)", 1, 20000, 1700)
            lot = st.number_input("Lot area (sq ft)", 1, 100000, 8200)
            quality = st.slider("Overall quality", 1, 10, 6)
            built = st.number_input("Year built", 1800, 2025, 1995)
            garage = st.number_input("Garage spaces", 0, 10, 2)
        with right:
            beds = st.number_input("Bedrooms", 0, 20, 3)
            baths = st.number_input("Bathrooms", 0, 20, 2)
            renovated = st.selectbox("Renovated", ["No", "Yes"])
            neighborhood = st.selectbox("Neighborhood", ["Brookside", "Cedar Park", "Old Town", "Westfield"])
            style = st.selectbox("House style", ["Ranch", "Two Story", "Split Level"])
        if st.form_submit_button("Estimate sale price"):
            example = pd.DataFrame([{
                "living_area_sqft": area, "lot_area_sqft": lot, "overall_quality": quality,
                "year_built": built, "garage_spaces": garage, "bedrooms": beds, "bathrooms": baths,
                "renovated": renovated, "neighborhood": neighborhood, "house_style": style,
            }])
            estimate = score(example, pipeline).iloc[0]["predicted_price_usd"]
            st.metric("Estimated sale price", f"${estimate:,.0f}")

with batch:
    st.download_button(
        "Download sample input", (ROOT / "data/example_scoring.csv").read_bytes(),
        "example_scoring.csv", "text/csv"
    )
    upload = st.file_uploader("Upload property CSV", type="csv")
    if upload is not None:
        try:
            scored = score(pd.read_csv(upload), pipeline)
        except (ValueError, UnicodeError) as error:
            st.error(str(error))
        else:
            st.dataframe(scored.head(100), hide_index=True)
            st.download_button("Download predictions", scored.to_csv(index=False),
                               "predictions.csv", "text/csv")
