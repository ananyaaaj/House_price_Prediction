# Housing data dictionary

The CSV is fully synthetic and comes from `python -m src.generate_data`. The simulated sample uses 2025 as its reference year. The generator assigns prices using a mostly additive equation with neighborhood premiums, a quality–size interaction, and random noise. It deliberately introduces missing fields, duplicate rows, and invalid values for cleaning practice.

| Column | Meaning |
| --- | --- |
| property_id | Synthetic property identifier, excluded from training features |
| living_area_sqft | Interior living area in square feet |
| lot_area_sqft | Land area in square feet |
| overall_quality | Simulated quality rating on a 1–10 scale |
| year_built | Construction year |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| garage_spaces | Number of garage parking spaces |
| renovated | Yes or No |
| neighborhood | Fictional neighborhood name |
| house_style | Ranch, Two Story, or Split Level |
| sale_price_usd | Simulated sale price in USD. Training target only |

`housing_raw.csv` is the input for training. `example_scoring.csv` shows the scoring format. Scoring requires all ten feature columns. Individual feature values can be blank and missing values are handled by the trained pipeline if the row has at least one valid input. Additional columns are preserved in the result.
