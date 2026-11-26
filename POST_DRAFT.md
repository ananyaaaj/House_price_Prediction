**Can linear regression beat a random forest at predicting house prices?**

It did in this project. I compared four classic ML models and a median-price baseline on a synthetic housing dataset, then built a Streamlit app to explore results and estimate prices.

**House Price Prediction with Classic Machine Learning**

**1. Data cleaning and exploration**

- Started with 2,520 raw records and kept 2,499 after removing duplicates and an invalid sale price.
- Standardized category labels, handled missing values, and explored how living area and neighborhood relate to price.
- Created two features from the raw data, home age and the lot-to-living-area ratio.

**2. Model comparison and evaluation**

- Compared Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting against a median-price baseline.
- Used five-fold cross-validation and fitted preprocessing inside each fold.
- Linear Regression had the lowest validation MAE. On 500 held-out rows, its MAE was **$15,176**, down from **$57,302** for the median baseline. It achieved a **$19,007 RMSE** and **0.935 R²**.

**3. Model interpretation and application**

- Used permutation importance to check which inputs the model relied on. Living area and neighborhood ranked highest in this simulation.
- Built a Streamlit app with model diagnostics, a property estimate form, and CSV upload with downloadable predictions.

My biggest takeaway was how much a baseline helps put an error metric in context. Comparing dollar errors made it easy to explain what the model improved and how far off a typical estimate could be. The benchmark also showed why it makes sense to test simple models early, especially when the data follows a mostly additive pattern.

**Stack** Python, pandas, NumPy, scikit-learn, Matplotlib, Streamlit

**GitHub** [Add your repository link after publishing]
