# Boston Housing Price Prediction Analysis

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) workflow predicting 1970s Boston home values using Linear Regression, target logging, and residual diagnostics.

---

##  Project Overview

This project analyzes spatial, economic, and environmental features of Boston neighborhoods to model house prices (`PRICE`). It highlights the impact of feature engineering—specifically log-transforming the target variable—to handle skewed distributions and improve model fit.

---

##  Key Pipeline Steps

- **Data Quality Check:** Verification of null values, duplicates, and feature data types.
- **Exploratory Data Analysis (EDA):** Visualizing distributions (`PRICE`, `RM`, `DIS`, `RAD`) and relationships (`NOX` vs. `INDUS`, `LSTAT` vs. `PRICE`) using Seaborn and Plotly.
- **Baseline Model:** Fits a Linear Regression model directly on raw housing prices.
- **Diagnostics:** Analysis of residual distribution and heteroscedasticity.
- **Log-Transformed Model:** Applies $\ln(\text{PRICE})$ to normalize target skewness, refitting the regression to optimize overall $R^2$ performance.
- **Custom Inference:** Demonstrates inverse-transform predictions ($\exp(\hat{y}) \times 1000$) for custom house profiles.

---

##  Technologies Used

- **Python 3.**
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `seaborn`, `matplotlib`, `plotly`
- **Machine Learning:** `scikit-learn` (`LinearRegression`, `train_test_split`)

---

##  Quickstart
   pip install pandas numpy seaborn matplotlib plotly scikit-learn
## Author 
  Anurag Dethe
   
