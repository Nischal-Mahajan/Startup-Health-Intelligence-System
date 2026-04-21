# Startup Health Intelligence System

An end-to-end Machine Learning project that predicts startup success probability, failure risk, and provides a custom health score using funding-based features.

---

## Problem Statement

Investors and founders often struggle to evaluate startup potential in early stages.
This project uses historical funding and growth data to predict startup success and financial health.

---

## Key Features

* Feature Engineering:

  * Funding Efficiency
  * Funding per Year
  * Log Funding

* Model Training:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * XGBoost (Final Model)

* Custom Outputs:

  * Success Probability
  * Health Score (0–100)

* Deployment:

  * Streamlit-based interactive dashboard

---

## Key Insights

* Feature engineering significantly improved model performance.
* Dataset was highly imbalanced, so recall and ROC-AUC were prioritized over accuracy.
* XGBoost performed best in identifying successful startups.

---

## Model Evaluation

| Model               | Performance Insight                     |
| ------------------- | --------------------------------------- |
| Logistic Regression | Struggles with non-linear patterns      |
| Random Forest       | Handles complex relationships well      |
| Gradient Boosting   | Improves performance on difficult cases |
| XGBoost             | Best overall performance                |

---

## Project Structure

```
Startup-Health-Intelligence-System/
│
├── dashboard/        # Streamlit app
├── models/           # Trained pipeline
├── notebooks/        # EDA and model training
├── README.md
├── requirements.txt
```

---

## Run Locally

```
cd dashboard
streamlit run app.py
```

---

## Live Demo

(Add Streamlit deployment link here)

---

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Streamlit

---

## Author

Nischal Mahajan

---

## Summary

This project demonstrates a complete machine learning pipeline including data preprocessing, feature engineering, model selection, and deployment in a real-world scenario.
