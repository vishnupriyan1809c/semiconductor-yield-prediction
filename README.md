# Semiconductor Manufacturing Yield Prediction

End-to-end machine learning pipeline to predict semiconductor manufacturing yield using the SECOM dataset. Built to minimize faulty chips reaching customers by prioritizing Fail detection over raw accuracy.

---

## Problem Statement

Semiconductor manufacturing generates hundreds of sensor readings per production unit. Not all signals are useful — many contain noise or missing values. The goal is to classify each unit as **Pass** or **Fail** based on sensor data, with a critical business constraint: **missing a faulty chip is far more costly than a false alarm.**

---

## Dataset

| Property | Detail |
|---|---|
| Source | SECOM Manufacturing Dataset (UCI) |
| Entities | 1,567 production units |
| Features | 591 sensor signal readings |
| Target | Pass (-1) / Fail (1) |
| Class Imbalance | 93.4% Pass / 6.6% Fail |

---

## ML Pipeline

```
Raw Data → Drop Useless Columns → EDA → Train-Test Split →
Median Imputation → StandardScaler → SMOTE → PCA →
Model Training → Evaluation → Save Best Model
```

**Key preprocessing decisions:**
- Train-Test Split performed **before** imputation and scaling to prevent data leakage
- Median Imputation applied to handle extensive missing sensor values
- SMOTE applied only on training data to balance the 93/7 class ratio
- PCA applied for dimensionality reduction of noisy high-dimensional sensor signals

---

## Model Comparison

| Model | Train Accuracy | Test Accuracy | Fail Recall | AUC Score |
|---|---|---|---|---|
| Logistic Regression | 79.4% | 70.1% | 0.29 | 0.550 |
| Random Forest | 99.9% | 74.2% | 0.33 | 0.616 |
| **XGBoost** | **41.9%** | **36.9%** | **0.76** | **0.550** |

**Selected Model: XGBoost**

XGBoost was selected despite lower test accuracy because it achieves the highest **Fail Recall of 0.76** — catching 76% of faulty chips. Only 5 faulty chips out of 21 were missed, the lowest across all models.

> Random Forest has better AUC (0.616) but catches only 33% of failures. In semiconductor manufacturing, Fail Recall is the priority metric — not accuracy.

---

## Business Recommendation

Deploy XGBoost as the production model. Prioritizing Fail Recall over accuracy is the correct business decision — a faulty chip reaching a customer causes warranty claims, reputation damage, and recall costs far exceeding the cost of re-inspecting a falsely flagged good chip.

---

## Confusion Matrix Results (XGBoost)

| | Predicted Pass | Predicted Fail |
|---|---|---|
| **Actual Pass** | 100 (True Pass) | 193 (False Alarm) |
| **Actual Fail** | 5 (Missed Failure) | 16 (Caught Failure) |

---

## Tech Stack

- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Imbalanced-learn, Joblib
- **Environment:** Jupyter Notebook, VS Code

---

## Project Structure

```
semiconductor-yield-prediction/
│
├── semiconductor_yield.ipynb          # Full ML pipeline notebook
├── signal-data.csv                    # SECOM dataset
├── semiconductor_yield_xgboost_model.pkl  # Saved best model
└── README.md
```

---

## Key Learnings

- Accuracy is a misleading metric for imbalanced datasets — a naive model predicting all Pass achieves 93.4% accuracy while catching zero failures
- Data leakage prevention: imputation and scaling must happen after the train-test split
- SMOTE must be applied only on training data, never on test data
- Business context must drive metric selection — Fail Recall > Accuracy for this use case

---

## Future Work

- Try LightGBM or CatBoost for potentially better Fail Recall
- Explore feature selection as an alternative to PCA
- Collect more labeled Fail samples to improve model training
- Deploy model as a REST API for real-time yield prediction
