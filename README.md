# 🏦 ATM Fraud Detection System

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to detect
fraudulent ATM transactions.\
The dataset is highly imbalanced (\~0.18% fraud rate), making
recall-focused evaluation critical.

The objective is to detect fraudulent transactions while maintaining
manageable false positives.

------------------------------------------------------------------------

## 🎯 Business Goals

-   **Recall ≥ 85%** (detect most fraud cases)
-   **Precision ≥ 35%** (avoid excessive false alarms)
-   Minimize cost of false negatives (missed fraud)

------------------------------------------------------------------------

## 🛠️ Technical Approach

### 1️⃣ Data Preprocessing

-   StandardScaler pipeline
-   Stratified train-test split
-   Cross-validation for robust evaluation

### 2️⃣ Models Implemented

-   Logistic Regression
-   XGBoost
-   Random Forest
-   LightGBM
-   **Stacking Ensemble (Final Selected Model)**

### 3️⃣ Threshold Optimization

Instead of default 0.5 threshold: - Optimized using cross-validation -
Applied business constraints (Recall & Precision targets) - Selected
best threshold per model

------------------------------------------------------------------------

## 📊 Final Model Comparison (Test Set - Fraud Class)

| Model               | Precision | Recall | F2 Score | AUPRC |
|---------------------|-----------|--------|----------|-------|
| Stacking            | 0.77      | 0.88   | 0.85     | 0.85  |
| Random Forest       | 0.66      | 0.88   | 0.82     | 0.83  |
| XGBoost             | 0.45      | 0.93   | 0.76     | 0.83  |
| LightGBM            | 0.35      | 0.93   | 0.69     | 0.81  |
| Logistic Regression | 0.14      | 0.94   | 0.44     | 0.75  |

------------------------------------------------------------------------

## 🏆 Final Model: Stacking Ensemble

### Performance:

-   **Precision:** 0.77
-   **Recall:** 0.88
-   **F2 Score:** 0.85
-   **AUPRC:** 0.85

### Business Interpretation:

-   Detects \~88% of fraud cases
-   High precision for extremely imbalanced data
-   Strong balance between fraud detection and investigation workload

------------------------------------------------------------------------

## 📈 Evaluation Metrics Used

-   Precision
-   Recall
-   F2 Score (Recall-focused)
-   AUPRC (Area Under Precision-Recall Curve)
-   Cost Weighted Accuracy
-   Average Response Time

------------------------------------------------------------------------

## 🚀 Production Readiness

-   Model and threshold saved using `joblib`
-   Consistent preprocessing pipeline
-   Same threshold used for training, validation, and testing
-   Stable cross-validation performance

------------------------------------------------------------------------

## 📦 Project Structure

-   Data preprocessing pipeline
-   Model training & evaluation
-   Threshold optimization
-   Model comparison visualization
-   Final stacking ensemble selection
-   Deployment-ready artifacts

------------------------------------------------------------------------

## 🧠 Key Learnings

-   Threshold tuning is critical in imbalanced fraud detection.
-   Precision-Recall trade-off must align with business cost structure.
-   Stacking improves stability and balance compared to individual
    models.
-   Evaluation must avoid data leakage when optimizing thresholds.

------------------------------------------------------------------------
------------------------------------------------------------------------

## 📂 Dataset

Due to GitHub file size limitations, the dataset is not included in this repository.

You can download the full dataset from Google Drive:

👉 [ATM Fraud Dataset (Google Drive)](https://drive.google.com/drive/folders/17S7VNe7x4U9_7CFBRPApy4hnnpPyUJy-?usp=sharing)

After downloading:

1. Create a folder named `data`
2. Place all dataset files inside the `data/` directory

## 👨‍💻 Author

ATM Fraud Detection Project -- End-to-End ML System
