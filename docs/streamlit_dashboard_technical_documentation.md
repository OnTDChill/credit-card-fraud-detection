# Streamlit Dashboard Technical Documentation

## Overview

This document provides the technical specification for the Fraud Detection Dashboard. The system is designed as a Decision Support System (DSS) that integrates machine learning predictions with business KPIs to enable data-driven risk management.

## System Architecture

### 1. Data Layer
The system utilizes a decoupled data architecture to ensure performance and stability.

#### Storage
- **Primary Database**: SQLite (`REVIEW_DB_PATH`) stores aggregated business metrics, system configurations, and audit trails.
- **Model Artifacts**: The `FRAUD_ARTIFACTS_DIR` contains:
    - Serialized models (`.joblib`)
    - Training reports (`training_report.json`)
    - Prediction sets (`predictions_train.csv`, `predictions_validation.csv`, `predictions_test.csv`)

#### Schema Highlights
- `dss_transaction_summary`: Core financial metrics (GMV, fraud rate, estimated revenue) aggregated by period.
- `dss_credit_portfolio`: Credit risk indicators including NPL (Non-Performing Loan) rates.
- `dss_merchant_accounts`: Merchant-level risk scores and anomaly flags.
- `dss_customer_service`: Service quality metrics (CSAT, Churn Rate).
- `system_config`: Dynamic parameters such as `low_threshold` and `high_threshold`.
- `audit_log`: Traceability log for all system configuration changes.

### 2. Logic & Analytics Layer
The analytics engine is implemented in Python using `pandas`, `numpy`, and `scikit-learn`.

#### ML Evaluation Engine
The system calculates model performance dynamically from the test prediction set:
- **Input**: `y_true` (Actual label) and `fraud_probability` (Model score).
- **Process**: Applies a configurable `threshold` to convert probabilities into binary predictions.
- **Metrics**:
    - **Accuracy**: $(TP + TN) / Total$
    - **Precision**: $TP / (TP + FP)$
    - **Recall**: $TP / (TP + FN)$
    - **F1-Score**: Harmonic mean of Precision and Recall.
    - **ROC-AUC**: Area under the ROC curve, measuring class separation capability.

#### Model Interpretability
Feature importance is extracted directly from the loaded estimator:
- **Tree-based models**: Utilizes `feature_importances_` (Gini importance).
- **Linear models**: Utilizes the absolute value of `coef_`.
- **Mapping**: Feature indices are mapped back to original names using the `preprocessor` metadata.

### 3. Presentation Layer
Built with **Streamlit**, the UI is divided into two primary personas:

#### CEO/Business Persona
- **Enterprise Health Score**: A composite metric (0-100) derived from Safety, Revenue, and Service targets.
- **Smart Alerts**: Triggered when MoM growth drops or risk metrics (NPL, Fraud Rate) exceed predefined safety bounds.
- **Geographic Drill-down**: Visualizes GMV and Risk distribution across regions and area types (Urban, Rural, Border, Island).

#### Technical/Analyst Persona
- **Model Diagnostics**: Interactive ROC-AUC and Precision-Recall curves.
- **Confusion Matrix**: Heatmap visualization of TP, TN, FP, and FN.
- **Error Analysis**: Dedicated views for False Positives and False Negatives to facilitate model iterative improvement.
- **Benchmarking**: Comparison table of different algorithm performances (e.g., RF vs. XGBoost).

## Implementation Details

### Performance Optimization
- **Caching**: Uses `@st.cache_data` for SQL queries and `@st.cache_resource` for loading heavy ML models via `joblib`.
- **Lazy Loading**: Technical artifacts are only loaded when the Technical Analysis page is accessed.

### Security & Audit
- **Audit Trail**: Every critical system change is captured in the `audit_log` table, recording the timestamp, user, old value, new value, and justification.

## Model Lifecycle

1. **Training**: Pipeline processes raw data $\rightarrow$ Feature Engineering $\rightarrow$ Training $\rightarrow$ Validation.
2. **Export**: The best model and a `training_report.json` (containing the optimal threshold) are saved to the artifacts directory.
3. **Deployment**: The Dashboard loads the latest artifact and applies the reported threshold for baseline evaluation.
4. **Monitoring**: Analysts use the Error Analysis tool to identify gaps, triggering a retraining cycle if performance degrades.

## Conclusion

The architecture balances the need for high-level business visibility with deep technical diagnostics. By separating the data ingestion (ETL), the prediction logic (ML), and the visualization (Streamlit), the system remains maintainable and scalable for evolving fraud patterns.