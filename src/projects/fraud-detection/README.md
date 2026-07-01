# Fraud Detection in Financial Transactions

## Overview

A machine learning pipeline to detect fraudulent credit card transactions. It addresses the critical business challenge of identifying rare fraudulent activies within highly imbalanced datasets, simulating a real-world scenario faced by financial institutions.

## The Problem

Financial fraud costs the global economy billions annually. The core challenge in fraud detection is class imbalance — legitimate transactions vastly outnumber fradulent ones. A naïve model might achieve 99% accuracy simply by predicting all transactions as legitimate, but it would fail entirely at its actual purpose.

This project aims to build a model that maximizes recall while maintaining acceptable precision.

## Architecture

Medallion Architecture for data processing:

1. Bronze Layer (Raw): Kaggle IEEE-CIS dataset (590K transactions, 400+ features).
2. Silver Layer (Processed): Data cleaned, missing values imputed, outliers detected and features selected.
3. Gold Layer (Modeled): Scaled and encoded data ready for machine learning algorithms.

## Data Modeling (ETL/ELT)

The ETL pipeline is build with software engineering practices:

- Configuration Management: Centralized parameters in `config.py`.
- Data Validation: IQR-based outlier detection and missing value thresholds.
- Feature Engineering: Handling categorical variables and scaling numerical features.
- Logging: Comprehensive execution logging for auditability.

## Machine Learning

Three algorithms were trained and compared:

1. Logistic Regression: Baseline model with class weighting.
2. Random Forest: Ensemble method robust to outliers.
3. Gradient Boosting: Advanced boosting algorithm for complex patterns.

## Visualizations

The project generates several analytical charts:

- Transaction amount distribution.
- Temporal fraud patterns
- Feature Importances
- Model Performance Comparison
- Confusion Matrices

## Structure

```text
fraud-detection/
├── data/
│   ├── raw/               # (gitignored)
│   ├── processed/         # (gitignored)
│   └── gold/              # integrated data
├── src/
│   ├── explore_data.py    # data exploration
│   ├── config.py          # centralized configuration
│   ├── logger.py          # logging setup
│   ├── etl_pipeline.py    # ETL orchestrator
│   ├── ml_models.py       # model training and evaluation
│   └── visualizations.py  # EDA and result plotting
├── output/
│   ├── logs/              # execution logs
│   ├── models/            # (gitignored)
│   ├── reports/           # quality and performance JSON reports
│   └── visualizations/    # PNG charts
└── README.md
```

## How to Run

1. Download the IEEE-CIS dataset from Kaggle.
2. Run the ETL pipeline: `src/etl_pipeline.py`.
3. Train the models: `src/ml_models.py`
4. Generate visualizations: `src/visualizations.py`