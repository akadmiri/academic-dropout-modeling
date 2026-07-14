# Student Churn Prediction

## Objective
A temporal binary classification pipeline predicting student dropout (décrochage scolaire). The model is optimized to minimize false negatives (undetected dropouts) in order to prioritize high-risk student interventions.

## Architecture & Methodology
- **Data Ingestion:** Extracted from the UCI Machine Learning Repository via API.
- **Feature Engineering:** Calculated first-order derivatives (velocity) of academic performance (`Efficiency Delta`, `Grade Delta`) to capture temporal degradation prior to dropout.
- **Dimensionality Reduction:** Dynamically pruned macroeconomic and socio-demographic noise (Information Gain < 0.015) using a baseline XGBoost evaluator to prevent data leakage and overfitting.
- **Model Evaluation:** Stratified 5-Fold Cross-Validation targeting PR-AUC and Recall. Applied a custom probability threshold to isolate the minority class effectively.

## Performance Metrics (Hold-out Test Set)
- **PR-AUC:** 0.9543
- **Recall (Dropout Class):** 0.86
- **Precision (Dropout Class):** 0.92
- **Global Accuracy:** 0.91

## Project Structure
```text
├── data/
│   ├── raw/               # Ignored in Git
│   └── processed/         # Ignored in Git
├── src/
│   ├── data.py            # UCI API extraction
│   ├── preprocessing.py   # Binarization and delta engineering
│   ├── feature_importance.py
│   ├── feature_selection.py
│   ├── model_training.py  # Stratified K-Fold validation
│   └── test.py            # Final threshold calibration and inference
├── .gitignore
├── requirements.txt
└── README.md