from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def cross_validation(train_path: str, n_splits: int = 5) -> Dict[str, float]:
    """Stratified K-Fold Cross Validation on the training dataset."""

    df = pd.read_csv(train_path)
    X = df.drop(columns=["churn_target"]).values
    y = df["churn_target"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=10,
    )

    metrics = {"pr_auc": [], "roc_auc": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        metrics["pr_auc"].append(average_precision_score(y_val, y_prob))
        metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        metrics["precision"].append(precision_score(y_val, y_pred))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["f1"].append(f1_score(y_val, y_pred))

    # Calculate average metrics across folds
    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return mean_metrics


if __name__ == "__main__":
    train_path = "data/processed/train_data_final.csv"

    results = cross_validation(train_path)
    print("Cross-Validation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
