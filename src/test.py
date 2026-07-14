import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    train_path: str, test_path: str, decision_threshold: float = 0.4
) -> None:
    """Trains the final model on all available training data and evaluates it on the test set, printing classification metrics and confusion matrix."""

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["churn_target"]).values
    y_train = train_df["churn_target"].values

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["churn_target"]).values
    y_test = test_df["churn_target"].values

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=10,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)
    pr_auc = average_precision_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Average Precision Score:", pr_auc)
    print("Confusion Matrix:")
    print(f"True Negatives (Safe correctly predicted):     {conf_matrix[0][0]}")
    print(f"False Positives (Safe incorrectly predicted):  {conf_matrix[0][1]}")
    print(f"False Negatives (Dropout incorrectly predicted): {conf_matrix[1][0]}")
    print(f"True Positives (Dropout correctly predicted):   {conf_matrix[1][1]}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))


if __name__ == "__main__":
    train_path = "data/processed/train_data_final.csv"
    test_path = "data/processed/test_data_final.csv"

    evaluate_model(train_path, test_path)
