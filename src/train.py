import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TARGET_COL = "churn_target"


def load_processed(
    train_path="data/processed/train_data.csv", test_path="data/processed/test_data.csv"
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=10)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_proba):.3f}")


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=10,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_xgb(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_proba):.3f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()

    print("Logistic Regression (baseline)")
    model, scaler = train_baseline(X_train, y_train)
    evaluate(model, scaler, X_test, y_test)

    print("\nXGBoost")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_xgb(xgb_model, X_test, y_test)
