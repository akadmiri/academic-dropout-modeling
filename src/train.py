import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib


TARGET_COL = "churn_target"


def load_processed(
    train_path="data/processed/train_data.csv", test_path="data/processed/test_data.csv"
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


# Logistic Regression:
def train_log(X_train, y_train):
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


# XGBoost tree:
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


# SVM:
def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(kernel="linear", class_weight="balanced", random_state=10)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_svm(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_score = model.decision_function(X_test_scaled)

    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_score):.3f}")


# Cross Validation: 
def compare_models(X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)

    # Logistic Regression    
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=10)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logreg_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=cv, scoring="average_precision")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # XGB tree
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=10,
    )
    xgb_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="average_precision")

    # SVM 
    svm = SVC(kernel="linear", class_weight="balanced", random_state=10)
    svm_scores = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring="average_precision")


    print(f"Logistic Regression PR-AUC: {logreg_scores.mean():.3f} ± {logreg_scores.std():.3f}")
    print(f"XGBoost PR-AUC:              {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")
    print(f"SVM PR-AUC: {svm_scores.mean():.3f} ± {svm_scores.std():.3f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()

    print("Cross-validated comparison: ")
    compare_models(X_train, y_train)

    print("Logistic Regression: ")
    model, scaler = train_log(X_train, y_train)
    evaluate(model, scaler, X_test, y_test)

    print("\nXGBoost: ")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_xgb(xgb_model, X_test, y_test)

    print("\nSVM: ")
    svm_model, svm_scaler = train_svm(X_train, y_train)
    evaluate_svm(svm_model, svm_scaler, X_test, y_test)

'''
    joblib.dump(model, "output/model.joblib" )
    joblib.dump(scaler, "output/scaler.joblib")
    joblib.dump(xgb_model, "output/xgbmodel.joblib")
'''
