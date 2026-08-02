import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
from pathlib import Path


TARGET_COL = "churn_target"


def load_processed(
    train_path="data/processed/train_data.csv", test_path="data/processed/test_data.csv", mask_path="data/processed/train_simulated_mask.csv"
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    # Load simulation mask
    mask_file = Path(mask_path)
    if mask_file.exists():
        sim_mask = pd.read_csv(mask_file).iloc[:, 0].values
    else:
        # If no mask file exists, assume all rows are real data
        sim_mask = np.zeros(len(train_df), dtype=int)
        
    return X_train, X_test, y_train, y_test, sim_mask


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
def train_svm(X_train_scaled, y_train):
    model = SVC(kernel="linear", class_weight="balanced", random_state=10)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_svm(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_score = model.decision_function(X_test_scaled)

    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_score):.3f}")

'''
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
'''

def compare_models_real_cv(X_train, y_train, sim_mask, n_splits=5):
    """
    Cross-validation scheme that validates models strictly on REAL data folds.
    Augments the training fold with synthetic data when available.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    
    is_syn = (sim_mask == 1)
    
    # Separate Real and Synthetic data
    X_real, y_real = X_train[~is_syn].reset_index(drop=True), y_train[~is_syn].reset_index(drop=True)
    X_syn, y_syn = X_train[is_syn].reset_index(drop=True), y_train[is_syn].reset_index(drop=True)
    
    logreg_scores = []
    xgb_scores = []
    svm_scores = []
    
    # Stratified split performed STRICTLY on real data
    for train_idx, val_idx in skf.split(X_real, y_real):
        X_real_tr, y_real_tr = X_real.iloc[train_idx], y_real.iloc[train_idx]
        X_real_val, y_real_val = X_real.iloc[val_idx], y_real.iloc[val_idx]
        
        # Combine Real Train fold with Synthetic rows for model training
        if len(X_syn) > 0:
            X_tr_fold = pd.concat([X_real_tr, X_syn], ignore_index=True)
            y_tr_fold = pd.concat([y_real_tr, y_syn], ignore_index=True)
        else:
            X_tr_fold, y_tr_fold = X_real_tr, y_real_tr
            
        # 1. Logistic Regression (Fit Scaler inside the fold to prevent leakage)
        scaler_log = StandardScaler()
        X_tr_scaled = scaler_log.fit_transform(X_tr_fold)
        X_val_scaled = scaler_log.transform(X_real_val)
        
        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=10)
        logreg.fit(X_tr_scaled, y_tr_fold)
        y_proba_log = logreg.predict_proba(X_val_scaled)[:, 1]
        logreg_scores.append(average_precision_score(y_real_val, y_proba_log))
        
        # 2. XGBoost
        scale_pos_weight = (y_tr_fold == 0).sum() / (y_tr_fold == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=10,
        )
        xgb.fit(X_tr_fold, y_tr_fold)
        y_proba_xgb = xgb.predict_proba(X_real_val)[:, 1]
        xgb_scores.append(average_precision_score(y_real_val, y_proba_xgb))
        
        # 3. SVM
        svm = SVC(kernel="linear", class_weight="balanced", random_state=10)
        svm.fit(X_tr_scaled, y_tr_fold)
        y_score_svm = svm.decision_function(X_val_scaled)
        svm_scores.append(average_precision_score(y_real_val, y_score_svm))

    print("=== Real-Fold Cross-Validated PR-AUC ===")
    print(f"Logistic Regression: {np.mean(logreg_scores):.3f} ± {np.std(logreg_scores):.3f}")
    print(f"XGBoost:             {np.mean(xgb_scores):.3f} ± {np.std(xgb_scores):.3f}")
    print(f"SVM:                 {np.mean(svm_scores):.3f} ± {np.std(svm_scores):.3f}\n")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, sim_mask = load_processed()
    
    # Run fixed CV
    compare_models_real_cv(X_train, y_train, sim_mask)

    print("Logistic Regression (Test Set Performance):")
    model, scaler = train_log(X_train, y_train)
    evaluate(model, scaler, X_test, y_test)
    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
    print("\nTop Coefficients:\n", coefs.head(10))

    print("\nXGBoost (Test Set Performance):")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_xgb(xgb_model, X_test, y_test)

    print("\nSVM (Test Set Performance):")
    svm_model = train_svm(scaler.transform(X_train), y_train)
    evaluate_svm(svm_model, scaler, X_test, y_test)

    # Save models
    Path("output").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "output/logreg.joblib")
    joblib.dump(scaler, "output/scaler.joblib")
    joblib.dump(xgb_model, "output/xgb.joblib")
    joblib.dump(svm_model, "output/svm.joblib")