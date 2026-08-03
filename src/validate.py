import numpy as np
import pandas as pd
import joblib
from train import load_processed
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Evaluate the test set  
def bootstrap_pr_auc_ci(y_true, y_proba, n_boot=2000, seed=10):
    rng = np.random.default_rng(seed)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = average_precision_score(y_true[idx], y_proba[idx])
    return np.percentile(scores, [2.5, 97.5])

# Model evaluation and confusion matrix generation
def evaluate(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_proba):.3f}")

def evaluate_xgb(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_proba):.3f}")


def evaluate_svm(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_score = model.decision_function(X_test_scaled)

    print(classification_report(y_test, y_pred, target_names=["Graduate", "Dropout"]))
    print(f"PR-AUC (Dropout): {average_precision_score(y_test, y_score):.3f}")


def main():
    X_train, X_test, y_train, y_test, sim_mask = load_processed()
    logreg_model = joblib.load("output/logreg.joblib")
    scaler = joblib.load("output/scaler.joblib")
    xgb_model = joblib.load("output/xgb.joblib")
    svm_model = joblib.load("output/svm.joblib")

    proba_log = logreg_model.predict_proba(scaler.transform(X_test))[:, 1]
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    score_svm = svm_model.decision_function(scaler.transform(X_test))

    for name, y_score in [("Logistic Regression", proba_log), ("XGBoost", proba_xgb), ("SVM", score_svm)]:
        point = average_precision_score(y_test, y_score)
        lo, hi = bootstrap_pr_auc_ci(y_test, y_score)
        print(f"{name}: PR-AUC {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, logreg_model.predict(scaler.transform(X_test)),
                                          display_labels=["Graduate", "Dropout"], ax=axes[0])
    axes[0].set_title("Logistic Regression")
    ConfusionMatrixDisplay.from_predictions(y_test, xgb_model.predict(X_test),
                                          display_labels=["Graduate", "Dropout"], ax=axes[1])
    axes[1].set_title("XGBoost")
    ConfusionMatrixDisplay.from_predictions(y_test, svm_model.predict(scaler.transform(X_test)),
                                          display_labels=["Graduate", "Dropout"], ax=axes[2])
    axes[2].set_title("SVM")
    plt.tight_layout()
    plt.savefig("output/Confusion_matrix.png")

    print("Logistic Regression (Test Set Performance):")
    evaluate(logreg_model, scaler, X_test, y_test)
    coefs = pd.Series(logreg_model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
    print("\nTop Coefficients:\n", coefs.head(10))

    print("\nXGBoost (Test Set Performance):")
    evaluate_xgb(xgb_model, X_test, y_test)
    
    print("\nSVM (Test Set Performance):") 
    evaluate_svm(svm_model, scaler, X_test, y_test)

if __name__ == "__main__":
    main()