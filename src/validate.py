import joblib
from train import load_processed
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test, sim_mask = load_processed()
    logreg_model = joblib.load("output/logreg.joblib")
    scaler = joblib.load("output/scaler.joblib")
    xgb_model = joblib.load("output/xgb.joblib")
    svm_model = joblib.load("output/svm.joblib")

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

if __name__ == "__main__":
    main()