import joblib
from train import load_processed
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test = load_processed()
    model = joblib.load("output/model.joblib")
    scaler = joblib.load("output/scaler.joblib")
    xgb_model = joblib.load("output/xgbmodel.joblib")

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(scaler.transform(X_test)),
                                          display_labels=["Graduate", "Dropout"], ax=axes[0])
    axes[0].set_title("Logistic Regression")
    ConfusionMatrixDisplay.from_predictions(y_test, xgb_model.predict(X_test),
                                          display_labels=["Graduate", "Dropout"], ax=axes[1])
    axes[1].set_title("XGBoost")
    plt.tight_layout()
    plt.savefig("output/Confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()