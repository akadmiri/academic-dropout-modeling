import pandas as pd
import xgboost as xgb


def train_xgboost_model(train_path: str) -> pd.DataFrame:
    """Trains an XGBoost model on the training data to calculate feature importances."""

    df = pd.read_csv(train_path)
    X = df.drop(columns=["churn_target"])
    y = df["churn_target"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=10,
    )
    model.fit(X, y)

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    return importance_df


if __name__ == "__main__":
    train_path = "data/processed/train_data.csv"
    output_path = "data/processed/feature_importances.csv"

    importance = train_xgboost_model(train_path)
    importance.to_csv(output_path, index=False)
    print(f"Feature importances saved: {importance.shape[0]} features")

    print("Top 10 features:")
    print(importance.head(10))

    print("Bottom 10 features:")
    print(importance.tail(10))
