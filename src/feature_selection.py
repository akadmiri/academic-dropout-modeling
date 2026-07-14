import pandas as pd


def feature_selection(
    df: pd.DataFrame, importance_path: str, threshold: float = 0.015
) -> pd.DataFrame:
    """Drops columns that provide no information about the target variable based on feature importance scores."""

    importance_df = pd.read_csv(importance_path)

    noise_features = importance_df[importance_df["importance"] < threshold][
        "feature"
    ].tolist()
    cols_to_drop = [col for col in noise_features if col in df.columns]
    df_reduced = df.drop(columns=cols_to_drop)
    return df_reduced


if __name__ == "__main__":
    train_path = "data/processed/train_data.csv"
    test_path = "data/processed/test_data.csv"
    importance_path = "data/processed/feature_importances.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    initial_cols = train_df.shape[1]

    train_df = feature_selection(train_df, importance_path)
    test_df = feature_selection(test_df, importance_path)
    print(f"Feature selection: {initial_cols - train_df.shape[1]} features dropped")

    # Save the processed datasets
    train_df.to_csv("data/processed/train_data_final.csv", index=False)
    test_df.to_csv("data/processed/test_data_final.csv", index=False)
    print(f"Final feature count: {train_df.shape[1] - 1}")
