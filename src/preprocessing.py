from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def load_data(input_path: str, target_col: str = "Target") -> pd.DataFrame:
    """Loads raw data from a CSV file and filters it binarize the target variable for dropout prediction."""

    df = pd.read_csv(input_path)
    df = df[df[target_col].isin(["Dropout", "Graduate"])].copy()
    df["churn_target"] = df[target_col].map({"Dropout": 1, "Graduate": 0})
    df = df.drop(columns=[target_col])
    return df


def delta(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the first-order derivatives of academic performance."""

    df = df.copy()

    # Semester 1 efficiency
    e1 = df["Curricular units 1st sem (enrolled)"]
    a1 = df["Curricular units 1st sem (approved)"]
    df["Efficiency 1st sem"] = np.where(e1 > 0, a1 / e1, 0)

    ev1 = df["Curricular units 1st sem (evaluations)"]
    gr1 = df["Curricular units 1st sem (grade)"]
    df["Evaluation status 1st sem"] = np.select(
        condlist=[ev1 == 0, (ev1 > 0) & (gr1 == 0)],
        choicelist=["no_evaluation", "evaluated_zero"],
        default="positive_grade",
    )

    return df


def split_data(
    df: pd.DataFrame, target_col: str = "churn_target", test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets, while preserving the exact base rate of the target class."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=10
    )
    return train_df, test_df


LEAKAGE_COLUMNS = [
    # Semester 2 raw columns — unavailable at prediction time for a currently-enrolled student
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 2nd sem (grade)",
]

NEGLIGIBLE_SIGNAL_COLUMNS = [
    # Cramér's V / Cohen's d ~0 and not statistically significant in the exploration notebook.
    "Unemployment rate",
    "GDP",
    "Inflation rate",
    "International",
    "Educational special needs",
    "Nacionality",
    "Daytime/evening attendance",
    "Marital Status",
    "Displaced",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces the semester-1-only decision boundary and drops columns with negligible
    target association."""

    to_drop = [
        c for c in LEAKAGE_COLUMNS + NEGLIGIBLE_SIGNAL_COLUMNS if c in df.columns
    ]
    return df.drop(columns=to_drop)


ONE_HOT_COLUMNS = [
    "Course",
    "Application mode",
    "Previous qualification",
    "Evaluation status 1st sem",
]

TARGET_ENCODE_COLUMNS = [
    "Father's occupation",
    "Mother's occupation",
    "Father's qualification",
    "Mother's qualification",
]


def target_encode(
    train_df, test_df, columns, target_col="churn_target", smoothing=10, n_splits=5
):
    """Encodes high-cardinality nominal columns as the smoothed target mean, computed
    out-of-fold on the training set to avoid leakage, and from full-train statistics
    when encoding the test set."""
    train_df, test_df = train_df.copy(), test_df.copy()
    global_mean = train_df[target_col].mean()

    for col in columns:
        train_df[f"{col}_te"] = np.nan
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=10)
        for tr_idx, val_idx in kf.split(train_df):
            stats = (
                train_df.iloc[tr_idx].groupby(col)[target_col].agg(["mean", "count"])
            )
            smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
                stats["count"] + smoothing
            )
            train_df.iloc[val_idx, train_df.columns.get_loc(f"{col}_te")] = (
                train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean)
            )

        full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        full_smoothed = (
            full_stats["mean"] * full_stats["count"] + global_mean * smoothing
        ) / (full_stats["count"] + smoothing)
        test_df[f"{col}_te"] = test_df[col].map(full_smoothed).fillna(global_mean)

        train_df, test_df = train_df.drop(columns=[col]), test_df.drop(columns=[col])

    return train_df, test_df


def one_hot_encode(train_df, test_df, columns):
    """One-hot encodes moderate-cardinality columns, aligning test columns to train so
    a category present only in one split doesn't break downstream shapes."""
    train_enc = pd.get_dummies(train_df, columns=columns)
    test_enc = pd.get_dummies(test_df, columns=columns)
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
    return train_enc, test_enc


if __name__ == "__main__":
    input_path = "data/raw/dropout.csv"

    clean_df = load_data(input_path)
    delta_df = delta(clean_df)
    selected_df = select_features(delta_df)
    train_data, test_data = split_data(selected_df)
    train_data, test_data = one_hot_encode(train_data, test_data, ONE_HOT_COLUMNS)
    train_data, test_data = target_encode(train_data, test_data, TARGET_ENCODE_COLUMNS)

    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)

    print(
        f"Processed dataset saved: {train_data.shape[0]} training rows, {test_data.shape[0]} testing rows"
    )
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(train_data.filter(like="_te").head())
