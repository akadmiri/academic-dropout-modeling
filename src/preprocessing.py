from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    input_path = "data/raw/dropout.csv"

    clean_df = load_data(input_path)
    delta_df = delta(clean_df)
    train_data, test_data = split_data(delta_df)

    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)

    print(
        f"Processed dataset saved: {train_data.shape[0]} training rows, {test_data.shape[0]} testing rows"
    )
