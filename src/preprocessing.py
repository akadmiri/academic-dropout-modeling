from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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

# Update the academic qualification for both parents to make it more significant and to get more information
ACADEMIC_QUALIFICATION = {
    35: 0, 36: 0,                                                          # 0: no education
    11: 1, 26: 1, 30: 1, 37: 1,                                            # 1: primary education
    12: 2, 14: 2, 19: 2, 27: 2, 9: 2, 10: 2, 15: 2, 25: 2, 38: 2, 29:2,    # 2: middle school equiv.
    1: 3, 13: 3, 18: 3, 20: 3, 22: 3, 31: 3, 33: 3, 6: 3,                  # 3: secondary
    2: 4, 3: 4, 40: 4, 41: 4, 42: 4, 39: 4,                                # 4: Higher education
    4: 4, 43: 4, 5: 4, 44: 4,                                              
    # 34 (Unknown) treated as missing.
}

def encode_qualifications(df: pd.DataFrame, columns=("Previous qualification", "Mother's qualification", "Father's qualification")) -> pd.DataFrame:
    """ Transforms raw Portuguese nominal qualification nomenclature into a 0-4 ordinal scale."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].map(ACADEMIC_QUALIFICATION).astype('Int64')
    return df

#Update occupation for both parents to get more information out of it
OCCUPATION_MAPPING = {
    # 0: Student
    0: 0,
    
    # 1: Directors/Executives
    1: 1, 112: 1, 114: 1,
    
    # 2: Specialists / Professionals
    2: 2, 121: 2, 122: 2, 123: 2, 124: 2, 125: 2,
    
    # 3: Technicians / Associate Professionals
    3: 3, 131: 3, 132: 3, 134: 3, 135: 3,
    
    # 4: Administrative / Clerical
    4: 4, 141: 4, 143: 4, 144: 4,
    
    # 5: Services and Sales
    5: 5, 151: 5, 152: 5, 153: 5, 154: 5,
    
    # 6: Agriculture / Fisheries
    6: 6, 161: 6, 163: 6,
    
    # 7: Skilled Trades / Industry
    7: 7, 171: 7, 172: 7, 173: 7, 174: 7, 175: 7,
    
    # 8: Machine Operators / Assembly
    8: 8, 181: 8, 182: 8, 183: 8,
    
    # 9: Unskilled / Elementary Occupations
    9: 9, 191: 9, 192: 9, 193: 9, 194: 9, 195: 9,
    
    # 10: Armed Forces
    10: 10, 101: 10, 102: 10, 103: 10,
    
    # Missing / Other (90, 99) left unmapped to become NaN
}

def encode_occupations(df: pd.DataFrame, columns=("Mother's occupation", "Father's occupation")) -> pd.DataFrame:
    """ 
    Reduces the cardinality of occupational codes to 11 nominal macro-categories. 
    Unmapped values (90, 99, etc.) are converted to pd.NA.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(OCCUPATION_MAPPING).astype('Int64')
    return df

UPDATED_COLUMNS = [
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Select the features to keep and drop the ones that aren't significant
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces the semester-1-only decision boundary and drops columns with negligible
    target association."""

    to_drop = [
        c for c in LEAKAGE_COLUMNS + NEGLIGIBLE_SIGNAL_COLUMNS if c in df.columns
    ]
    return df.drop(columns=to_drop)

# Treat missing data:
def data_filler(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: List[str])-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values in ordinal columns using the median of the training set.
    Prevents data leakage by fitting strictly on out-of-fold data.
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    cols_to_fill = [c for c in columns if c in train_df.columns]
    if not cols_to_fill:
        return train_df, test_df
    imputer = SimpleImputer(strategy='median')

    #fit strictly on the training set to not leak any data to the models
    train_df[cols_to_fill] = imputer.fit_transform(train_df[cols_to_fill])
    test_df[cols_to_fill] = imputer.transform(test_df[cols_to_fill])

    return train_df, test_df

ONE_HOT_COLUMNS = [
    "Course",
    "Application mode",
    "Evaluation status 1st sem",
]

TARGET_ENCODE_COLUMNS = [
    "Father's occupation",
    "Mother's occupation",
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
    delta_df = encode_qualifications(delta_df)
    delta_df = encode_occupations(delta_df)
    selected_df = select_features(delta_df)
    train_data, test_data = split_data(selected_df)

    qualifications = [
        "Previous qualification", 
        "Mother's qualification", 
        "Father's qualification"
    ]
    train_data, test_data = data_filler(train_data, test_data, qualifications)

    train_data, test_data = one_hot_encode(train_data, test_data, ONE_HOT_COLUMNS)
    train_data, test_data = target_encode(train_data, test_data, TARGET_ENCODE_COLUMNS)


    #Missing Values check
    train_nulls = train_data.isna().sum()
    test_nulls = test_data.isna().sum()
    
    print("Columns with missing values in Train:")
    print(train_nulls[train_nulls > 0] if train_nulls.sum() > 0 else "None")
    
    print("\nColumns with missing values in Test:")
    print(test_nulls[test_nulls > 0] if test_nulls.sum() > 0 else "None")
    
    # Save the processed datasets to CSV files
    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)

    print(
        f"Processed dataset saved: {train_data.shape[0]} training rows, {test_data.shape[0]} testing rows"
    )
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
