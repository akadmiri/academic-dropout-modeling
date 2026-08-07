from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import (
    load_data, delta, encode_qualifications, encode_occupations,
    select_features, split_data, data_filler, fill_unknown, one_hot_encode,
    ORDINAL_MISSING_COLUMNS, NOMINAL_MISSING_COLUMNS, ONE_HOT_COLUMNS,
)
from train import TARGET_COL

RANDOM_SEED = 10
N_SYNTHETIC_ROWS = 22000  # real (2904) + synthetic ≈ 25k

CONTINUOUS_COLUMNS = [
    "Efficiency 1st sem",
    "Admission grade",
    "Previous qualification (grade)",
    "Age at enrollment",
]
COUNT_COLUMNS = {"Age at enrollment"}

GRADE_COLUMN = "Curricular units 1st sem (grade)"
EVAL_STATUS_COLUMN = "Evaluation status 1st sem"

DISCRETE_COLUMNS = [
    "Course", "Application mode",
    "Mother's occupation", "Father's occupation",
    EVAL_STATUS_COLUMN, "Gender", "Debtor", "Scholarship holder", "Tuition fees up to date",
]


def build_preprocessed_pre_onehot(input_path="data/raw/dropout.csv"):
    """Runs the real preprocessing pipeline up to (not including) one-hot
    encoding, so train/test carry literal category labels."""
    clean_df = load_data(input_path)
    delta_df = delta(clean_df)
    delta_df = encode_qualifications(delta_df)
    delta_df = encode_occupations(delta_df)
    selected_df = select_features(delta_df)

    train_df, test_df = split_data(selected_df)
    train_df, test_df = data_filler(train_df, test_df, ORDINAL_MISSING_COLUMNS)
    train_df = fill_unknown(train_df, NOMINAL_MISSING_COLUMNS)
    test_df = fill_unknown(test_df, NOMINAL_MISSING_COLUMNS)
    return train_df, test_df


class ContinuousBlockModel:
    """Class-conditional Gaussian copula for the continuous block, fit via
    MCMC with an LKJ prior on the correlation matrix."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.ecdfs = {}
        self.trace = None
        self.columns = []

    def _to_latent(self, series: pd.Series, col: str) -> np.ndarray:
        self.ecdfs[col] = np.sort(series.values)
        u = series.rank(method="average").values / (len(series) + 1)
        u = np.clip(u, self.epsilon, 1 - self.epsilon)
        return norm.ppf(u)

    def _from_latent(self, latent: np.ndarray, col: str) -> np.ndarray:
        u = norm.cdf(latent)
        return np.quantile(self.ecdfs[col], u, method="inverted_cdf")

    def fit(self, class_df: pd.DataFrame, columns, seed: int):
        self.columns = columns
        latent = np.column_stack([self._to_latent(class_df[c], c) for c in columns])
        n_dims = len(columns)
        with pm.Model():
            sd_dist = pm.Exponential.dist(1.0, shape=n_dims)
            chol, _, _ = pm.LKJCholeskyCov("packed", n=n_dims, eta=2.0, sd_dist=sd_dist, compute_corr=True)
            chol = pm.Deterministic("chol", chol)  # store the full matrix, not just the packed draw
            mu = pm.Normal("mu", 0.0, 1.0, shape=n_dims)
            pm.MvNormal("obs", mu=mu, chol=chol, observed=latent)
            self.trace = pm.sample(1000, tune=1000, chains=4, cores=4, random_seed=seed, progressbar=False)

    def sample(self, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
        post = self.trace.posterior
        n_draws = post.sizes["chain"] * post.sizes["draw"]
        mu_samples = post["mu"].values.reshape(n_draws, -1)
        chol_samples = post["chol"].values.reshape(n_draws, len(self.columns), len(self.columns))

        draw_idx = rng.integers(0, n_draws, size=n_samples)
        latent = np.empty((n_samples, len(self.columns)))
        for i, d in enumerate(draw_idx):
            z = rng.standard_normal(len(self.columns))
            latent[i] = mu_samples[d] + chol_samples[d] @ z

        out = {c: self._from_latent(latent[:, i], c) for i, c in enumerate(self.columns)}
        df = pd.DataFrame(out)
        for c in COUNT_COLUMNS:
            df[c] = df[c].round().astype(int)
        return df


def sample_discrete_block(class_df: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """Joint resampling from the real class-conditional data — every synthetic
    row's discrete profile is copied whole from one real student, so every
    real pairwise association in this block is preserved exactly."""
    idx = rng.integers(0, len(class_df), size=n_samples)
    return class_df[DISCRETE_COLUMNS].iloc[idx].reset_index(drop=True)


def sample_grade(eval_status: pd.Series, class_df: pd.DataFrame, rng: np.random.Generator,
                  jitter_frac: float = 0.05) -> np.ndarray:
    """0 wherever Evaluation status forces it (mirrors delta()'s own logic);
    resampled from real positive grades in this class otherwise, with light
    jitter so it isn't a literal copy."""
    positive_real = class_df.loc[class_df[EVAL_STATUS_COLUMN] == "positive_grade", GRADE_COLUMN].values
    grade = np.zeros(len(eval_status))
    mask = (eval_status == "positive_grade").values
    n_pos = int(mask.sum())
    if n_pos > 0:
        base = rng.choice(positive_real, size=n_pos, replace=True)
        jitter = rng.normal(0, positive_real.std() * jitter_frac, size=n_pos)
        grade[mask] = np.clip(base + jitter, 0.01, 20.0)
    return grade


def generate_class(class_df: pd.DataFrame, n_samples: int, seed: int, rng: np.random.Generator) -> pd.DataFrame:
    discrete = sample_discrete_block(class_df, n_samples, rng)

    continuous_model = ContinuousBlockModel()
    continuous_model.fit(class_df, CONTINUOUS_COLUMNS, seed=seed)
    continuous = continuous_model.sample(n_samples, rng)
    for c in CONTINUOUS_COLUMNS:  # keep values inside the real observed range
        continuous[c] = continuous[c].clip(class_df[c].min(), class_df[c].max())

    grade = sample_grade(discrete[EVAL_STATUS_COLUMN], class_df, rng)

    synthetic = pd.concat([discrete.reset_index(drop=True), continuous.reset_index(drop=True)], axis=1)
    synthetic[GRADE_COLUMN] = grade
    return synthetic

# Check the simulated data
def discriminator_check(real_pre_onehot: pd.DataFrame, synthetic_pre_onehot: pd.DataFrame,
                         discrete_columns, continuous_columns, seed: int = 10) -> float:
    """AUC of a classifier trained to tell real rows from synthetic ones.
    Near 0.5 = statistically convincing. Near 1.0 = something's off."""
    real = real_pre_onehot.copy()
    real["is_synthetic"] = 0
    synth = synthetic_pre_onehot.copy()
    synth["is_synthetic"] = 1
    combined = pd.concat([real, synth], ignore_index=True)

    X = pd.get_dummies(combined[discrete_columns + continuous_columns], columns=discrete_columns)
    y = combined["is_synthetic"]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc")
    return scores.mean()


def main():
    train_df, test_df = build_preprocessed_pre_onehot()
    rng = np.random.default_rng(RANDOM_SEED)

    base_rate = train_df[TARGET_COL].mean()
    n_dropout = int(round(N_SYNTHETIC_ROWS * base_rate))
    n_graduate = N_SYNTHETIC_ROWS - n_dropout

    synthetic_parts = []
    for cls, n_cls, seed in [(1, n_dropout, RANDOM_SEED), (0, n_graduate, RANDOM_SEED + 1)]:
        class_df = train_df[train_df[TARGET_COL] == cls]
        part = generate_class(class_df, n_cls, seed, rng)
        part[TARGET_COL] = cls
        synthetic_parts.append(part)
    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)

    # Check the synthetic data's realism with a discriminator
    auc = discriminator_check(train_df, synthetic_df, DISCRETE_COLUMNS, CONTINUOUS_COLUMNS + [GRADE_COLUMN], seed=RANDOM_SEED)
    print(f"Discriminator AUC (real vs synthetic): {auc:.3f}")

    train_tagged = train_df.copy()
    train_tagged["Simulated"] = 0
    synthetic_df["Simulated"] = 1
    augmented_train = pd.concat([train_tagged, synthetic_df], ignore_index=True)

    simulated_mask = augmented_train["Simulated"]
    augmented_train = augmented_train.drop(columns=["Simulated"])

    # real category labels in
    augmented_train_enc, test_enc = one_hot_encode(augmented_train, test_df, ONE_HOT_COLUMNS)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    augmented_train_enc.to_csv("data/processed/train_data.csv", index=False)
    test_enc.to_csv("data/processed/test_data.csv", index=False)
    simulated_mask.to_csv("data/processed/train_simulated_mask.csv", index=False)

    print(f"Real training rows: {len(train_df)}, synthetic rows: {len(synthetic_df)}")
    print(f"Augmented train shape: {augmented_train_enc.shape}, test shape: {test_enc.shape}")


if __name__ == "__main__":
    main()