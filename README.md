# Academic Dropout Modeling

## Objective

A temporal binary classification pipeline predicting student dropout (décrochage scolaire), where the eventual goal is an early-warning model for student churn on YOOL's private data.

The model is scoped as an **early-warning system**: only features available at admission time
and during the first semester are used. A currently-enrolled student doesn't have second-semester
data yet, so second-semester columns are deliberately excluded from training, not just unused.
The prediction cutoff is defined as end of semester 1.

## Data

[UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
(id 697), fetched via `src/data.py` using the `ucimlrepo` API. The original target has three
classes (`Dropout` / `Enrolled` / `Graduate`). `Enrolled` rows are excluded from training since
their outcome is still unresolved — it's a censored class, and merging it into either resolved
class would introduce label noise into whichever one absorbed it. At inference time,
currently-enrolled students are exactly the population this model is meant to score.

## Pipeline

### 1. Data loading — `src/data.py`

Fetches dataset 697, saves the raw CSV to `data/raw/dropout.csv`.

### 2. Preprocessing — `src/preprocessing.py`

**Target binarization** (`load_data`): keeps `Dropout`/`Graduate`, drops `Enrolled`, maps to
`churn_target` (1 = Dropout).

**Feature engineering** (`delta`):

- `Efficiency 1st sem` = approved / enrolled (0 when enrolled = 0). Captures how much of a
  student's course load actually converted to passed units — the strongest single predictor
  on real data.
- `Evaluation status 1st sem`: a 3-state categorical (`no_evaluation` / `evaluated_zero` /
  `positive_grade`) resolving the fact that a raw grade of 0 conflates two very different
  situations — never sitting an evaluation, vs. sitting one and failing it.

**Leakage boundary** (`LEAKAGE_COLUMNS`): all six 2nd-semester curricular columns
(credited/enrolled/evaluations/approved/without evaluations/grade) are dropped — unavailable
at the semester-1 decision point for a currently-enrolled student.

**Multicollinearity** (`REDUNDANT_COLUMNS`): 1st-semester `approved` and `enrolled` are dropped
as structurally embedded in `Efficiency 1st sem`, along with `credited` and `evaluations`
(non-significant individually and part of the same near-identity block). VIF was confirmed via
`variance_inflation_factor` in `notebooks/exploration.ipynb`.

**Negligible signal** (`NEGLIGIBLE_SIGNAL_COLUMNS`): dropped on effect size, not p-value alone —
`Unemployment rate`, `GDP`, `Inflation rate`, `International`, `Educational special needs`,
`Nacionality`, `Daytime/evening attendance`, `Marital Status`, `Displaced`.

**Qualification recoding** (`encode_qualifications` / `ACADEMIC_QUALIFICATION`): `Previous
qualification`, `Mother's qualification`, `Father's qualification` are recoded onto a 0–4
ordinal scale (no education → primary → middle-school-equivalent → secondary → higher
education), collapsing Portugal's granular qualification nomenclature into broad education
tiers. Unknown/blank codes route to missing, median-imputed on the training set only
(`data_filler`).

**Occupation regrouping** (`encode_occupations` / `OCCUPATION_MAPPING`): `Mother's occupation`,
`Father's occupation` are collapsed from granular job-title codes into 11 broad occupational
macro-groups (student, executives, professionals, ..., armed forces). Unmapped codes become
missing, filled with an explicit sentinel category (`fill_unknown`) rather than imputed, since
there's no meaningful "median occupation."

**Split** (`split_data`): stratified 80/20, `random_state=10`, preserving the exact class base
rate in both splits.

### 3. Simulation — `src/simulation.py`

Augments the real training set with class-conditional synthetic rows, targeting ~25k total rows
as requested by the supervisor:

- **Continuous block** (`Efficiency 1st sem`, `Admission grade`, `Previous qualification
  (grade)`, `Age at enrollment`, `Curricular units 1st sem (without evaluations)`, `Application
  order`): modeled jointly per class with a Gaussian copula — each variable's empirical CDF is
  transformed to a latent normal, and an LKJ-prior correlation matrix over the latent space is
  fit via PyMC's NUTS sampler — so the real pairwise correlations within this block are
  preserved in the synthetic draws.
- **Discrete block**: joint class-conditional bootstrap — each synthetic row's whole discrete
  profile (course, application mode, occupation/qualification tiers, financial-pressure flags,
  evaluation status, gender) is copied from one real student, so every real pairwise association
  among discrete features is preserved exactly.
- **`Curricular units 1st sem (grade)`**: currently sampled separately from the continuous
  copula — zero wherever `Evaluation status` forces it, otherwise resampled from real positive
  grades in the same class with light jitter. This means grade is **not** constrained to
  correlate with `Efficiency 1st sem` or the rest of the continuous block in synthetic rows,
  unlike in real data. This is an open item — see below.
- **Realism check**: a logistic-regression discriminator trained to separate real from synthetic
  rows scores ≈0.53 AUC (near chance), which is good marginal-distribution evidence. A linear
  discriminator can't detect the kind of broken pairwise correlation described above, so this
  doesn't yet validate the joint structure of the continuous block.
- Synthetic rows are tagged (`data/processed/train_simulated_mask.csv`); the held-out test set
  is always 100% real and untouched by the generation process.

### 4. Training — `src/train.py`

Three models share the same final feature set deliberately, to keep the pipeline simple rather
than maintaining per-model feature sets:

- **Logistic Regression** — `class_weight="balanced"`, standardized via a `StandardScaler` fit
  once on the training set.
- **XGBoost** — `scale_pos_weight` for imbalance, unscaled inputs (tree-based, doesn't need it).
- **SVM** (linear kernel) — `class_weight="balanced"`, reuses the *same fitted scaler* from
  logistic regression rather than refitting one. SVM's score in evaluation is `decision_function`
  (an unbounded margin, not a calibrated probability) — fine for PR-AUC ranking, not a substitute
  for `predict_proba` if a calibrated risk score is ever needed from this model specifically.

`compare_models_real_cv` cross-validates strictly on real-data folds (synthetic rows, when
present, are added to each fold's training portion but never appear in a validation fold), giving
a variance estimate that isn't contaminated by scoring the model on synthetic data. Note: the
synthetic set itself is generated once from the *full* real training set before any CV split
happens, so each fold's added synthetic data is informed in part by that fold's own held-out
rows — a mild optimistic bias in the CV estimate that does not affect the final test-set numbers
below, since the test set was never involved in synthetic generation.

### 5. Validation — `src/validate.py`

Classification reports and confusion matrices for all three models against the held-out real
test set, plus a bootstrap (2,000 resamples) 95% confidence interval on PR-AUC per model.

### 6. Exploration — `notebooks/exploration.ipynb`

EDA notebook. Variable typing and univariate stats, distribution plots, bivariate
analysis restricted to semester-1/admission-time features (boxplots, Cohen's d, Mann-Whitney U
with Benjamini-Hochberg correction), categorical association (Cramér's V), correlation matrix
and VIF on the curricular-units block.

Statistical findings worth flagging: `Efficiency 1st sem` has by far the largest effect on real
data (d ≈ −1.85, p < 1e-300). Parental qualification is asymmetric — `Mother's qualification` is
significant after correction (p_adj = 0.003), `Father's qualification` is not (p_adj = 0.55).
**Open**: `Application order` has never been run through significance testing despite being used
as a model feature, and the post-recoding occupation macro-groups (`encode_occupations` output)
haven't been re-tested — the notebook's categorical association section still runs on the raw,
high-cardinality occupation codes.

## Class imbalance & evaluation choice

~39% Dropout / 61% Graduate after excluding `Enrolled`. Handled via class weighting
(`class_weight="balanced"` / `scale_pos_weight`), not oversampling. **PR-AUC (average
precision)** on the Dropout class is the primary metric, over ROC-AUC, given the imbalance and
because in an intervention context, missing an at-risk student (false negative) is generally
costlier than an unnecessary outreach (false positive).
