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
  student's course load actually converted to passed units — the strongest single predictor.
- `Evaluation status 1st sem`: a 3-state categorical (`no_evaluation` / `evaluated_zero` /
  `positive_grade`) resolving the fact that a raw grade of 0 conflates two very different
  situations — never sitting an evaluation, vs. sitting one and failing it.

**Leakage boundary** (`LEAKAGE_COLUMNS`): all six 2nd-semester curricular columns
(credited/enrolled/evaluations/approved/without evaluations/grade) are dropped — unavailable
at the semester-1 decision point for a currently-enrolled student.

**Multicollinearity** (`REDUNDANT_COLUMNS`): 1st-semester `approved` and `enrolled` are dropped
as structurally embedded in `Efficiency 1st sem` (VIF: approved = 21, efficiency = 16 when
both present). `credited` and `evaluations` are dropped alongside them — non-significant
individually (Mann-Whitney p_adj > 0.2) and part of the same near-identity block, so keeping
them only inflates VIF further without adding signal. Removing all four confirmed no PR-AUC
loss (~0.930 before and after), which is the actual evidence that they carried no independent
information beyond what `Efficiency 1st sem` and `grade` already capture.

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
macro-groups.


**Split** (`split_data`): stratified 80/20, `random_state=10`, preserving the exact class base
rate in both splits.

### 3. Training — `src/train.py`

Three models share the same final feature set deliberately, to keep the pipeline simple rather
than maintaining per-model feature sets:

- **Logistic Regression** — `class_weight="balanced"`, standardized via a `StandardScaler` fit
  once on the training set.
- **XGBoost** — `scale_pos_weight` for imbalance, unscaled inputs (tree-based, doesn't need it).
- **SVM** (linear kernel) — `class_weight="balanced"`, reuses the *same fitted scaler* from
  logistic regression rather than refitting one, so both linear models see identically scaled
  inputs and no scaler is fit twice on the same data.

5-fold stratified cross-validation on PR-AUC precedes the final single fit, for a variance
estimate alongside the point estimate.

### 4. Validation — `src/validate.py`

Confusion matrices for all three models against the held-out real test set.

### 5. Exploration — `notebooks/exploration.ipynb`

EDA notebook (French). Variable typing and univariate stats, distribution plots, bivariate
analysis restricted to semester-1/admission-time features (boxplots, Cohen's d, Mann-Whitney U
with Benjamini-Hochberg correction), categorical association (Cramér's V), correlation matrix
and VIF on the curricular-units block.

Statistical findings worth flagging: `Efficiency 1st sem` has by far the largest effect
(d ≈ −1.85, p < 1e-300). Parental qualification is asymmetric — `Mother's qualification` is
significant after correction (p_adj = 0.003), `Father's qualification` is not (p_adj = 0.55) —
documented rather than treated as symmetric. `Application order` has not yet been run through
significance testing.

## Class imbalance & evaluation choice

~39% Dropout / 61% Graduate after excluding `Enrolled`. Handled via class weighting
(`class_weight="balanced"` / `scale_pos_weight`), not oversampling. **PR-AUC (average
precision)** on the Dropout class is the primary metric, over ROC-AUC, given the imbalance and
because in an intervention context, missing an at-risk student (false negative) is generally
costlier than an unnecessary outreach (false positive) — precision/recall is tracked
explicitly rather than collapsed into a single accuracy number.

## Current results (real data, 2904 train / 726 test)

| Model | CV PR-AUC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.930 ± 0.010 | 0.927 |
| XGBoost | 0.929 ± 0.010 | 0.923 |
| SVM | 0.931 ± 0.011 | 0.926 |

Logistic regression coefficients are the primary interpretability artifact (trustworthy since
the VIF cleanup — see multicollinearity above). Full report in `results.md`. Strongest
predictors: `Efficiency 1st sem`, 1st-semester grade, then financial-pressure indicators
(`Tuition fees up to date`, `Debtor`, `Scholarship holder`).

## Known open decisions

- **Occupation encoding**: target encoding vs. one-hot — see above, not yet resolved.
- **Precision/recall operating point**: current model sits at 0.86 precision / 0.82 recall on
  Dropout. This is a business decision (cost of a missed at-risk student vs. cost of an
  unnecessary intervention) requiring supervisor input, not something to resolve unilaterally.
- **Data simulation**: expanding to 20,000 rows via simulation, per supervisor request.
  Method under active discussion.

## Next steps


1. Build the simulation module and re-validate model performance on the expanded dataset, real-only test set.
