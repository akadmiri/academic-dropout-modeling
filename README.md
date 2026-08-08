# Academic Dropout Modeling

A prediction model for early identification of students at risk of academic dropout.

The project uses the UCI _Predict Students' Dropout and Academic Success_ dataset as a development benchmark and is designed as a foundation for a future early-warning system using private student data.

> **Project status:** Active development

---

## Overview

Student dropout is rarely caused by a single factor. Academic performance, engagement, educational background, and other characteristics can interact in ways that make early identification difficult.

This project explores whether machine learning can identify students at higher risk of dropping out before their final outcome is known.

The main constraint is deliberately practical:

> **The model can only use information that would actually be available at the end of a student's first semester.**

This means that information from the second semester is excluded from model development, even when it is available in the original dataset. This prevents data leakage and makes the problem closer to a real-world early-warning scenario.

The longer-term objective is to adapt the resulting methodology to private student data and use it as part of a student retention and intervention system.

---

## Objectives

The project focuses on four main goals:

- Build a reliable early-warning classification model.
- Identify the academic and demographic factors most associated with dropout.
- Compare several machine learning approaches.
- Establish a reproducible methodology that can later be transferred to real-world student data.

The current implementation compares:

- **Logistic Regression**
- **XGBoost**
- **SVM**

---

## Dataset

The development dataset is the publicly available **UCI Predict Students' Dropout and Academic Success** dataset.

It contains student information collected around admission and during the first two academic semesters. The original target contains three outcomes:

- `Dropout`
- `Enrolled`
- `Graduate`

For this project, the problem is formulated as a binary classification task.

Students whose final outcome is `Dropout` are treated as positive cases, while `Graduate` students form the negative class.

The `Enrolled` class is excluded from model training because its final outcome is unresolved at the time represented by the data. Treating these students as either graduates or dropouts would introduce label noise.

This is consistent with the intended use case: **currently enrolled students are the population that the future early-warning system should evaluate.**

Dataset source: [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## Methodology

The pipeline is organized into several stages.

### 1. Data Loading

The dataset is retrieved programmatically and stored locally.

```text
src/data.py
```

The raw dataset is saved under:

```text
data/raw/dropout.csv
```

---

### 2. Preprocessing and Feature Engineering

The preprocessing stage prepares the data for modeling while enforcing the first-semester prediction boundary.

Key steps include:

- Converting the original three-class target into a binary dropout target.
- Removing second-semester variables that would not be available at prediction time.
- Handling missing values.
- Recoding educational qualifications into broader education levels.
- Grouping detailed parental occupations into broader occupational categories.
- Removing highly redundant variables.
- Removing variables with negligible predictive signal.
- Creating derived academic indicators.

One of the main engineered variables is:

- **First-semester efficiency**

```text
approved courses / enrolled courses
```

This provides a simple measure of how much of a student's first-semester workload was successfully completed.

Another engineered feature captures the distinction between:

- no evaluation,
- evaluation with a zero result,
- and a positive evaluation.

This distinction is useful because a raw grade of zero can represent very different academic situations.

---

### 3. Train/Test Split

After preprocessing, the data is divided using a stratified 80/20 split.

The split uses a fixed random seed to make experiments reproducible while preserving the dropout/graduate class distribution.

The test set remains completely separate from the synthetic-data generation process and is used only for final evaluation.

---

### 4. Simulation

Because the real training dataset is relatively small, the project also experiments with conditional synthetic data generation.

The goal is to increase the size of the training data while preserving important characteristics of the real population.

The current approach combines:

- Gaussian-copula modeling for selected continuous variables.
- Class-conditional bootstrap sampling for discrete variables.
- Distribution and realism checks comparing synthetic observations with the original training data.

Synthetic observations are clearly identified and are never used in the held-out test set.

This component is still experimental and is being refined as the project develops.

---

### 5. Model Training

Three models are currently evaluated using the same final feature set.

|Model|Configuration|
|---|---|
|Logistic Regression|Balanced class weights + standardized features|
|XGBoost|Class weighting through `scale_pos_weight`|
|Linear SVM|Balanced class weights + standardized features|

Using the same feature set across models keeps the comparison straightforward and makes differences in performance easier to interpret.

---

## Handling Class Imbalance

After removing the unresolved `Enrolled` class, the dataset contains approximately:

- **39% Dropout**
- **61% Graduate**

Instead of oversampling the minority class, the current pipeline uses class weighting.

This includes:

- `class_weight="balanced"` for Logistic Regression and SVM.
- `scale_pos_weight` for XGBoost.

This approach avoids unnecessarily modifying the original observations while still accounting for the unequal class distribution.

---

## Evaluation

Because the objective is to identify students who may require intervention, **PR-AUC (Average Precision)** is used as the primary evaluation metric.

This is more informative than relying exclusively on accuracy or ROC-AUC when the positive class is relatively less frequent and false negatives can have a meaningful practical cost.

The validation stage includes:

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion matrices
- Bootstrap confidence intervals for PR-AUC

The final evaluation is performed on a held-out test set containing only real observations.

Detailed experimental results are available in `RESULTS.md`.

---

## Exploratory Analysis

The exploratory analysis is contained in:

```text
notebooks/exploration.ipynb
```

It covers:

- Data quality and variable types
- Univariate distributions
- Academic-performance analysis
- Effect-size analysis
- Categorical associations
- Correlation analysis
- Multicollinearity analysis
- Feature-level statistical testing

One of the strongest signals identified so far is first-semester academic efficiency, which shows a substantially larger effect than most other individual variables.

The exploratory analysis is used to inform feature engineering rather than simply selecting variables based on model performance.

---

## Project Structure

```text
academic-dropout-modeling/
│
├── data/
│   ├── raw/
│   │   └── dropout.csv
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── simulation.py
│   ├── train.py
│   └── validate.py
│
├── RESULTS.md
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

Clone the repository:

```bash
git clone https://github.com/akadmiri/academic-dropout-modeling.git
cd academic-dropout-modeling
```

Create the environment and install dependencies:

```bash
uv sync
```

Activate the environment if required by your shell:

```bash
source .venv/bin/activate
```

---

## Running the Pipeline

The project is organized as a modular pipeline rather than a single script.

The general workflow is:

```text
Data
  ↓
Exploration
  ↓
Preprocessing
  ↓
Simulation
  ↓
Model Training
  ↓
Validation
  ↓
Results
```

Individual stages can be run through the corresponding modules in `src/`.

Refer to the source files and `RESULTS.md` for the current execution workflow and experimental configuration.

---

## Key Decisions

Several decisions are intentional and important to the project.

### No second-semester information

Second-semester variables are excluded because they would not exist at the intended prediction point.

Using them would produce artificially strong results while making the model unsuitable for genuine early intervention.

### Unresolved students are excluded from training

`Enrolled` students represent an unresolved outcome rather than a clean negative class.

Keeping them out of training avoids forcing ambiguous observations into either the dropout or graduate category.

### The test set remains real

Synthetic observations are used only as an augmentation strategy for training.

The final test set contains real observations that were not involved in synthetic-data generation.

This provides a more meaningful estimate of how the model performs on real students.

### PR-AUC as the primary metric

The objective is not simply to maximize the number of correct predictions.

The more important question is:

> **How effectively can the model identify students who are actually at risk of dropping out?**

This makes precision-recall based evaluation particularly relevant.

---

## Limitations and Ongoing Work

This project is still under development.
Current areas of improvement include:

- Optimizing decision thresholds based on the cost of false positives versus false negatives.
- Improving model interpretability.
- Testing the methodology on real institutional data.
