# Academic Dropout Modeling

## Objective

A temporal binary classification pipeline predicting student dropout (décrochage scolaire),
built as practice ahead of an internship project at YOOL Education, where the eventual goal
is an early-warning model for student churn on YOOL's private data.

The model is scoped as an **early-warning system**: only features available at admission time
and during the first semester are used. A currently-enrolled student doesn't have second-semester
data yet, so second-semester columns are deliberately excluded from training, not just unused —
see `src/preprocessing.py::select_features`.

## Data

[UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
(id 697), fetched via the `ucimlrepo` API. The original target has three classes
(`Dropout` / `Enrolled` / `Graduate`); `Enrolled` rows are excluded from training since their
outcome is still unresolved — merging them into either class would introduce label noise into
whichever class absorbed them. At inference time, currently-enrolled students are exactly the
population this model is meant to score.
