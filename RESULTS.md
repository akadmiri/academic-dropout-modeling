# Results

## 1. Real data

> 2904 training rows, 726 testing rows
> Train shape: (2904, 82)
> Test shape: (726, 82)

### Cross-validated comparison

- Logistic Regression PR-AUC: 0.930 ± 0.010

- XGBoost PR-AUC: 0.930 ± 0.008

- SVM PR-AUC: 0.930 ± 0.012

### Logistic Regression

             precision    recall  f1-score   support

    Graduate       0.89      0.91      0.90       442
     Dropout       0.86      0.82      0.84       284

    accuracy                           0.88       726
   macro avg       0.87      0.87      0.87       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.926

- Top Coefficients

Efficiency 1st sem                         -2.566961
Curricular units 1st sem (grade)           -1.126471
Tuition fees up to date                    -0.958403
Evaluation status 1st sem_positive_grade    0.695703
Evaluation status 1st sem_no_evaluation    -0.646939

### XGBoost

               precision    recall  f1-score   support

    Graduate       0.89      0.93      0.91       442
     Dropout       0.88      0.82      0.85       284

    accuracy                           0.88       726
   macro avg       0.88      0.87      0.88       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.929

### SVM

              precision    recall  f1-score   support

    Graduate       0.89      0.92      0.90       442
     Dropout       0.86      0.83      0.85       284

    accuracy                           0.88       726
   macro avg       0.88      0.87      0.87       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.929

## Simulated data 

> 22904 training rows, 726 testing rows
> Train shape: (22904, 82)
> Test shape: (726, 82)

### Cross-validated comparison

- Logistic Regression PR-AUC: 0.919 ± 0.011

- XGBoost PR-AUC: 0.942 ± 0.009

- SVM PR-AUC: 0.917 ± 0.011

### Logistic Regression

              precision    recall  f1-score   support

    Graduate       0.89      0.89      0.89       442
     Dropout       0.83      0.83      0.83       284

    accuracy                           0.87       726
   macro avg       0.86      0.86      0.86       726
weighted avg       0.87      0.87      0.87       726

> PR-AUC (Dropout): 0.904

- Top Coefficients

Curricular units 1st sem (grade)           -2.622261
Efficiency 1st sem                         -2.239258
Tuition fees up to date                    -1.089262
Course_171                                 -0.684752
Scholarship holder                         -0.470012
Course_33                                   0.452898
Evaluation status 1st sem_positive_grade    0.419524
Course_9853                                 0.399077
Debtor                                      0.392471
Evaluation status 1st sem_no_evaluation    -0.382312

### XGBoost

              precision    recall  f1-score   support

    Graduate       0.90      0.91      0.91       442
     Dropout       0.86      0.85      0.85       284

    accuracy                           0.89       726
   macro avg       0.88      0.88      0.88       726
weighted avg       0.89      0.89      0.89       726

> PR-AUC (Dropout): 0.932

### SVM

              precision    recall  f1-score   support

    Graduate       0.89      0.90      0.90       442
     Dropout       0.84      0.83      0.83       284

    accuracy                           0.87       726
   macro avg       0.87      0.86      0.87       726
weighted avg       0.87      0.87      0.87       726

> PR-AUC (Dropout): 0.906