# Results

## 1. Real data

> 2904 training rows, 726 testing rows
> Train shape: (2904, 71)
> Test shape: (726, 71)

### Cross-validated comparison

- Logistic Regression PR-AUC: 0.929 ± 0.010

- XGBoost PR-AUC: 0.930 ± 0.008

- SVM PR-AUC: 0.930 ± 0.011

### Logistic Regression

              precision    recall  f1-score   support

    Graduate       0.89      0.92      0.90       442
     Dropout       0.87      0.82      0.84       284

    accuracy                           0.88       726
   macro avg       0.88      0.87      0.87       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.927

- Top Coefficients

Efficiency 1st sem                         -2.502160
Curricular units 1st sem (grade)           -1.082023
Tuition fees up to date                    -0.942884
Evaluation status 1st sem_positive_grade    0.664061
Course_171                                 -0.648589
Evaluation status 1st sem_no_evaluation    -0.596444
Debtor                                      0.414777
Course_9853                                 0.374457
Scholarship holder                         -0.371774
Evaluation status 1st sem_evaluated_zero   -0.297675

### XGBoost

               precision    recall  f1-score   support

    Graduate       0.89      0.93      0.91       442
     Dropout       0.89      0.82      0.85       284

    accuracy                           0.89       726
   macro avg       0.89      0.88      0.88       726
weighted avg       0.89      0.89      0.89       726

> PR-AUC (Dropout): 0.928

### SVM

              precision    recall  f1-score   support

    Graduate       0.89      0.92      0.91       442
     Dropout       0.87      0.83      0.85       284

    accuracy                           0.88       726
   macro avg       0.88      0.87      0.88       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.928

## Simulated data 

> 22904 training rows, 726 testing rows
> Train shape: (22904, 71)
> Test shape: (726, 71)

### Cross-validated comparison

- Logistic Regression PR-AUC: 0.921 ± 0.010

- XGBoost PR-AUC: 0.941 ± 0.008

- SVM PR-AUC: 0.918 ± 0.010

### Logistic Regression

              precision    recall  f1-score   support

    Graduate       0.89      0.88      0.89       442
     Dropout       0.82      0.83      0.83       284

    accuracy                           0.87       726
   macro avg       0.86      0.86      0.86       726
weighted avg       0.87      0.87      0.87       726

> PR-AUC (Dropout): 0.905

- Top Coefficients

Curricular units 1st sem (grade)           -2.680469
Efficiency 1st sem                         -2.241966
Tuition fees up to date                    -1.012835
Course_171                                 -0.751165
Scholarship holder                         -0.521432
Evaluation status 1st sem_positive_grade    0.441837
Debtor                                      0.426965
Evaluation status 1st sem_no_evaluation    -0.397178
Course_9853                                 0.372514
Course_33                                   0.369270

### XGBoost

              precision    recall  f1-score   support

    Graduate       0.90      0.91      0.90       442
     Dropout       0.86      0.84      0.85       284

    accuracy                           0.88       726
   macro avg       0.88      0.87      0.88       726
weighted avg       0.88      0.88      0.88       726

> PR-AUC (Dropout): 0.929

### SVM

              precision    recall  f1-score   support

    Graduate       0.89      0.90      0.90       442
     Dropout       0.85      0.82      0.83       284

    accuracy                           0.87       726
   macro avg       0.87      0.86      0.86       726
weighted avg       0.87      0.87      0.87       726

> PR-AUC (Dropout): 0.908
