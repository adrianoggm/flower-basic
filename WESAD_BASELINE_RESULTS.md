# WESAD Baseline

* Train samples: 572
* Test samples: 287
* Features: 30

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| logistic_regression | 0.930 | 0.921 |
| random_forest | 0.962 | 0.959 |

## Subject-Based 5-Fold Cross-Validation

| Model | Accuracy (mean +/- std) | Macro F1 (mean +/- std) |
| --- | --- | --- |
| logistic_regression | 0.865 +/- 0.079 | 0.854 +/- 0.087 |
| random_forest | 0.768 +/- 0.083 | 0.738 +/- 0.081 |

Fuente: subject_cv_results/subject_cv_summary.csv.

