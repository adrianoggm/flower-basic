# SWELL Baseline

* Train samples: 2500
* Test samples: 639
* Features: 17

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| logistic_regression | 0.953 | 0.948 |
| random_forest | 0.992 | 0.991 |

## Subject-Based 5-Fold Cross-Validation

| Model | Accuracy (mean +/- std) | Macro F1 (mean +/- std) |
| --- | --- | --- |
| logistic_regression | 0.951 +/- 0.009 | 0.946 +/- 0.009 |
| random_forest | 0.989 +/- 0.006 | 0.987 +/- 0.008 |

Fuente: subject_cv_results/subject_cv_summary.csv.

