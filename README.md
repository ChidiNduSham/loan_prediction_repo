# Loan Approval Prediction: Feature Engineering & Data Preprocessing

**Author:** Shammah Ndubuisi
**Task:** Week 2, Feature Engineering & Data Preprocessing for Machine Learning

## Overview

A lender has to decide whether each loan application is approved (`Y`) or not (`N`). This project takes the raw loan application data and prepares it for machine learning: cleaning it, creating new features, encoding and scaling it, and selecting the features worth keeping.

No model is trained here. The Random Forest used in the notebook only ranks feature importance. The output is a clean, fully numeric, leakage-free dataset that is ready for the modelling stage.

## Repository Structure

```
├── ShammahNdubuisi_Week_2_Submission.ipynb   # Full workflow: inspection, cleaning, features, selection
├── Original_Data/                            # Raw data (if permitted)
│   ├── train_dataset.csv
│   └── test_dataset.csv
├── Cleaned_Data/                             # Cleaned, still human-readable
│   ├── train_cleaned.csv
│   ├── test_cleaned.csv
│   └── Preprocessing_Log.csv                 # Every step: what was done and why
├── ML_Ready_Data/                            # Encoded, capped, scaled, selected
│   ├── ml_ready_train.csv
│   ├── ml_ready_test.csv
│   └── test_loan_ids.csv                     # Matches test rows back to applicants
├── Reports/
│   ├── Business_Understanding_Report.pptx
│   └── Data_Preprocessing_Report.pptx
└── Visualizations/                           # 11 charts used in the notebook and reports
```

## The Data

| | Training set | Test set |
|---|---|---|
| Rows | 614 | 367 |
| Columns | 13 (12 features + `Loan_Status`) | 12 (no target) |

The test set has no `Loan_Status`, so it can be prepared in the same way as the training set but cannot be used to measure accuracy. About 69% of training applications are approved and 31% are not.

## What Was Done

**1. Cleaning**
- Renamed all columns to short, lower-case names.
- Checked for duplicate rows and repeated `loan_id`s (none found).
- Filled missing values with the training-set mode (categories, credit history, loan term, dependents) or median (loan amount).
- Fixed data types: `3+` dependents became `3`, and credit history and term became whole numbers.

**2. Feature engineering**

| Feature | Formula |
|---|---|
| `total_household_income` | applicant income + co-applicant income |
| `loan_income_ratio` | loan amount ÷ total household income |
| `emi` | loan amount ÷ loan term (rough monthly repayment) |
| `balance_income` | total household income − `emi` |

Loan amount is recorded in thousands, so it is multiplied by 1,000 before these calculations.

**3. Encoding**
- Label encoding (0/1): gender, married, education, self-employed, and the target.
- `no_of_dependents` kept as a number (0 to 3), because the order matters.
- One-hot encoding for `property_area` (with `drop_first=True`).

**4. Outliers and scaling**
- Outliers detected with the IQR rule (1.5 × IQR) and **capped**, not deleted.
- Eight continuous columns standardised with `StandardScaler`.

**5. Feature selection**
- Correlation heatmap, correlation with the target, and Random Forest feature importance.
- Removed `loan_id` (an identifier), `balance_income` and `emi` (redundant), and `gender` (no link to approval, and a sensitive attribute).

**No data leakage:** fill values, outlier limits and the scaler are all learned from the training set only and then applied unchanged to the test set.

## Final ML-Ready Dataset

13 features plus the target (`loan_status`: 1 = approved, 0 = not approved):

`married_or_not`, `no_of_dependents`, `education_level`, `self_employed_or_not`, `applicant_income`, `co_applicant_income`, `loan_amount`, `loan_amount_term`, `credit_history`, `total_household_income`, `loan_income_ratio`, `property_area_Semiurban`, `property_area_Urban`

Checks passed: no missing values, all columns numeric, train and test have the same features, and the target is 0/1.

## Key Findings

- `credit_history` is by far the strongest signal: about 80% of applicants with a good credit history are approved, against about 8% without. It has a correlation of about 0.54 with approval and accounts for roughly half of the total feature importance.
- The engineered `loan_income_ratio` ranked second in importance.
- Semi-urban properties and married applicants have somewhat higher approval rates. Gender and self-employment make little difference.

## Notes for Modelling

- The dataset is small (614 rows), so use a validation split of the training data to measure performance.
- The target is imbalanced (69% approved), so accuracy alone can mislead. Consider precision, recall and F1.
- Tree-based models do not need scaled data, but logistic regression and KNN do.

## How to Run

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Put `train_dataset.csv` and `test_dataset.csv` in an `Original_Data/` folder next to the notebook.
4. Open `ShammahNdubuisi_Week_2_Submission.ipynb` and run all cells. The notebook creates `Cleaned_Data/`, `ML_Ready_Data/` and `Visualizations/` automatically.

## Reports

- **Business Understanding Report:** the business problem, the key preprocessing and feature decisions, and what the data shows.
- **Data Preprocessing Report:** every cleaning, encoding, outlier, scaling and selection decision, with the reasoning behind it.
