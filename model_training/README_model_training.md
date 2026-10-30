# Model training (complaint classification)

`train_model.py` trains a text classifier that predicts `COMPLAINT_SUBTYPE` from `COMPLAINT_DETAILS`.

## Run
```
pip install pandas numpy scikit-learn joblib openpyxl
python train_model.py --data Consumer_Complain.xlsx
```
The dataset is NOT stored in this repo (confidential). Put your copy next to the script.
Creates `output/complaint_model.joblib` and `output/metrics.json` (about 1 minute).

## What the script does
1. Clean text, merge label spelling variants, remove duplicate complaints, merge categories with <10 examples
2. Lock away 20% of the data as a test set
3. Compare Naive Bayes, Logistic Regression and Linear SVM with 5-fold cross-validation
4. Overfitting checks: train vs validation gap, regularization sweep (C), learning curve
5. Evaluate the selected model on the locked test set and save it

## Results (1,898 unique complaints, 23 categories)
| Model | Train acc | CV acc | CV macro-F1 | Test acc |
|---|---|---|---|---|
| Naive Bayes | 0.84 | 0.68 | 0.47 | 0.69 |
| Logistic Regression | 0.97 | 0.73 | 0.60 | 0.74 |
| **Linear SVM** | 0.96 | 0.74 | 0.61 | 0.75 |

## Overfitting
Train accuracy (0.96) is far above validation accuracy (0.74), so the model does overfit.
Stronger regularization (smaller C) did not improve validation scores, and the learning curve
flattens at about 0.60-0.61 macro-F1 from ~700 examples onward. The main limits are overlapping
categories and conflicting labels, not model size. Always report CV / test scores, never train scores.
