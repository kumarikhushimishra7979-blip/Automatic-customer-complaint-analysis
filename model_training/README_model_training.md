# Model training (complaint classification)

`train_model.py` trains a text classifier that predicts `COMPLAINT_SUBTYPE` from `COMPLAINT_DETAILS`
and runs the evaluation checks required by the project plan.

## Run
```
pip install pandas numpy scikit-learn joblib openpyxl matplotlib
python train_model.py --data Consumer_Complain.xlsx
```
Takes about 1 minute. Add `--skip-self-training` for a faster run. `matplotlib` is optional (only for the PNG picture).

**Do not commit the dataset or the `output/` folder** - they contain confidential consumer text.
Suggested `.gitignore`: `*.xlsx`, `*.csv`, `output/`, `venv/`, `__pycache__/`

## Steps in the script -> project plan
| Step | What it does | Plan section |
|---|---|---|
| A | Compares Naive Bayes, Logistic Regression, Linear SVM (5-fold CV, same split) | 4.2 - 4.5 |
| B, C, D | Overfitting checks: train-vs-validation gap, regularization sweep, learning curve | 7.2 |
| E | Test-set report, confusion matrix (CSV + PNG), most confused pairs, weak categories | 7.1, 7.3 |
| F | Near-duplicate check: accuracy on genuinely new wording | 3.4 |
| G | Review rule: how many complaints can be accepted automatically, at what accuracy | 6.3 |
| H | Label review files: conflicting labels and labels the model strongly disagrees with | 3.4, 7.2 |
| I | Self-training experiment (simulated: 30% of labels kept) | 5 |

## Results (1,898 unique complaints, 23 categories)
| Model | Train acc | CV acc | CV macro-F1 | Test acc |
|---|---|---|---|---|
| Naive Bayes | 0.84 | 0.68 | 0.47 | 0.69 |
| Logistic Regression | 0.97 | 0.73 | 0.60 | 0.74 |
| **Linear SVM** | 0.96 | 0.74 | 0.61 | 0.75 |

* Overfitting: train 0.96 vs validation 0.74. Smaller C did not help; the learning curve flattens at ~0.61.
* Near-duplicates: 23% of test complaints have a near-copy in training. Accuracy is 0.89 on those but
  **0.71 on genuinely new wording** - the more honest number.
* Review rule: 64% of test complaints accepted automatically at 89% accuracy; the other 36% go to review.
* Self-training: no clear gain (macro-F1 0.545 -> 0.551; 83% of pseudo-labels correct), so the supervised model is kept.
* Weak categories: "Power in meter but site disconnected" (F1 0.29), "BALANCE POSITIVE BUT POWER not
  available in meter" (0.49) and three tiny categories. They overlap in meaning with other categories.
* 32 identical texts carry conflicting labels (see `output/label_conflicts.csv`).

## Not covered here (needs more data or other parts of the project)
New-category discovery (K-Means / DBSCAN), the Streamlit app, Excel/PDF reports, Windows packaging,
an independently verified test set.
