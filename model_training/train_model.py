"""
Complaint classification - MODEL TRAINING ONLY
==============================================
Trains a text classifier (TF-IDF + Linear SVM) that predicts COMPLAINT_SUBTYPE
from COMPLAINT_DETAILS, compares it with two other models, and checks for overfitting.

Run:
    python train_model.py --data Consumer_Complain.xlsx

Output (created in ./output):
    complaint_model.joblib   the trained model
    metrics.json             all scores (model comparison, overfitting checks)

NOTE: do NOT commit the dataset to GitHub. It contains confidential consumer data.
"""
import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RARE_NAME = "Other (Rare Categories)"


# ---------------------------------------------------------------- 1. DATA
def clean_text(text) -> str:
    """Lowercase, remove numbers and punctuation. Hindi (Devanagari) letters are kept."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare(path, text_col, label_col, min_samples):
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().copy()
    df["text"] = df[text_col].map(clean_text)
    df = df[df["text"].str.len() >= 3]

    # 'Power in  meter' and 'power in meter' must count as ONE category
    df["key"] = df[label_col].map(lambda s: re.sub(r"\s+", " ", str(s).strip()).lower())
    nice_name = df.groupby("key")[label_col].agg(
        lambda s: Counter(re.sub(r"\s+", " ", str(x).strip()) for x in s).most_common(1)[0][0])
    df["label"] = df["key"].map(nice_name)

    # Many complaints are copy-pasted. If the same text sits in both the training and the
    # test set, the score looks better than it really is -> keep one copy of each text.
    n_before = len(df)
    conflicts = int((df.groupby("text")["label"].nunique() > 1).sum())
    df = df.groupby("text")["label"].agg(lambda s: s.value_counts().index[0]).reset_index()

    # A category with only a few examples cannot be learned -> merge into one group
    counts = df["label"].value_counts()
    rare = counts[counts < min_samples].index
    df.loc[df["label"].isin(rare), "label"] = RARE_NAME

    info = {"rows_after_cleaning": n_before, "unique_texts": len(df),
            "texts_with_conflicting_labels": conflicts,
            "categories_merged_as_rare": int(len(rare)), "final_categories": int(df["label"].nunique())}
    return df["text"].to_numpy(dtype=object), df["label"].to_numpy(dtype=object), info


# ---------------------------------------------------------------- 2. MODEL
def make_pipeline_for(classifier):
    """TF-IDF (word + character patterns) followed by a classifier."""
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),                       # words
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),   # spelling variants
    )
    return make_pipeline(features, classifier)


def candidate_models():
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=10),
        "Linear SVM": LinearSVC(class_weight="balanced", C=0.5),
    }


# ---------------------------------------------------------------- 3. MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Excel/CSV file with complaints")
    ap.add_argument("--text-col", default="COMPLAINT_DETAILS")
    ap.add_argument("--label-col", default="COMPLAINT_SUBTYPE")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--output-dir", default="output")
    a = ap.parse_args()

    X, y, info = load_and_prepare(a.data, a.text_col, a.label_col, a.min_samples)
    print("Data summary:", info)

    # The test set is locked away and used only for the final check.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Step A: compare models (5-fold cross-validation on the training part)
    print("\n[A] Model comparison")
    comparison = {}
    for name, clf in candidate_models().items():
        pipe = make_pipeline_for(clf)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=True)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        comparison[name] = {
            "train_accuracy": float(res["train_accuracy"].mean()),
            "cv_accuracy": float(res["test_accuracy"].mean()),
            "cv_macro_f1": float(res["test_f1_macro"].mean()),
            "cv_macro_f1_std": float(res["test_f1_macro"].std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro"))}
        c = comparison[name]
        print(f"  {name:25s} train acc {c['train_accuracy']:.2f} | CV acc {c['cv_accuracy']:.2f} | "
              f"CV macro-F1 {c['cv_macro_f1']:.2f} (+/-{c['cv_macro_f1_std']:.2f}) | test acc {c['test_accuracy']:.2f}")
    best = max(comparison, key=lambda k: comparison[k]["cv_macro_f1"])
    print("  -> selected:", best)

    # ---- Step B: overfitting check 1 - gap between training score and validation score
    gap = comparison[best]["train_accuracy"] - comparison[best]["cv_accuracy"]
    print(f"\n[B] Overfitting check: train accuracy {comparison[best]['train_accuracy']:.2f} vs "
          f"CV accuracy {comparison[best]['cv_accuracy']:.2f}  (gap = {gap:.2f})")

    # ---- Step C: overfitting check 2 - regularization strength C (smaller C = simpler model)
    print("\n[C] Effect of regularization (Linear SVM)")
    c_results = {}
    for C in [0.05, 0.1, 0.5, 1, 5]:
        res = cross_validate(make_pipeline_for(LinearSVC(class_weight="balanced", C=C)), X_train, y_train,
                             cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=True)
        c_results[str(C)] = {"train_acc": float(res["train_accuracy"].mean()),
                             "cv_acc": float(res["test_accuracy"].mean()),
                             "cv_macro_f1": float(res["test_f1_macro"].mean())}
        r = c_results[str(C)]
        print(f"  C={C:<5} train acc {r['train_acc']:.2f} | CV acc {r['cv_acc']:.2f} | CV macro-F1 {r['cv_macro_f1']:.2f}")

    # ---- Step D: overfitting check 3 - learning curve (does more data help?)
    print("\n[D] Learning curve (Linear SVM): validation score vs amount of training data")
    sizes, train_sc, val_sc = learning_curve(
        make_pipeline_for(candidate_models()["Linear SVM"]), X_train, y_train, cv=cv,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE)
    curve = []
    for n, t, v in zip(sizes, train_sc.mean(axis=1), val_sc.mean(axis=1)):
        curve.append({"training_examples": int(n), "train_macro_f1": float(t), "validation_macro_f1": float(v)})
        print(f"  {int(n):5d} examples | train macro-F1 {t:.2f} | validation macro-F1 {v:.2f}")

    # ---- Step E: final check on the locked test set
    final = make_pipeline_for(candidate_models()[best]).fit(X_train, y_train)
    pred = final.predict(X_test)
    print("\n[E] Final test-set report for", best)
    print(classification_report(y_test, pred, zero_division=0))

    # ---- Step F: save a model trained on ALL data, plus the scores
    out = Path(a.output_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(make_pipeline_for(candidate_models()[best]).fit(X, y), out / "complaint_model.joblib")
    (out / "metrics.json").write_text(json.dumps({
        "data": info, "kselected_model": best, "model_comparison": comparison,
        "train_vs_validation_gap": gap, "regularization_sweep": c_results, "learning_curve": curve,
        "test_report": classification_report(y_test, pred, output_dict=True, zero_division=0)}, indent=2))
    print(f"Saved {out / 'complaint_model.joblib'} and {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
