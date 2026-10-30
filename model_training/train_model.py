"""
Complaint classification - MODEL TRAINING (version 2)
=====================================================
Trains a text classifier (TF-IDF + Linear SVM) that predicts COMPLAINT_SUBTYPE from
COMPLAINT_DETAILS, compares it with two other models and runs the checks the project plan asks for.

Run:
    python train_model.py --data Consumer_Complain.xlsx

Everything is written to ./output :
    complaint_model.joblib     trained model
    metrics.json               all scores and thresholds
    confusion_matrix.csv/.png  which categories get confused (plan 7.1)
    label_conflicts.csv        same text, different labels            (plan 3.4 / 7.2)
    label_review.csv           labels the model strongly disagrees with (plan 7.2)

NOTE: output/ and the dataset contain confidential consumer text. NEVER commit them to GitHub.

Steps
  A  compare 3 models (plan 4.2-4.5)        F  near-duplicate leakage check (plan 3.4)
  B  overfitting gap                         G  review rule: coverage vs accuracy (plan 6.3)
  C  regularization sweep                    H  label-quality review files (plan 7.2)
  D  learning curve                          I  self-training experiment (plan 5)
  E  final test + confusion matrix (plan 7.1/7.3)    J  save everything
"""
import argparse
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RARE_NAME = "Other (Rare Categories)"
NEAR_DUPLICATE_SIMILARITY = 0.80     # texts at least this similar are treated as near-duplicates
TARGET_ACCEPTED_ACCURACY = 0.90      # the review rule should be this accurate on accepted complaints
WEAK_F1 = 0.50                       # categories below this F1 need attention


# ============================================================ 1. DATA
def clean_text(text) -> str:
    """Lowercase, remove numbers and punctuation. Hindi (Devanagari) letters are kept."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare(path, text_col, label_col, min_samples):
    """Returns a table with columns: original, text, label  (+ a summary and the conflict table)."""
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().copy()
    df["original"] = df[text_col].astype(str)
    df["text"] = df[text_col].map(clean_text)
    df = df[df["text"].str.len() >= 3]

    # 'Power in  meter' and 'power in meter' must count as ONE category
    df["key"] = df[label_col].map(lambda s: re.sub(r"\s+", " ", str(s).strip()).lower())
    nice_name = df.groupby("key")[label_col].agg(
        lambda s: Counter(re.sub(r"\s+", " ", str(x).strip()) for x in s).most_common(1)[0][0])
    df["label"] = df["key"].map(nice_name)
    n_before = len(df)

    # Same text with different labels = label noise. Save these for review.
    per_text = df.groupby("text")["label"].agg(lambda s: sorted(set(s)))
    conflicts = per_text[per_text.str.len() > 1].reset_index()
    conflicts["labels"] = conflicts["label"].map(" | ".join)
    conflicts = conflicts[["text", "labels"]]

    # Keep ONE copy of each text (its most frequent label) so the same text can't be in train and test
    first_original = df.groupby("text")["original"].first()
    top_label = df.groupby("text")["label"].agg(lambda s: s.value_counts().index[0])
    df = pd.DataFrame({"original": first_original, "label": top_label}).reset_index()

    # Categories with only a few examples cannot be learned -> merge them
    counts = df["label"].value_counts()
    rare = counts[counts < min_samples].index
    df.loc[df["label"].isin(rare), "label"] = RARE_NAME

    info = {"rows_after_cleaning": n_before, "unique_texts": len(df),
            "texts_with_conflicting_labels": int(len(conflicts)),
            "categories_merged_as_rare": int(len(rare)), "final_categories": int(df["label"].nunique())}
    return df, info, conflicts


# ============================================================ 2. MODELS
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


# ============================================================ 3. HELPERS
def get_scores(pipe, texts):
    """One score per category for every text (higher = more likely)."""
    clf = pipe[-1]
    X = pipe[:-1].transform(texts)
    return clf.decision_function(X) if hasattr(clf, "decision_function") else clf.predict_log_proba(X)


def confidence_margin(scores):
    """Best score minus second-best score. A big margin = the model is sure."""
    top_two = np.sort(scores, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def cv_predict(make_model, X, y, n_splits=5):
    """Prediction + confidence for every item, always made by a model that did NOT see that item."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = np.empty(len(X), dtype=object)
    margin = np.zeros(len(X))
    for train_idx, val_idx in skf.split(X, y):
        pipe = make_model().fit(X[train_idx], y[train_idx])
        S = get_scores(pipe, X[val_idx])
        pred[val_idx] = pipe[-1].classes_[S.argmax(axis=1)]
        margin[val_idx] = confidence_margin(S)
    return pred, margin


def pick_margin_threshold(margin, correct, target=TARGET_ACCEPTED_ACCURACY):
    """Smallest margin so that complaints at or above it are right at least `target` of the time."""
    for t in np.unique(np.quantile(margin, np.linspace(0, 0.95, 96))):
        keep = margin >= t
        if keep.sum() >= 20 and correct[keep].mean() >= target:
            return float(t)
    return float(np.quantile(margin, 0.5))


def save_confusion_png(cm, labels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping the PNG picture; the CSV is still saved)")
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    short = [l[:34] for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), short, fontsize=7)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("Confusion matrix (each row sums to 100%: how the actual complaints were classified)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================ 4. MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Excel/CSV file with complaints")
    ap.add_argument("--text-col", default="COMPLAINT_DETAILS")
    ap.add_argument("--label-col", default="COMPLAINT_SUBTYPE")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--skip-self-training", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(exist_ok=True)

    df, info, conflicts = load_and_prepare(a.data, a.text_col, a.label_col, a.min_samples)
    print("Data summary:", info)
    X = df["text"].to_numpy(dtype=object)
    y = df["label"].to_numpy(dtype=object)

    # The test set is locked away and used only for the final checks.
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------- A. compare models
    print("\n[A] Model comparison (5-fold cross-validation on the training part)")
    comparison = {}
    for name, clf in candidate_models().items():
        pipe = make_pipeline_for(clf)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=True)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        pred = pipe.predict(X_test)
        per_class_f1 = f1_score(y_test, pred, average=None, labels=sorted(set(y_test)), zero_division=0)
        comparison[name] = {
            "train_accuracy": float(res["train_accuracy"].mean()),
            "cv_accuracy": float(res["test_accuracy"].mean()),
            "cv_macro_f1": float(res["test_f1_macro"].mean()),
            "cv_macro_f1_std": float(res["test_f1_macro"].std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
            "categories_with_f1_below_0.5": int((per_class_f1 < WEAK_F1).sum()),
            "fit_seconds": round(fit_seconds, 1)}
        c = comparison[name]
        print(f"  {name:25s} train acc {c['train_accuracy']:.2f} | CV acc {c['cv_accuracy']:.2f} | "
              f"CV macro-F1 {c['cv_macro_f1']:.2f} (+/-{c['cv_macro_f1_std']:.2f}) | test acc {c['test_accuracy']:.2f} | "
              f"weak categories {c['categories_with_f1_below_0.5']} | fit {c['fit_seconds']}s")
    best = max(comparison, key=lambda k: comparison[k]["cv_macro_f1"])
    print("  -> selected:", best)
    make_best = lambda: make_pipeline_for(candidate_models()[best])  # noqa: E731

    # ---------- B. overfitting gap
    gap = comparison[best]["train_accuracy"] - comparison[best]["cv_accuracy"]
    print(f"\n[B] Overfitting gap: train accuracy {comparison[best]['train_accuracy']:.2f} vs "
          f"CV accuracy {comparison[best]['cv_accuracy']:.2f}  (gap = {gap:.2f})")

    # ---------- C. regularization sweep
    print("\n[C] Regularization sweep (Linear SVM): smaller C = simpler model")
    c_results = {}
    for C in [0.05, 0.1, 0.5, 1, 5]:
        res = cross_validate(make_pipeline_for(LinearSVC(class_weight="balanced", C=C)), X_train, y_train, cv=cv,
                             scoring=["accuracy", "f1_macro"], return_train_score=True)
        c_results[str(C)] = {"train_acc": float(res["train_accuracy"].mean()), "cv_acc": float(res["test_accuracy"].mean()),
                             "cv_macro_f1": float(res["test_f1_macro"].mean())}
        r = c_results[str(C)]
        print(f"  C={C:<5} train acc {r['train_acc']:.2f} | CV acc {r['cv_acc']:.2f} | CV macro-F1 {r['cv_macro_f1']:.2f}")

    # ---------- D. learning curve
    print("\n[D] Learning curve: does more data help?")
    sizes, train_sc, val_sc = learning_curve(make_best(), X_train, y_train, cv=cv, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
                                             scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE)
    curve = []
    for n, t, v in zip(sizes, train_sc.mean(axis=1), val_sc.mean(axis=1)):
        curve.append({"training_examples": int(n), "train_macro_f1": float(t), "validation_macro_f1": float(v)})
        print(f"  {int(n):5d} examples | train macro-F1 {t:.2f} | validation macro-F1 {v:.2f}")

    # ---------- E. final test report + confusion matrix
    final = make_best().fit(X_train, y_train)
    pred_test = final.predict(X_test)
    report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    print(f"\n[E] Final test-set result for {best}: accuracy {accuracy_score(y_test, pred_test):.3f}, "
          f"macro-F1 {f1_score(y_test, pred_test, average='macro'):.3f}")
    labels = sorted(set(y_test) | set(pred_test))
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    pd.DataFrame(cm, index=[f"actual: {l}" for l in labels], columns=[f"pred: {l}" for l in labels]).to_csv(out / "confusion_matrix.csv")
    save_confusion_png(cm, labels, out / "confusion_matrix.png")
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda p: -p[2])
    print("  Most confused pairs (actual -> predicted : count):")
    for act, prd, n in pairs[:6]:
        print(f"    {act[:45]:45s} -> {prd[:45]:45s} : {n}")
    weak = sorted([(report[l]["f1-score"], l, int(report[l]["support"])) for l in labels if l in report and report[l]["f1-score"] < WEAK_F1])
    print(f"  Weak categories (F1 < {WEAK_F1}) that need attention:")
    for f1v, l, n in weak:
        print(f"    F1 {f1v:.2f} (only {n} test complaints)  {l}")

    # ---------- F. near-duplicate check
    print(f"\n[F] Near-duplicate check (similarity >= {NEAR_DUPLICATE_SIMILARITY})")
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(X_train)
    sim = np.asarray((vec.transform(X_test) @ vec.transform(X_train).T).max(axis=1).todense()).ravel()
    near = sim >= NEAR_DUPLICATE_SIMILARITY
    correct_test = pred_test == y_test
    acc_near = float(correct_test[near].mean()) if near.any() else float("nan")
    acc_new = float(correct_test[~near].mean()) if (~near).any() else float("nan")
    near_dup = {"share_of_test_with_near_duplicate_in_train": float(near.mean()),
                "accuracy_on_those": acc_near, "accuracy_on_genuinely_new_wording": acc_new}
    print(f"  {near.mean():.0%} of test complaints have a near-copy in the training data.")
    print(f"  Accuracy on those: {acc_near:.2f}   |   on genuinely new wording: {acc_new:.2f}  <- the stricter, more honest number")

    # ---------- G. review rule (coverage vs accuracy)
    print("\n[G] Review rule: accept only confident predictions")
    cv_pred, cv_margin = cv_predict(make_best, X_train, y_train)
    threshold = pick_margin_threshold(cv_margin, cv_pred == y_train)
    S_test = get_scores(final, X_test)
    m_test = confidence_margin(S_test)
    accepted = m_test >= threshold
    review_rule = {"margin_threshold": threshold, "target_accuracy": TARGET_ACCEPTED_ACCURACY,
                   "test_share_accepted": float(accepted.mean()),
                   "test_accuracy_of_accepted": float(correct_test[accepted].mean()) if accepted.any() else None,
                   "test_accuracy_of_review_group": float(correct_test[~accepted].mean()) if (~accepted).any() else None}
    print(f"  threshold {threshold:.2f} (chosen on training data only)")
    print(f"  On the test set: {accepted.mean():.0%} accepted automatically at {review_rule['test_accuracy_of_accepted']:.0%} accuracy; "
          f"the other {(~accepted).mean():.0%} go to human review (their accuracy would be only {review_rule['test_accuracy_of_review_group']:.0%})")

    # ---------- H. label-quality review files
    print("\n[H] Label-quality review (for sir / the data owner)")
    all_pred, all_margin = cv_predict(make_best, X, y)
    suspects = df.assign(model_prediction=all_pred, confidence=all_margin.round(2))
    suspects = suspects[(suspects["model_prediction"] != suspects["label"]) & (suspects["confidence"] >= threshold)]
    suspects = suspects.sort_values("confidence", ascending=False)[["original", "label", "model_prediction", "confidence"]]
    suspects.rename(columns={"label": "given_label"}).head(150).to_csv(out / "label_review.csv", index=False)
    conflicts.to_csv(out / "label_conflicts.csv", index=False)
    print(f"  {len(conflicts)} texts appear with different labels -> output/label_conflicts.csv")
    print(f"  {len(suspects)} complaints where the model confidently disagrees with the given label -> output/label_review.csv (top 150)")

    # ---------- I. self-training experiment (simulated: we hide most labels)
    self_training = None
    if not a.skip_self_training:
        print("\n[I] Self-training experiment (plan section 5): keep only 30% of the training labels, treat the rest as unlabeled")
        lab, unl = train_test_split(np.arange(len(X_train)), train_size=0.3, stratify=y_train, random_state=1)
        base = make_best().fit(X_train[lab], y_train[lab])
        base_pred = base.predict(X_test)
        p_cv, m_cv = cv_predict(make_best, X_train[lab], y_train[lab], n_splits=3)
        thr = pick_margin_threshold(m_cv, p_cv == y_train[lab])
        cur_X, cur_y, pool = list(X_train[lab]), list(y_train[lab]), list(unl)
        added = added_ok = 0
        model = base
        for _ in range(4):
            if not pool:
                break
            S = get_scores(model, X_train[pool])
            guess = model[-1].classes_[S.argmax(axis=1)]
            take = set(np.flatnonzero(confidence_margin(S) >= thr).tolist())
            if not take:
                break
            for k in take:
                cur_X.append(X_train[pool[k]]); cur_y.append(guess[k])
                added += 1; added_ok += int(guess[k] == y_train[pool[k]])
            pool = [p for k, p in enumerate(pool) if k not in take]
            model = make_best().fit(np.array(cur_X, dtype=object), np.array(cur_y, dtype=object))
        st_pred = model.predict(X_test)
        full_pred = final.predict(X_test)
        self_training = {
            "baseline_30pct_labels": {"accuracy": float(accuracy_score(y_test, base_pred)), "macro_f1": float(f1_score(y_test, base_pred, average="macro"))},
            "after_self_training": {"accuracy": float(accuracy_score(y_test, st_pred)), "macro_f1": float(f1_score(y_test, st_pred, average="macro")),
                                    "pseudo_labels_added": added, "pseudo_label_accuracy": (added_ok / added) if added else None},
            "reference_all_labels": {"accuracy": float(accuracy_score(y_test, full_pred)), "macro_f1": float(f1_score(y_test, full_pred, average="macro"))}}
        s = self_training
        print(f"  baseline (30% labels):  accuracy {s['baseline_30pct_labels']['accuracy']:.3f}  macro-F1 {s['baseline_30pct_labels']['macro_f1']:.3f}")
        print(f"  after self-training:    accuracy {s['after_self_training']['accuracy']:.3f}  macro-F1 {s['after_self_training']['macro_f1']:.3f}  "
              f"({added} pseudo-labels, {s['after_self_training']['pseudo_label_accuracy']:.0%} correct)")
        print(f"  reference (all labels): accuracy {s['reference_all_labels']['accuracy']:.3f}  macro-F1 {s['reference_all_labels']['macro_f1']:.3f}")
        gain = s["after_self_training"]["macro_f1"] - s["baseline_30pct_labels"]["macro_f1"]
        print("  Decision:", "keep self-training (clear gain)" if gain >= 0.02 else "no clear gain -> keep the normal supervised model (as the plan allows)")

    # ---------- J. save
    joblib.dump(make_best().fit(X, y), out / "complaint_model.joblib")
    (out / "metrics.json").write_text(json.dumps({
        "data": info, "selected_model": best, "model_comparison": comparison, "train_vs_validation_gap": gap,
        "regularization_sweep": c_results, "learning_curve": curve, "near_duplicate_check": near_dup,
        "review_rule": review_rule, "most_confused_pairs": [{"actual": p[0], "predicted": p[1], "count": p[2]} for p in pairs[:10]],
        "weak_categories": [{"category": l, "f1": f1v, "test_support": n} for f1v, l, n in weak],
        "self_training_experiment": self_training, "test_report": report}, indent=2, default=float))
    print(f"\n[J] Saved model and results in {out}/")


if __name__ == "__main__":
    main()
