from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans

MODEL_VERSION = 1

def _build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1, max_features=100_000, sublinear_tf=True)),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42))
    ])

def train_and_evaluate(texts, labels, test_size=0.2):
    pairs = [(str(t).strip(), str(y).strip()) for t, y in zip(texts, labels) if str(t).strip() and str(y).strip()]
    texts, labels = [p[0] for p in pairs], [p[1] for p in pairs]
    if len(texts) < 4: raise ValueError("Please provide at least 4 non-empty labeled examples; more is strongly recommended.")
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2: raise ValueError("Training requires at least two distinct categories.")
    metrics = {"accuracy": None, "macro_f1": None, "report": {}, "evaluation_note": ""}
    n_classes = len(counts)
    n_test = max(n_classes, int(round(len(texts) * float(test_size))))
    can_stratify = counts.min() >= 2 and n_test >= n_classes and (len(texts) - n_test) >= n_classes
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42, stratify=labels)
        evaluation_model = _build_model()
        evaluation_model.fit(X_train, y_train)
        pred = evaluation_model.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, pred))
        metrics["macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
        metrics["report"] = classification_report(y_test, pred, output_dict=True, zero_division=0)
        metrics["evaluation_note"] = f"Stratified holdout evaluation: {len(X_train)} training rows and {len(X_test)} test rows. The final model is retrained on all rows after evaluation."
    else:
        metrics["evaluation_note"] = "No reliable stratified holdout could be created because the data is small or one or more categories have too few examples. A model was trained on all rows, but no holdout score is reported. Add verified examples per category before judging performance."
    final_model = _build_model()
    final_model.fit(texts, labels)
    return final_model, metrics

def save_model(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": MODEL_VERSION, "model": model}, path)

def load_model(path):
    path = Path(path)
    if not path.exists(): return None
    payload = joblib.load(path)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload

def predict_complaints(model, texts):
    clean = [str(t) if t is not None else "" for t in texts]
    predictions = ["Empty complaint text" if not t.strip() else "" for t in clean]
    confidence = [0.0] * len(clean)
    status = ["empty_text" if not t.strip() else "" for t in clean]
    indices = [i for i, t in enumerate(clean) if t.strip()]
    if indices:
        valid = [clean[i] for i in indices]
        pred = model.predict(valid)
        decision = model.decision_function(valid)
        if np.ndim(decision) == 1:
            margins = np.abs(decision)
        else:
            sorted_scores = np.sort(decision, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        # Review indicator only; not a calibrated probability.
        normalized = margins / (1.0 + np.abs(margins))
        for j, i in enumerate(indices):
            predictions[i] = str(pred[j])
            confidence[i] = float(normalized[j])
            status[i] = "classified" if confidence[i] >= 0.20 else "review"
    return pd.DataFrame({"predicted_category": predictions, "confidence": np.round(confidence, 4), "status": status})

def discover_candidate_groups(texts, statuses, min_cluster_size=3):
    groups = [""] * len(texts)
    indices = [i for i, (t, s) in enumerate(zip(texts, statuses)) if str(t).strip() and s == "review"]
    if len(indices) < min_cluster_size * 2: return groups
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000)
        matrix = vectorizer.fit_transform([str(texts[i]) for i in indices])
        if matrix.shape[1] < 2: return groups
        n_clusters = min(max(2, len(indices) // min_cluster_size), 8, len(indices) - 1)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix)
        counts = pd.Series(labels).value_counts()
        names = {}
        next_id = 1
        for label, count in counts.items():
            if count >= min_cluster_size:
                names[int(label)] = f"Candidate group {next_id}"; next_id += 1
        for local_i, original_i in enumerate(indices):
            if int(labels[local_i]) in names: groups[original_i] = names[int(labels[local_i])]
    except Exception:
        pass  # Discovery is optional; classification should still complete.
    return groups
