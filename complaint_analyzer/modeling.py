from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.cluster import KMeans


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

MODEL_VERSION = 3
RANDOM_STATE = 42

REVIEW_THRESHOLD = 0.40

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================================
# 2. LOAD SENTENCE TRANSFORMER
# ==========================================

@lru_cache(maxsize=1)
def _get_encoder():
    """
    Load and cache the pretrained Sentence Transformer.

    Caching prevents the model from loading repeatedly
    during application execution.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==========================================
# 3. PREPARE TRAINING DATA
# ==========================================

def _prepare_training_data(texts, labels):
    """
    Clean complaint text and category labels.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "Number of complaints and labels must match."
        )

    df = pd.DataFrame({
        "complaint": texts,
        "category": labels
    })

    # Remove missing values
    df = df.dropna(
        subset=["complaint", "category"]
    )

    # Convert values into clean strings
    df["complaint"] = (
        df["complaint"].astype(str).str.strip()
    )

    df["category"] = (
        df["category"].astype(str).str.strip()
    )

    # Remove empty complaints and labels
    df = df[
        (df["complaint"] != "")
        & (df["category"] != "")
    ]

    # Identify complaints having multiple category labels
    category_counts = (
        df.groupby("complaint")["category"].nunique()
    )

    conflicting_texts = category_counts[
        category_counts > 1
    ].index

    conflict_count = len(conflicting_texts)

    # Remove conflicting complaints
    df = df[
        ~df["complaint"].isin(conflicting_texts)
    ]

    # Remove duplicate complaint-category pairs
    before = len(df)

    df = df.drop_duplicates(
        subset=["complaint", "category"]
    )

    duplicate_count = before - len(df)

    # Validate the remaining dataset
    if len(df) < 4:
        raise ValueError(
            "At least four valid labeled examples are required."
        )

    if df["category"].nunique() < 2:
        raise ValueError(
            "At least two distinct categories are required."
        )

    # Store data-quality information
    diagnostics = {
        "usable_rows": len(df),
        "duplicate_rows_removed": duplicate_count,
        "conflicting_complaints_removed": conflict_count,
        "category_distribution": (
            df["category"].value_counts().to_dict()
        )
    }

    return (
        df["complaint"].tolist(),
        df["category"].tolist(),
        diagnostics
    )


# ==========================================
# 4. TRAIN AND EVALUATE MODEL
# ==========================================

def train_and_evaluate(texts, labels, test_size=0.2):
    """
    Train a Logistic Regression classifier using
    Sentence Transformer embeddings.
    """
    texts, labels, diagnostics = (
        _prepare_training_data(texts, labels)
    )

    encoder = _get_encoder()

    X_embeddings = encoder.encode(
        texts,
        show_progress_bar=False
    )

    counts = pd.Series(labels).value_counts()

    n_classes = len(counts)
    total_samples = len(texts)

    metrics = {
        "accuracy": None,
        "macro_f1": None,
        "report": {},
        "evaluation_note": "",
        "diagnostics": diagnostics
    }

    n_test = max(
        n_classes,
        int(np.ceil(total_samples * test_size))
    )

    n_train = total_samples - n_test

    can_stratify = (
        counts.min() >= 2
        and n_test < total_samples
        and n_test >= n_classes
        and n_train >= n_classes
    )

    if can_stratify:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X_embeddings,
                labels,
                test_size=n_test,  # Fixed to use integer n_test
                random_state=RANDOM_STATE,
                stratify=labels
            )
        )

        evaluation_model = LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000
        )

        evaluation_model.fit(
            X_train,
            y_train
        )

        predictions = evaluation_model.predict(
            X_test
        )

        metrics["accuracy"] = float(
            accuracy_score(y_test, predictions)
        )

        metrics["macro_f1"] = float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        metrics["report"] = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        metrics["evaluation_note"] = (
            f"Holdout evaluation completed. "
            f"Training rows: {len(X_train)}. "
            f"Testing rows: {len(X_test)}."
        )

    else:
        metrics["evaluation_note"] = (
            "A reliable stratified holdout evaluation "
            "could not be created due to small class sizes."
        )

    final_model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    final_model.fit(
        X_embeddings,
        labels
    )

    return final_model, metrics


# ==========================================
# 5. SAVE TRAINED MODEL
# ==========================================

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        "version": MODEL_VERSION,
        "model_name": EMBEDDING_MODEL_NAME,
        "model": model
    }

    joblib.dump(
        payload,
        path
    )


# ==========================================
# 6. LOAD SAVED MODEL
# ==========================================

def load_model(path):
    path = Path(path)

    if not path.exists():
        return None

    payload = joblib.load(path)

    if isinstance(payload, dict):
        return payload.get("model")

    return payload


# ==========================================
# 7. PREDICT NEW COMPLAINTS
# ==========================================

def predict_complaints(model, texts):
    clean_texts = [
        str(text) if text is not None else ""
        for text in texts
    ]

    predictions = [
        "Empty complaint text"
        for _ in clean_texts
    ]

    confidence = [
        0.0 for _ in clean_texts
    ]

    status = [
        "empty_text" for _ in clean_texts
    ]

    valid_indices = [
        index
        for index, text in enumerate(clean_texts)
        if text.strip()
    ]

    if valid_indices:
        valid_texts = [
            clean_texts[index]
            for index in valid_indices
        ]

        encoder = _get_encoder()

        X_valid_embeddings = encoder.encode(
            valid_texts,
            show_progress_bar=False
        )

        predicted_labels = model.predict(
            X_valid_embeddings
        )

        probabilities = model.predict_proba(
            X_valid_embeddings
        )

        max_probs = np.max(
            probabilities,
            axis=1
        )

        for position, original_index in enumerate(
            valid_indices
        ):
            predictions[original_index] = str(
                predicted_labels[position]
            )

            confidence[original_index] = float(
                max_probs[position]
            )

            status[original_index] = (
                "classified"
                if max_probs[position] >= REVIEW_THRESHOLD
                else "review"
            )

    return pd.DataFrame({
        "predicted_category": predictions,
        "confidence": np.round(confidence, 4),
        "status": status
    })


# ==========================================
# 8. DISCOVER POTENTIAL NEW COMPLAINT GROUPS
# ==========================================

def discover_candidate_groups(
    texts,
    statuses,
    min_cluster_size=3
):
    """
    Discover potential new complaint groups among complaints
    marked for review.

    Returns exactly one entry per input complaint. Complaints
    not assigned to a sufficiently large cluster receive "".
    """

    if len(texts) != len(statuses):
        raise ValueError(
            "Number of complaints and statuses must match."
        )

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    # Preserve alignment with every original complaint row.
    candidate_groups = [""] * len(texts)

    review_indices = []
    review_texts = []

    for original_index, (text, status) in enumerate(zip(texts, statuses)):
        if text is None:
            continue

        text_value = str(text).strip()
        status_value = str(status).strip().lower()

        if (
            text_value
            and text_value.lower() not in {"nan", "none"}
            and status_value == "review"
        ):
            review_indices.append(original_index)
            review_texts.append(text_value)

    # Even when clustering is skipped, return a list aligned
    # with the complete input dataset.
    if len(review_texts) < min_cluster_size * 2:
        return candidate_groups

    encoder = _get_encoder()
    embeddings = encoder.encode(
        review_texts,
        show_progress_bar=False
    )

    n_clusters = max(
        2,
        min(
            int(np.sqrt(len(review_texts) / 2)),
            len(review_texts) // min_cluster_size
        )
    )

    if n_clusters >= len(review_texts):
        return candidate_groups

    clustering = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cluster_labels = clustering.fit_predict(embeddings)

    # Map each cluster member back to its original complaint row.
    for cluster_id in range(n_clusters):
        cluster_positions = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_positions)

        if cluster_size >= min_cluster_size:
            group_name = (
                f"Potential new complaint group "
                f"{cluster_id + 1} "
                f"({cluster_size} complaints)"
            )

            for review_position in cluster_positions:
                original_index = review_indices[review_position]
                candidate_groups[original_index] = group_name

    return candidate_groups
