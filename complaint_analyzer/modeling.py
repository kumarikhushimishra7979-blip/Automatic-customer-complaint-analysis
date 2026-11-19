from pathlib import Path
from functools import lru_cache
import re

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

MODEL_VERSION = 4

RANDOM_STATE = 42

REVIEW_THRESHOLD = 0.40

MIN_EVAL_CLASS_SAMPLES = 5

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================================
# 2. LOAD SENTENCE TRANSFORMER
# ==========================================

@lru_cache(maxsize=1)
def _get_encoder():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ==========================================
# 3. TEXT NORMALIZATION
# ==========================================

def _normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().casefold()


# ==========================================
# 4. PREPARE TRAINING DATA
# ==========================================

def _prepare_training_data(texts, labels):

    if len(texts) != len(labels):

        raise ValueError(
            "Number of complaints and labels must match."
        )

    df = pd.DataFrame({

        "complaint": texts,
        "category": labels

    })

    df = df.dropna(
        subset=["complaint", "category"]
    )

    df["complaint"] = (
        df["complaint"].astype(str).str.strip()
    )

    df["category"] = (
        df["category"].astype(str).str.strip()
    )

    df = df[
        (df["complaint"] != "")
        & (df["category"] != "")
        & (~df["complaint"].str.lower().isin(
            ["nan", "none"]
        ))
    ].copy()

    # Normalize category capitalization and whitespace.

    df["category"] = df["category"].str.replace(
        r"\s+",
        " ",
        regex=True
    )

    category_mapping = {}

    for key, group in df.groupby(
        df["category"].str.casefold()
    ):

        category_mapping[key] = (
            group["category"].value_counts().index[0]
        )

    df["category"] = (
        df["category"].str.casefold().map(
            category_mapping
        )
    )

    # Create normalized complaint text for duplicate checking.

    df["normalized_text"] = (
        df["complaint"].apply(_normalize_text)
    )

    # Detect identical complaints with different labels.

    category_counts = (
        df.groupby("normalized_text")["category"]
        .nunique()
    )

    conflicting_texts = category_counts[
        category_counts > 1
    ].index

    conflict_count = len(conflicting_texts)

    df = df[
        ~df["normalized_text"].isin(conflicting_texts)
    ].copy()

    # Remove duplicate complaint-category pairs.

    before = len(df)

    df = df.drop_duplicates(
        subset=["normalized_text", "category"]
    )

    duplicate_count = before - len(df)

    if len(df) < 4:

        raise ValueError(
            "At least four valid labeled complaints are required."
        )

    if df["category"].nunique() < 2:

        raise ValueError(
            "At least two distinct categories are required."
        )

    category_distribution = (
        df["category"].value_counts().to_dict()
    )

    diagnostics = {

        "usable_rows": len(df),

        "duplicate_rows_removed": duplicate_count,

        "conflicting_complaints_removed": conflict_count,

        "total_categories": df["category"].nunique(),

        "category_distribution": category_distribution

    }

    return (

        df["complaint"].tolist(),

        df["category"].tolist(),

        diagnostics

    )


# ==========================================
# 5. CREATE CLASSIFIER
# ==========================================

def _create_classifier():

    return LogisticRegression(

        class_weight="balanced",

        random_state=RANDOM_STATE,

        max_iter=2000

    )


# ==========================================
# 6. TRAIN AND EVALUATE MODEL
# ==========================================

def train_and_evaluate(
    texts,
    labels,
    test_size=0.20
):

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1."
        )

    texts, labels, diagnostics = (
        _prepare_training_data(texts, labels)
    )

    encoder = _get_encoder()

    # Generate embeddings once.

    X_embeddings = encoder.encode(

        texts,

        show_progress_bar=True,

        batch_size=64,

        convert_to_numpy=True

    )

    counts = pd.Series(labels).value_counts()

    # Only categories with at least five examples
    # participate in the primary evaluation.

    eligible_categories = counts[
        counts >= MIN_EVAL_CLASS_SAMPLES
    ].index.tolist()

    eligible_indices = [

        index

        for index, label in enumerate(labels)

        if label in eligible_categories

    ]

    X_eval = X_embeddings[eligible_indices]

    y_eval = np.array([
        labels[index]
        for index in eligible_indices
    ])

    n_classes = len(eligible_categories)

    metrics = {

        "accuracy": None,

        "macro_f1": None,

        "weighted_f1": None,

        "report": {},

        "evaluation_note": "",

        "evaluated_categories": eligible_categories,

        "excluded_categories": counts[
            counts < MIN_EVAL_CLASS_SAMPLES
        ].to_dict(),

        "diagnostics": diagnostics

    }

    # ======================================
    # HOLDOUT EVALUATION
    # ======================================

    if n_classes >= 2:

        n_test = max(

            n_classes,

            int(np.ceil(
                len(y_eval) * test_size
            ))

        )

        n_train = len(y_eval) - n_test

        if n_test < len(y_eval) and n_train >= n_classes:

            X_train, X_test, y_train, y_test = (
                train_test_split(

                    X_eval,

                    y_eval,

                    test_size=n_test,

                    random_state=RANDOM_STATE,

                    stratify=y_eval

                )
            )

            evaluation_model = _create_classifier()

            evaluation_model.fit(
                X_train,
                y_train
            )

            predictions = evaluation_model.predict(
                X_test
            )

            metrics["accuracy"] = float(
                accuracy_score(
                    y_test,
                    predictions
                )
            )

            metrics["macro_f1"] = float(
                f1_score(

                    y_test,

                    predictions,

                    average="macro",

                    zero_division=0

                )
            )

            metrics["weighted_f1"] = float(
                f1_score(

                    y_test,

                    predictions,

                    average="weighted",

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

                f"Testing rows: {len(X_test)}. "

                f"Evaluated categories: {n_classes}. "

                f"Categories excluded from evaluation: "

                f"{len(counts) - n_classes}."

            )

        else:

            metrics["evaluation_note"] = (
                "Insufficient data for reliable stratified evaluation."
            )

    else:

        metrics["evaluation_note"] = (
            "At least two categories with five or more "
            "examples each are required for evaluation."
        )

    # ======================================
    # FINAL MODEL TRAINING
    # ======================================

    # IMPORTANT:
    # Train using ALL valid categories,
    # including categories with fewer than five records.

    final_model = _create_classifier()

    final_model.fit(

        X_embeddings,

        labels

    )

    return final_model, metrics


# ==========================================
# 7. SAVE TRAINED MODEL
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
# 8. LOAD SAVED MODEL
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
# 9. PREDICT NEW COMPLAINTS
# ==========================================

def predict_complaints(model, texts):

    if model is None:

        raise ValueError(
            "A trained model is required."
        )

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
        and text.strip().lower() not in {"nan", "none"}

    ]

    if valid_indices:

        valid_texts = [

            clean_texts[index]

            for index in valid_indices

        ]

        encoder = _get_encoder()

        X_valid_embeddings = encoder.encode(

            valid_texts,

            show_progress_bar=False,

            batch_size=64,

            convert_to_numpy=True

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
# 10. DISCOVER POTENTIAL NEW COMPLAINT GROUPS
# ==========================================

def discover_candidate_groups(
    texts,
    statuses,
    min_cluster_size=3
):

    if len(texts) != len(statuses):

        raise ValueError(
            "Number of complaints and statuses must match."
        )

    if min_cluster_size < 2:

        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    # Maintain one output for every input row.

    candidate_groups = [""] * len(texts)

    review_indices = []

    review_texts = []

    for original_index, (text, status) in enumerate(
        zip(texts, statuses)
    ):

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

    if len(review_texts) < min_cluster_size * 2:

        return candidate_groups

    encoder = _get_encoder()

    embeddings = encoder.encode(

        review_texts,

        show_progress_bar=False,

        batch_size=64,

        convert_to_numpy=True

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

    cluster_labels = clustering.fit_predict(
        embeddings
    )

    for cluster_id in range(n_clusters):

        cluster_positions = np.where(
            cluster_labels == cluster_id
        )[0]

        cluster_size = len(cluster_positions)

        if cluster_size >= min_cluster_size:

            group_name = (

                f"Potential new complaint group "

                f"{cluster_id + 1} "

                f"({cluster_size} complaints)"

            )

            for review_position in cluster_positions:

                original_index = review_indices[
                    review_position
                ]

                candidate_groups[original_index] = group_name

    return candidate_groups