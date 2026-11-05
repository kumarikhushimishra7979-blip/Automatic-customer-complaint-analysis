from pathlib import Path
import sys
import re

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from complaint_analyzer.data_utils import (
    read_table,
    normalize_training_frame,
    make_summary,
)
from complaint_analyzer.modeling import (
    train_and_evaluate,
    load_model,
    save_model,
    predict_complaints,
    discover_candidate_groups,
)
from complaint_analyzer.reporting import build_excel, build_pdf


st.set_page_config(
    page_title="Customer Complaint Analyzer",
    page_icon="📊",
    layout="wide",
)

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "data" / "training").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

MODEL_PATH = ROOT / "models" / "complaint_model.joblib"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def read_uploaded_file(uploaded_file):
    """Read an uploaded file safely, resetting its pointer before reading."""
    uploaded_file.seek(0)
    dataframe = read_table(uploaded_file)
    uploaded_file.seek(0)
    return dataframe


def preferred_column_index(columns, preferred_names):
    """Return the index of the first preferred column that exists."""
    normalized = {str(column).strip().casefold(): i for i, column in enumerate(columns)}
    for name in preferred_names:
        if name.casefold() in normalized:
            return normalized[name.casefold()]
    return 0


def clean_file_key(filename, index):
    """Create a predictable Streamlit widget key for each uploaded file."""
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(filename))
    return f"{index}_{safe_name}"


def make_bar_chart(summary):
    """Create a descending horizontal complaint-count chart."""
    if summary is None or summary.empty:
        return None

    plot_df = summary.sort_values("count", ascending=True).copy()
    fig_height = max(3.5, min(11, len(plot_df) * 0.38))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    bars = ax.barh(plot_df["category"].astype(str), plot_df["count"])
    ax.set_xlabel("Number of complaints")
    ax.set_ylabel("Issue Type")
    ax.set_title("Complaints by Issue Type")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    max_count = max(1, int(plot_df["count"].max()))
    ax.set_xlim(0, max_count * 1.18)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        count = int(row["count"])
        percentage = float(row["percentage"])
        ax.text(
            count + max_count * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({percentage:.1f}%)",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Application header and privacy information
# ---------------------------------------------------------------------
st.title("📊 Customer Complaint Analyzer")
st.caption(
    "Local-first complaint classification, category summaries, "
    "potential-group discovery, and downloadable reports."
)

with st.sidebar:
    st.header("Project status")
    st.write("**Processing:** on this computer")
    st.write("**External AI API:** not used")
    st.write("**Classifier:** Sentence Transformers + Logistic Regression")
    st.info(
        "Complaint files and trained models are processed locally by this app. "
        "Keep the computer protected and do not expose the app to an untrusted "
        "network. Local processing alone does not guarantee data security."
    )
    st.markdown("---")
    st.markdown("**Recommended source columns**")
    st.code("Complaint text: COMPLAINT_DETAILS\nCategory: COMPLAINT_SUBTYPE")


analyze_tab, train_tab, guide_tab = st.tabs(
    ["Analyze complaints", "Train / update model", "Instructions"]
)


# =====================================================================
# TAB 1 — ANALYZE COMPLAINTS
# =====================================================================
with analyze_tab:
    st.subheader("1. Upload complaints")
    st.write(
        "Upload a new CSV or Excel file. This file does not need category labels. "
        "You will choose the column containing the actual complaint description."
    )

    analyze_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="analyze_file",
    )

    if analyze_file is not None:
        try:
            raw = read_uploaded_file(analyze_file)

            if raw.empty:
                st.error("The uploaded file contains no data rows.")
            else:
                st.success(
                    f"Loaded {len(raw):,} rows and {len(raw.columns):,} columns."
                )
                st.caption(
                    "For confidentiality, the preview is limited to the first five rows. "
                    "Avoid sharing screenshots containing consumer names, account numbers, "
                    "phone numbers, or other personal information."
                )
                st.dataframe(raw.head(5), use_container_width=True)

                columns = list(raw.columns)
                default_text_idx = preferred_column_index(
                    columns,
                    ["COMPLAINT_DETAILS", "complaint", "complaint_text", "description"],
                )
                complaint_col = st.selectbox(
                    "Select the complaint-description column",
                    options=columns,
                    index=default_text_idx,
                    key="analysis_complaint_column",
                    help=(
                        "Choose the column containing the customer's actual complaint "
                        "description. Do not select COMPLAINT_NO or another ID column."
                    ),
                )

                if st.button(
                    "Analyze complaints",
                    type="primary",
                    key="run_analysis",
                ):
                    model = load_model(MODEL_PATH)

                    if model is None:
                        st.error(
                            "No trained model was found. Open **Train / update model** "
                            "and train a model using labelled complaint data first."
                        )
                    else:
                        texts = raw[complaint_col].fillna("").astype(str).tolist()

                        if not any(text.strip() for text in texts):
                            st.error(
                                "The selected column does not contain any usable "
                                "complaint descriptions."
                            )
                        else:
                            with st.spinner(
                                "Classifying complaints and checking possible groups..."
                            ):
                                predictions = predict_complaints(model, texts)

                                results = raw.copy()
                                results["predicted_category"] = predictions[
                                    "predicted_category"
                                ]
                                results["prediction_confidence"] = predictions[
                                    "confidence"
                                ]
                                results["prediction_status"] = predictions["status"]

                                candidate_groups = discover_candidate_groups(
                                    texts,
                                    predictions["status"],
                                    min_cluster_size=3,
                                )
                                results["candidate_group"] = candidate_groups
                                # Ensure every complaint gets a corresponding candidate-group value.
                                if len(candidate_groups) != len(results):
                                    raise ValueError(
                                                     f"Candidate group mismatch: received {len(candidate_groups)} values "
                                                     f"for {len(results)} complaints."
                                                    )
 

                                summary = make_summary(
                                    results,
                                    category_col="predicted_category",
                                )

                                candidate_rows = results[
                                    results["candidate_group"]
                                    .fillna("")
                                    .astype(str)
                                    .str.strip()
                                    .ne("")
                                ].copy()

                                st.session_state["analysis_results"] = results
                                st.session_state["analysis_summary"] = summary
                                st.session_state["analysis_candidates"] = candidate_rows
                                st.session_state["analysis_complaint_column"] = complaint_col
                                st.session_state["analysis_source_name"] = analyze_file.name

                            st.success("Complaint analysis completed.")

        except Exception as exc:
            st.error(f"Could not read or analyze this file: {exc}")

    # Show the most recently completed analysis.
    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        summary = st.session_state["analysis_summary"]
        candidates = st.session_state["analysis_candidates"]
        text_column = st.session_state["analysis_complaint_column"]

        st.markdown("---")
        st.subheader("2. Analysis results")

        total_complaints = len(results)
        predicted_categories = results["predicted_category"].nunique()
        review_count = (
            results["prediction_status"].astype(str).ne("classified").sum()
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Complaints processed", f"{total_complaints:,}")
        col2.metric("Predicted categories", f"{predicted_categories:,}")
        col3.metric("Flagged for review", f"{review_count:,}")

        st.subheader("Categorized complaints")
        st.dataframe(results, use_container_width=True, height=360)

        st.subheader("Category frequency analysis")
        st.dataframe(summary, use_container_width=True)

        chart = make_bar_chart(summary)
        if chart is not None:
            st.pyplot(chart, use_container_width=True)
            plt.close(chart)

        review_rows = results[
            results["prediction_status"].astype(str).ne("classified")
        ]

        if not review_rows.empty:
            st.warning(
                "Some complaints need human review. The displayed confidence score "
                "is the model's highest class score; it is not a guarantee that the "
                "predicted category is correct."
            )
            st.subheader("Complaints requiring review")
            st.dataframe(review_rows, use_container_width=True)
        else:
            st.success("No complaints were flagged for review by the current threshold.")

        st.subheader("Potential new or unusual complaint groups")
        if candidates.empty:
            st.write(
                "No sufficiently large candidate groups were detected in this run."
            )
        else:
            st.caption(
                "These are exploratory clusters of review-status complaints. "
                "They are not confirmed categories. A person must inspect and approve "
                "a group before adding it to the training labels."
            )
            candidate_columns = [
                "candidate_group",
                text_column,
                "predicted_category",
                "prediction_confidence",
                "prediction_status",
            ]
            candidate_columns = [
                column for column in candidate_columns if column in candidates.columns
            ]
            st.dataframe(
                candidates[candidate_columns],
                use_container_width=True,
            )

        st.subheader("Download reports")
        try:
            excel_data = build_excel(results, summary, candidates)
            pdf_data = build_pdf(
                results,
                summary,
                candidates,
                st.session_state.get("analysis_source_name", "uploaded file"),
            )

            download_left, download_right = st.columns(2)
            download_left.download_button(
                "⬇️ Download categorized Excel",
                data=excel_data,
                file_name="categorized_complaints.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                key="download_excel_report",
            )
            download_right.download_button(
                "⬇️ Download PDF report",
                data=pdf_data,
                file_name="complaint_analysis_report.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )
        except Exception as exc:
            st.error(f"Could not generate the reports: {exc}")


# =====================================================================
# TAB 2 — TRAIN / UPDATE MODEL
# =====================================================================
with train_tab:
    st.subheader("Train a model using labelled complaint data")

    st.warning(
        "Use complaint categories that have been checked by a person. "
        "Automatically generated or keyword-based labels may contain mistakes. "
        "Audit the labels before treating them as ground truth."
    )

    st.write(
        "Upload one or more labelled CSV/Excel files. For every file, select "
        "the column containing complaint text and the column containing its "
        "verified category."
    )

    training_files = st.file_uploader(
        "Upload labelled CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="training_files",
    )

    selected_columns = {}

    if training_files:
        st.markdown("### Select columns for each training file")

        for file_index, training_file in enumerate(training_files):
            file_key = clean_file_key(training_file.name, file_index)

            try:
                training_preview = read_uploaded_file(training_file)
                columns = list(training_preview.columns)

                if len(columns) < 2:
                    st.error(
                        f"{training_file.name} must contain at least two columns."
                    )
                    continue

                with st.expander(
                    f"{training_file.name} — {len(training_preview):,} rows",
                    expanded=True,
                ):
                    st.caption("Preview of the first five rows")
                    st.dataframe(
                        training_preview.head(5),
                        use_container_width=True,
                    )

                    text_idx = preferred_column_index(
                        columns,
                        [
                            "COMPLAINT_DETAILS",
                            "complaint",
                            "complaint_text",
                            "description",
                        ],
                    )
                    category_idx = preferred_column_index(
                        columns,
                        [
                            "COMPLAINT_SUBTYPE",
                            "category",
                            "label",
                            "issue_type",
                        ],
                    )

                    # Avoid selecting the same column for both fields by default.
                    if category_idx == text_idx and len(columns) > 1:
                        category_idx = 1 if text_idx == 0 else 0

                    text_col = st.selectbox(
                        "Complaint-text column",
                        options=columns,
                        index=text_idx,
                        key=f"training_text_{file_key}",
                    )
                    category_col = st.selectbox(
                        "Category-label column",
                        options=columns,
                        index=category_idx,
                        key=f"training_category_{file_key}",
                    )

                    selected_columns[file_index] = (
                        text_col,
                        category_col,
                    )

                    if text_col == category_col:
                        st.error(
                            "The complaint-text and category columns must be different."
                        )

            except Exception as exc:
                st.error(f"Could not read {training_file.name}: {exc}")

    st.markdown("---")
    test_size = st.slider(
        "Test-set proportion",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help=(
            "The model uses this portion for holdout evaluation when the category "
            "counts allow a valid split. Some small datasets may not support "
            "stratified evaluation."
        ),
    )

    save_training_data = st.checkbox(
        "Save the combined labelled training data locally for future model updates",
        value=False,
        help=(
            "This stores complaint text and category labels in "
            "data/training/combined_training_data.csv. Leave unchecked if you "
            "do not want an additional local copy."
        ),
    )

    if training_files:
        if len(selected_columns) != len(training_files):
            st.info(
                "Check that every uploaded file loaded successfully and has "
                "both columns selected."
            )

    train_disabled = (
        not bool(training_files)
        or len(selected_columns) != len(training_files)
        or any(
            text_col == category_col
            for text_col, category_col in selected_columns.values()
        )
    )

    if st.button(
        "Train / refresh model",
        type="primary",
        disabled=train_disabled,
        key="train_model_button",
    ):
        try:
            frames = []

            for file_index, training_file in enumerate(training_files):
                text_col, category_col = selected_columns[file_index]
                dataframe = read_uploaded_file(training_file)

                if text_col not in dataframe.columns or category_col not in dataframe.columns:
                    raise ValueError(
                        f"Selected columns were not found in {training_file.name}."
                    )

                frame = normalize_training_frame(
                    dataframe,
                    text_col,
                    category_col,
                )
                frame["source_file"] = training_file.name
                frames.append(frame)

            if not frames:
                raise ValueError("No usable training files were provided.")

            training_data = pd.concat(frames, ignore_index=True)
            training_data = training_data.dropna(
                subset=["complaint", "category"]
            )
            training_data["complaint"] = (
                training_data["complaint"].astype(str).str.strip()
            )
            training_data["category"] = (
                training_data["category"].astype(str).str.strip()
            )
            training_data = training_data[
                (training_data["complaint"] != "")
                & (training_data["category"] != "")
            ]

            if training_data.empty:
                raise ValueError(
                    "No rows remain after removing blank complaint text and labels."
                )

            duplicate_pair_count = training_data.duplicated(
                subset=["complaint", "category"]
            ).sum()
            training_data = training_data.drop_duplicates(
                subset=["complaint", "category"]
            )

            category_counts = (
                training_data.groupby("category")
                .size()
                .rename("examples")
                .reset_index()
                .sort_values("examples", ascending=False)
            )

            st.markdown("### Training data overview")
            first, second, third = st.columns(3)
            first.metric("Usable labelled rows", f"{len(training_data):,}")
            second.metric("Distinct categories", f"{training_data['category'].nunique():,}")
            third.metric("Duplicate pairs removed", f"{duplicate_pair_count:,}")

            st.dataframe(category_counts, use_container_width=True)

            rare_categories = category_counts[
                category_counts["examples"] < 5
            ]
            if not rare_categories.empty:
                st.warning(
                    f"{len(rare_categories)} categories have fewer than five "
                    "examples. The model may not learn these categories reliably. "
                    "They are retained in the training data rather than silently "
                    "discarded."
                )

            categories_with_one_example = category_counts[
                category_counts["examples"] < 2
            ]
            if not categories_with_one_example.empty:
                st.warning(
                    f"{len(categories_with_one_example)} categories have only "
                    "one example. Such categories cannot be meaningfully evaluated "
                    "on a stratified holdout set."
                )

            with st.spinner(
                "Creating text embeddings, training the classifier, and evaluating..."
            ):
                model, metrics = train_and_evaluate(
                    training_data["complaint"].tolist(),
                    training_data["category"].tolist(),
                    test_size=float(test_size),
                )

                # Save only after training has completed successfully.
                save_model(model, MODEL_PATH)

                if save_training_data:
                    training_data.to_csv(
                        ROOT / "data" / "training" / "combined_training_data.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

            st.success("Model trained and saved locally.")

            if save_training_data:
                st.info(
                    "The combined labelled dataset was also saved locally in "
                    "data/training/combined_training_data.csv."
                )
            else:
                st.info(
                    "Only the trained model was saved. The uploaded training "
                    "dataset was not copied into the training folder."
                )

            evaluation_note = metrics.get("evaluation_note")
            if evaluation_note:
                st.write(evaluation_note)

            accuracy = metrics.get("accuracy")
            macro_f1 = metrics.get("macro_f1")
            report = metrics.get("report")

            if accuracy is not None:
                metric_left, metric_right = st.columns(2)
                metric_left.metric("Holdout accuracy", f"{accuracy:.3f}")
                if macro_f1 is not None:
                    metric_right.metric("Holdout macro F1", f"{macro_f1:.3f}")

                if report:
                    st.subheader("Per-category evaluation report")
                    st.dataframe(
                        pd.DataFrame(report).T,
                        use_container_width=True,
                    )

                st.caption(
                    "Holdout metrics are estimates, not guarantees. Review the "
                    "per-category precision, recall, F1-score, test-set size, "
                    "and label quality. Categories with very few examples may "
                    "not receive meaningful evaluation."
                )
            else:
                st.warning(
                    "The model was trained, but a reliable holdout evaluation "
                    "was not available for this dataset."
                )

            st.info(
                "For future updates, upload the previous verified labelled data "
                "together with newly verified examples. Keep category names "
                "consistent across files."
            )

        except Exception as exc:
            st.error(f"Training failed: {exc}")


# =====================================================================
# TAB 3 — INSTRUCTIONS
# =====================================================================
with guide_tab:
    st.subheader("Quick start")

    st.markdown(
        """
1. Activate your project's virtual environment.
2. Install the packages listed in `requirements.txt`.
3. Run the application from the project root:

   ```bash
   streamlit run app.py
   ```

4. Open **Train / update model**.
5. Upload labelled complaint data.
6. Select the actual complaint-text and category columns for every file.
7. Train and evaluate the model.
8. Open **Analyze complaints** and upload a new complaint file.
9. Select the complaint-description column and click **Analyze complaints**.
10. Review uncertain predictions and potential complaint groups.
11. Download the categorized Excel workbook and PDF report.

### Recommended columns for the current electricity complaint dataset

- **Complaint text:** `COMPLAINT_DETAILS`
- **Category label:** `COMPLAINT_SUBTYPE`

Do not select `COMPLAINT_NO` as complaint text. It is a complaint identifier, not the customer's description.

### Training data guidance

- Use labels verified by a person.
- Keep category names consistent across files.
- Categories with very few examples may be difficult to learn.
- Do not interpret a high overall accuracy as proof that every category is predicted correctly.
- Review macro F1 and per-category precision, recall, and F1-score.
- Add newly reviewed complaints to the labelled training data before retraining.

### Model

The current classifier uses Sentence Transformer text embeddings with a Logistic Regression classifier. The text encoder represents complaint descriptions numerically; the classifier learns to map those representations to the supplied category labels.

The confidence value is the classifier's highest predicted class score. It is a useful review signal, but it is not a calibrated guarantee of correctness.

Potential new complaint groups are exploratory clusters of complaints flagged for review. They are not automatically approved categories.

### Privacy

The application is designed for local execution and does not call an external AI API. Uploaded complaint records are processed by the local application. The trained model is saved in the project's `models` folder.

If you choose to save the combined labelled training data, it is stored in `data/training/combined_training_data.csv`. Treat that file as confidential.

Do not expose the local Streamlit service to the public internet or an untrusted network. For use by coworkers, deployment should be on an organisation-approved server with authentication, HTTPS, access controls, and an agreed data-retention policy.
        """
    )

    with st.expander("Project folder structure"):
        st.code(
            """Automatic-customer-complaint-analysis/
├── app.py
├── requirements.txt
├── README.md
├── complaint_analyzer/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── modeling.py
│   └── reporting.py
├── data/
│   └── training/
├── models/
├── outputs/
└── tests/""",
            language="text",
        )
