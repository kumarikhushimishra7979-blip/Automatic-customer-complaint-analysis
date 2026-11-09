from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from complaint_analyzer.data_utils import read_table, find_complaint_column, make_summary
from complaint_analyzer.modeling import (
    load_model, predict_complaints, discover_candidate_groups
)
from complaint_analyzer.reporting import build_excel, build_pdf

MODEL_PATH = ROOT / "models" / "complaint_model.joblib"


@st.cache_resource
def get_trained_model():
    return load_model(MODEL_PATH)


st.set_page_config(page_title="Customer Complaint Analyzer", page_icon="📊", layout="wide")
st.title("📊 Customer Complaint Analyzer")
st.caption(
    "Upload an Excel/CSV file to automatically generate complaint categories, "
    "insights, charts, and downloadable reports."
)

with st.sidebar:
    st.header("Application")
    st.success("Ready for complaint analysis")
    st.caption("The trained model is maintained by the application developer.")

model = get_trained_model()
if model is None:
    st.error(
        "The trained model is missing. The application administrator must run "
        "training/train_model.py before users can analyze files."
    )
    st.stop()

st.subheader("Upload your complaint file")
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        raw = read_table(uploaded_file)
        if raw.empty:
            st.error("The uploaded file contains no data.")
            st.stop()

        st.success(f"File loaded: {uploaded_file.name} — {len(raw):,} rows")

        columns = list(raw.columns)
        preferred = [
            "COMPLAINT_DETAILS", "complaint", "complaint_text",
            "Complaint Text", "description"
        ]
        suggested = next(
            (name for name in preferred if name in columns),
            find_complaint_column(columns)
        )
        default_index = columns.index(suggested) if suggested in columns else 0

        complaint_column = st.selectbox(
            "Select the column containing complaint descriptions",
            columns,
            index=default_index
        )

        with st.expander("Preview uploaded data"):
            st.dataframe(raw.head(10), use_container_width=True)

        if st.button("Generate Complaint Insights", type="primary"):
            texts = raw[complaint_column].fillna("").astype(str).tolist()
            if not any(text.strip() for text in texts):
                st.error("The selected column contains no complaint text.")
                st.stop()

            with st.spinner("Analyzing complaints and preparing reports..."):
                predictions = predict_complaints(model, texts)
                results = raw.copy()
                results["predicted_category"] = predictions["predicted_category"].tolist()
                results["prediction_confidence"] = predictions["confidence"].tolist()
                results["prediction_status"] = predictions["status"].tolist()

                groups = discover_candidate_groups(
                    texts, predictions["status"].tolist(), min_cluster_size=3
                )
                if len(groups) != len(results):
                    raise ValueError(
                        "Candidate-group output does not match uploaded row count. "
                        "Check discover_candidate_groups() in modeling.py."
                    )

                results["candidate_group"] = groups
                summary = make_summary(results)
                candidates = results[
                    results["candidate_group"].astype(str).str.strip().ne("")
                ].copy()

                st.session_state["analysis_results"] = results
                st.session_state["analysis_summary"] = summary
                st.session_state["analysis_candidates"] = candidates
                st.session_state["analysis_source"] = uploaded_file.name
                st.session_state["analysis_column"] = complaint_column

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    summary = st.session_state["analysis_summary"]
    candidates = st.session_state["analysis_candidates"]
    source_name = st.session_state["analysis_source"]
    complaint_column = st.session_state["analysis_column"]

    st.divider()
    st.header("Complaint Analysis Dashboard")

    total = len(results)
    category_count = int(results["predicted_category"].nunique())
    review_count = int((results["prediction_status"] != "classified").sum())
    top3_count = int(summary.head(3)["count"].sum()) if not summary.empty else 0
    top3_share = top3_count / total * 100 if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total complaints", f"{total:,}")
    m2.metric("Predicted categories", f"{category_count:,}")
    m3.metric("Flagged for review", f"{review_count:,}")
    m4.metric("Top 3 category share", f"{top3_share:.1f}%")

    st.subheader("Category Summary")
    if not summary.empty:
        shown = summary.copy()
        shown.insert(0, "Rank", range(1, len(shown) + 1))
        shown = shown.rename(columns={
            "category": "Issue Type",
            "count": "Complaint Count",
            "percentage": "Percentage"
        })
        st.dataframe(shown, hide_index=True, use_container_width=True)

        plot_df = summary.sort_values("count", ascending=True)

        fig, ax = plt.subplots(
            figsize=(12, max(3.5, min(12, len(plot_df) * 0.38)))
        )

        # Generate a different colour for every complaint category.
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))
        bars = ax.barh(
            plot_df["category"].astype(str),
            plot_df["count"],
            color=colors,
            edgecolor="white",
            linewidth=0.8
        )
        ax.bar_label(bars, padding=4, fontsize=9)
        ax.set_title("Complaints by Issue Type", fontsize=14, weight="bold")
        ax.set_xlabel("Number of Complaints")
        ax.set_ylabel("Issue Type")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Categorized Complaints")
    st.dataframe(results, use_container_width=True, height=400)

    review = results[results["prediction_status"] != "classified"]
    if not review.empty:
        st.subheader("Complaints Requiring Human Review")
        st.warning(
            "These predictions did not meet the model's confidence threshold. "
            "Review them before taking action."
        )
        st.dataframe(review, use_container_width=True)

    st.subheader("Potential New Complaint Groups")
    if candidates.empty:
        st.info("No sufficiently large candidate groups were detected.")
    else:
        st.caption(
            "These are exploratory text clusters, not confirmed official categories."
        )
        st.dataframe(
            candidates[[
                "candidate_group", complaint_column,
                "predicted_category", "prediction_confidence"
            ]],
            use_container_width=True
        )

    st.subheader("Download Reports")
    excel_bytes = build_excel(results, summary, candidates)
    pdf_bytes = build_pdf(results, summary, candidates, source_name)

    left, right = st.columns(2)
    left.download_button(
        "⬇️ Download Excel Report",
        data=excel_bytes,
        file_name="customer_complaint_categorization_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    right.download_button(
        "⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="customer_complaint_categorization_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
