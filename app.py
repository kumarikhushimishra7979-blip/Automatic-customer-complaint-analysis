from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from complaint_analyzer.data_utils import read_table, find_complaint_column, find_category_column, normalize_training_frame, make_summary
from complaint_analyzer.modeling import train_and_evaluate, load_model, save_model, predict_complaints, discover_candidate_groups
from complaint_analyzer.reporting import build_excel, build_pdf

st.set_page_config(page_title="Customer Complaint Analyzer", page_icon="📊", layout="wide")
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "data" / "training").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

st.title("📊 Customer Complaint Analyzer")
st.caption("Local-first complaint classification, category summaries, candidate-group discovery, and downloadable reports.")
with st.sidebar:
    st.header("Project status")
    st.write("**Processing:** on this computer")
    st.write("**External AI API:** not used")
    st.info("Local processing reduces exposure but does not replace device security or access controls.")
    st.markdown("**Training columns:** `complaint`, `category`")

analyze_tab, train_tab, guide_tab = st.tabs(["Analyze complaints", "Train / update model", "Instructions"])

with analyze_tab:
    st.subheader("1. Upload complaints")
    upload = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="analyze_file")
    if upload:
        try:
            raw = read_table(upload)
            st.write("Preview")
            st.dataframe(raw.head(10), use_container_width=True)
            suggested = find_complaint_column(raw.columns)
            cols = list(raw.columns)
            default_idx = cols.index(suggested) if suggested in cols else 0
            complaint_col = st.selectbox("Select the complaint-text column", cols, index=default_idx)
            if st.button("Analyze complaints", type="primary"):
                model = load_model(ROOT / "models" / "complaint_model.joblib")
                if model is None:
                    st.error("No trained model found. Train it first in **Train / update model**.")
                else:
                    texts = raw[complaint_col].fillna("").astype(str).tolist()
                    if not any(t.strip() for t in texts):
                        st.error("The selected column contains no complaint text.")
                    else:
                        with st.spinner("Classifying complaints and checking for possible groups..."):
                            pred = predict_complaints(model, texts)
                            results = raw.copy()
                            results["predicted_category"] = pred["predicted_category"]
                            results["prediction_confidence"] = pred["confidence"]
                            results["prediction_status"] = pred["status"]
                            groups = discover_candidate_groups(texts, pred["status"].tolist(), min_cluster_size=3)
                            results["candidate_group"] = groups
                            st.session_state["results"] = results
                            st.session_state["summary"] = make_summary(results)
                            st.session_state["candidate_rows"] = results[results["candidate_group"].astype(str).str.strip().ne("")].copy()
                            st.session_state["complaint_col"] = complaint_col
                            st.session_state["source_name"] = upload.name
        except Exception as exc:
            st.error(f"Could not read or analyze this file: {exc}")

    if "results" in st.session_state:
        results = st.session_state["results"]
        summary = st.session_state["summary"]
        candidates = st.session_state["candidate_rows"]
        a, b, c = st.columns(3)
        a.metric("Complaints processed", f"{len(results):,}")
        b.metric("Predicted categories", f"{results['predicted_category'].nunique():,}")
        c.metric("Flagged for review", f"{(results['prediction_status'] != 'classified').sum():,}")
        st.subheader("Categorized complaints")
        st.dataframe(results, use_container_width=True, height=360)
        st.subheader("Category distribution")
        st.dataframe(summary, use_container_width=True)
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(10, max(3, min(8, len(summary) * 0.35))))
            plot_df = summary.sort_values("count", ascending=True)
            ax.barh(plot_df["category"].astype(str), plot_df["count"])
            ax.set_xlabel("Number of complaints")
            ax.set_title("Complaint category distribution")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        review = results[results["prediction_status"] != "classified"]
        if not review.empty:
            st.warning("Low-confidence predictions are flagged for review. Confidence is a margin indicator, not a probability.")
            st.dataframe(review, use_container_width=True)
        st.subheader("Possible new or unusual complaint groups")
        if candidates.empty:
            st.write("No sufficiently large candidate groups were detected in this run.")
        else:
            st.caption("These exploratory text clusters are not confirmed categories. Review them before adding any official category.")
            st.dataframe(candidates[["candidate_group", st.session_state["complaint_col"], "predicted_category", "prediction_confidence"]], use_container_width=True)
        excel_data = build_excel(results, summary, candidates)
        pdf_data = build_pdf(results, summary, candidates, st.session_state.get("source_name", "uploaded file"))
        left, right = st.columns(2)
        left.download_button("⬇️ Download categorized Excel", excel_data, "categorized_complaints.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        right.download_button("⬇️ Download PDF report", pdf_data, "complaint_analysis_report.pdf", "application/pdf")

with train_tab:
    st.subheader("Train a model using verified labels")
    st.warning("Use categories checked by a person. Keyword-generated labels may contain errors and should be audited before training.")
    st.write("Each file must contain a complaint-text column and a category-label column. Preferred names are `complaint` and `category`.")
    train_files = st.file_uploader("Upload one or more labeled CSV/Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="training_files")
    if train_files:
        for f in train_files:
            try:
                preview = read_table(f)
                st.write(f"**{f.name}**")
                st.dataframe(preview.head(5), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read {f.name}: {exc}")
    test_size = st.slider("Test-set proportion", 0.1, 0.35, 0.2, 0.05)
    if st.button("Train / refresh model", type="primary", disabled=not bool(train_files)):
        try:
            frames = []
            for f in train_files:
                df = read_table(f)
                ccol = find_complaint_column(df.columns)
                kcol = find_category_column(df.columns)
                if ccol is None or kcol is None or ccol == kcol:
                    st.error(f"Cannot identify distinct complaint and category columns in {f.name}. Rename them to `complaint` and `category`.")
                    st.stop()
                frame = normalize_training_frame(df, ccol, kcol)
                frame["source_file"] = f.name
                frames.append(frame)
            training = pd.concat(frames, ignore_index=True)
            training = training.dropna(subset=["complaint", "category"])
            training["complaint"] = training["complaint"].astype(str).str.strip()
            training["category"] = training["category"].astype(str).str.strip()
            training = training[(training["complaint"] != "") & (training["category"] != "")]
            old_n = len(training)
            training = training.drop_duplicates(subset=["complaint", "category"])
            st.write(f"Usable rows: **{len(training):,}**; exact duplicate complaint/category pairs removed: **{old_n-len(training):,}**")
            st.dataframe(training.groupby("category").size().rename("examples").reset_index().sort_values("examples", ascending=False), use_container_width=True)
            with st.spinner("Training and evaluating..."):
                model, metrics = train_and_evaluate(training["complaint"].tolist(), training["category"].tolist(), test_size=float(test_size))
                save_model(model, ROOT / "models" / "complaint_model.joblib")
                training.to_csv(ROOT / "data" / "training" / "combined_training_data.csv", index=False, encoding="utf-8-sig")
            st.success("Model trained and saved locally.")
            st.write(metrics["evaluation_note"])
            if metrics["accuracy"] is not None:
                x, y = st.columns(2)
                x.metric("Holdout accuracy", f"{metrics['accuracy']:.3f}")
                y.metric("Holdout macro F1", f"{metrics['macro_f1']:.3f}")
                st.dataframe(pd.DataFrame(metrics["report"]).T, use_container_width=True)
                st.caption("Holdout scores are estimates, not guarantees. Review per-class scores and label quality.")
            st.info("For later updates, upload the old labeled data together with new verified labeled datasets, then retrain. Keep category names consistent across files.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

with guide_tab:
    st.subheader("Quick start")
    st.markdown("""
1. Install Python 3.11 or newer and the packages in `requirements.txt`.
2. Run `streamlit run app.py` from this project folder.
3. Open **Train / update model**, upload labeled data, and train.
4. Open **Analyze complaints**, upload an unlabeled file, confirm the complaint column, and analyze.
5. Download the categorized Excel workbook and PDF report.

**Training format:** one complaint per row, with the original complaint text in `complaint` and a checked label in `category`. Categories must be consistent across datasets.

**Model:** TF-IDF text features + Linear SVM. It is a baseline text classifier. The app does not call an external AI API or cloud database. New-category discovery is exploratory clustering of lower-confidence complaints, not automatic confirmation of new categories.

**Privacy:** the application is designed for local execution. Do not expose its local web service to the internet or an untrusted network. Use non-confidential sample data when testing. Local execution alone does not guarantee security.
""")
    st.code("""customer_complaint_analyzer/\n├── app.py\n├── requirements.txt\n├── README.md\n├── complaint_analyzer/\n│   ├── data_utils.py\n│   ├── modeling.py\n│   └── reporting.py\n├── data/training/\n├── models/\n└── outputs/""")
