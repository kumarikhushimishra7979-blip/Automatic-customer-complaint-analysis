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
from complaint_analyzer.ui import STYLE, callout, model_status, hero_html, sidebar_html, category_chart

st.set_page_config(page_title="Customer Complaint Analyzer", page_icon="📊", layout="wide")
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "data" / "training").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)
st.markdown(STYLE, unsafe_allow_html=True)

# The hero is filled in at the end of the script so it reflects training and analysis done in this run.
hero_slot = st.empty()

CONF_COLUMN = st.column_config.ProgressColumn("prediction_confidence", min_value=0.0, max_value=1.0, format="%.2f")

analyze_tab, train_tab, guide_tab = st.tabs(["Analyze complaints", "Train / update model", "Instructions"])

# ----------------------------------------------------------------------------------------------
# Analyze
# ----------------------------------------------------------------------------------------------
with analyze_tab:
    if model_status(ROOT) is None:
        st.markdown(callout("No trained model yet. Open <b>Train / update model</b>, upload labeled complaints, and train once. Then return here to analyze.", "warn"), unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Upload complaints")
        st.caption("A CSV or Excel file with one complaint per row. The file does not need categories.")
        upload = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"], key="analyze_file", label_visibility="collapsed")
        if upload:
            try:
                raw = read_table(upload)
                st.caption(f"{len(raw):,} rows and {len(raw.columns)} columns found. First 10 rows:")
                st.dataframe(raw.head(10), use_container_width=True)
                suggested = find_complaint_column(raw.columns)
                cols = list(raw.columns)
                default_idx = cols.index(suggested) if suggested in cols else 0
                complaint_col = st.selectbox("Which column contains the complaint text?", cols, index=default_idx)
                if st.button("Analyze complaints", type="primary"):
                    model = load_model(ROOT / "models" / "complaint_model.joblib")
                    if model is None:
                        st.error("No trained model found. Train one in **Train / update model** first.")
                    else:
                        texts = raw[complaint_col].fillna("").astype(str).tolist()
                        if not any(t.strip() for t in texts):
                            st.error("The selected column has no complaint text. Choose a different column.")
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
        review = results[results["prediction_status"] != "classified"]
        total, flagged = len(results), len(review)

        st.subheader("Results")
        st.caption(f"From {st.session_state.get('source_name', 'uploaded file')}")
        a, b, c, d = st.columns(4)
        a.metric("Complaints processed", f"{total:,}")
        b.metric("Categories found", f"{results['predicted_category'].nunique():,}")
        c.metric("Classified automatically", f"{(total - flagged) / max(total, 1):.0%}")
        d.metric("Need review", f"{flagged:,}")

        excel_data = build_excel(results, summary, candidates)
        pdf_data = build_pdf(results, summary, candidates, st.session_state.get("source_name", "uploaded file"))
        left, right, _ = st.columns([1, 1, 2])
        left.download_button("Download Excel", excel_data, "categorized_complaints.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)
        right.download_button("Download PDF report", pdf_data, "complaint_analysis_report.pdf", "application/pdf", use_container_width=True)

        t_over, t_all, t_review, t_groups = st.tabs(["Overview", "All complaints", f"Need review ({flagged:,})", f"Possible new groups ({candidates['candidate_group'].nunique():,})"])

        with t_over:
            if summary.empty:
                st.write("No categories to show.")
            else:
                chart_col, table_col = st.columns([3, 2])
                with chart_col:
                    st.markdown("#### Complaints by category")
                    fig = category_chart(summary, top_n=15)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    if len(summary) > 15:
                        st.caption(f"Showing the 15 largest of {len(summary)} categories. The full list is in the table and the Excel file.")
                with table_col:
                    st.markdown("#### Category table")
                    st.dataframe(summary, use_container_width=True, hide_index=True, height=420,
                                 column_config={"count": st.column_config.NumberColumn("Complaints"), "percentage": st.column_config.NumberColumn("Share (%)", format="%.1f")})

        with t_all:
            st.dataframe(results, use_container_width=True, height=440, column_config={"prediction_confidence": CONF_COLUMN})

        with t_review:
            if review.empty:
                st.markdown(callout("Every complaint was classified with enough confidence.", "ok"), unsafe_allow_html=True)
            else:
                st.markdown(callout("These predictions had low confidence, so check them by hand. Confidence is a margin indicator, not a probability.", "warn"), unsafe_allow_html=True)
                st.dataframe(review, use_container_width=True, height=440, column_config={"prediction_confidence": CONF_COLUMN})

        with t_groups:
            if candidates.empty:
                st.markdown(callout("No candidate groups were detected in this run. Groups appear when several low-confidence complaints describe a similar issue.", "info"), unsafe_allow_html=True)
            else:
                st.markdown(callout("These clusters are suggestions, not confirmed categories. Review them before adding an official category.", "info"), unsafe_allow_html=True)
                st.dataframe(candidates[["candidate_group", st.session_state["complaint_col"], "predicted_category", "prediction_confidence"]],
                             use_container_width=True, height=440, column_config={"prediction_confidence": CONF_COLUMN})

# ----------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------
with train_tab:
    st.markdown(callout("Train only on categories a person has checked. Labels generated from keywords can contain errors and should be reviewed first.", "warn"), unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Upload labeled complaints")
        st.caption("Each file needs a column with the complaint text and a column with its category. You can upload several files.")
        train_files = st.file_uploader("Upload one or more labeled CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="training_files", label_visibility="collapsed")

        selections = {}
        if train_files:
            for i, f in enumerate(train_files):
                try:
                    preview = read_table(f)
                except Exception as exc:
                    st.error(f"Could not read {f.name}: {exc}")
                    continue
                with st.expander(f"{f.name}  ({len(preview):,} rows)", expanded=(i == 0)):
                    st.dataframe(preview.head(5), use_container_width=True)
                    cols = list(preview.columns)
                    guess_c, guess_k = find_complaint_column(cols), find_category_column(cols)
                    p1, p2 = st.columns(2)
                    ccol = p1.selectbox("Complaint text column", cols, index=cols.index(guess_c) if guess_c in cols else 0, key=f"tc_{i}_{f.name}")
                    kcol = p2.selectbox("Category column", cols, index=cols.index(guess_k) if guess_k in cols else 0, key=f"tk_{i}_{f.name}")
                    n_cat = preview[kcol].dropna().nunique()
                    if ccol == kcol:
                        st.error("Choose two different columns.")
                    elif n_cat < 2:
                        st.error(f"The column '{kcol}' has {n_cat} distinct value. The model needs at least 2 categories, so pick the column that holds the category.")
                    else:
                        st.caption(f"{n_cat:,} distinct categories in '{kcol}'.")
                    selections[i] = (ccol, kcol)

        test_size = st.slider("Share of data held back for testing", 0.1, 0.35, 0.2, 0.05, help="The model is scored on this part of the data, which it never sees while learning.")
        train_clicked = st.button("Train / refresh model", type="primary", disabled=not bool(train_files))

    if train_clicked:
        try:
            frames = []
            for i, f in enumerate(train_files):
                df = read_table(f)
                ccol, kcol = selections.get(i, (None, None))
                if ccol is None or kcol is None or ccol == kcol:
                    st.error(f"Cannot identify distinct complaint and category columns in {f.name}. Select them above.")
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
            with st.spinner("Training and evaluating..."):
                model, metrics = train_and_evaluate(training["complaint"].tolist(), training["category"].tolist(), test_size=float(test_size))
                save_model(model, ROOT / "models" / "complaint_model.joblib")
                training.to_csv(ROOT / "data" / "training" / "combined_training_data.csv", index=False, encoding="utf-8-sig")
            st.markdown(callout("Model trained and saved on this computer. You can now analyze complaints.", "ok"), unsafe_allow_html=True)

            st.subheader("Training summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Usable examples", f"{len(training):,}")
            s2.metric("Categories", f"{training['category'].nunique():,}")
            s3.metric("Duplicates removed", f"{old_n - len(training):,}")
            if metrics["accuracy"] is not None:
                x, y, _ = st.columns([1, 1, 1])
                x.metric("Accuracy on held-back data", f"{metrics['accuracy']:.1%}")
                y.metric("Macro F1", f"{metrics['macro_f1']:.2f}")
            st.caption(metrics["evaluation_note"])

            counts = training.groupby("category").size().rename("examples").reset_index().sort_values("examples", ascending=False)
            st.markdown("#### Examples per category")
            st.dataframe(counts, use_container_width=True, hide_index=True,
                         column_config={"examples": st.column_config.ProgressColumn("Examples", min_value=0, max_value=int(counts["examples"].max()), format="%d")})
            if metrics["accuracy"] is not None:
                with st.expander("Per-category scores"):
                    st.dataframe(pd.DataFrame(metrics["report"]).T, use_container_width=True)
                    st.caption("Scores are estimates, not guarantees. Review each category's score and the quality of its labels.")
            st.markdown(callout("To update the model later, upload the old labeled data together with the new checked data and train again. Keep category names the same across files.", "info"), unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

# ----------------------------------------------------------------------------------------------
# Guide
# ----------------------------------------------------------------------------------------------
with guide_tab:
    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.subheader("Quick start")
            st.markdown("""
1. Open **Train / update model**, upload labeled data, check the two column pickers, and train.
2. Open **Analyze complaints**, upload an unlabeled file, and confirm the complaint column.
3. Review the results, then download the Excel workbook and PDF report.

**Training format:** one complaint per row, with the complaint text and a checked category label. Category names must be consistent across files.
""")
    with g2:
        with st.container(border=True):
            st.subheader("How it works")
            st.markdown("""
The model turns each complaint into word-based features (TF-IDF) and classifies it with a Linear SVM. It is a baseline text classifier. The app calls no external AI API or cloud database.

Complaints with low confidence are flagged for review and grouped by similarity. These groups are suggestions, not confirmed categories.
""")
    st.markdown(callout("<b>Privacy:</b> the app is designed to run locally. Do not expose its web address to the internet or an untrusted network, and use non-confidential data when testing. Running locally does not by itself guarantee security.", "info"), unsafe_allow_html=True)
    with st.expander("Project layout"):
        st.code("""customer_complaint_analyzer/\n├── app.py\n├── requirements.txt\n├── README.md\n├── .streamlit/config.toml\n├── complaint_analyzer/\n│   ├── data_utils.py\n│   ├── modeling.py\n│   ├── reporting.py\n│   └── ui.py\n├── data/training/\n├── models/\n└── outputs/""")

# ----------------------------------------------------------------------------------------------
# Header and sidebar (rendered last so they reflect this run)
# ----------------------------------------------------------------------------------------------
status = model_status(ROOT)
with st.sidebar:
    st.markdown(sidebar_html(status), unsafe_allow_html=True)
hero_slot.markdown(hero_html(status, len(st.session_state["results"]) if "results" in st.session_state else 0), unsafe_allow_html=True)
