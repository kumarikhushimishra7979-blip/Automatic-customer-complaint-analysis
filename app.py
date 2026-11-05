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