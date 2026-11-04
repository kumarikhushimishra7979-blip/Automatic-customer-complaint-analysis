"""Visual styling and small display helpers for the Streamlit app."""
from datetime import datetime
from html import escape
import pandas as pd

INK = "#10242B"
MUTED = "#5A6F72"
TEAL = "#0F766E"

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
:root{--ink:#10242B;--muted:#5A6F72;--paper:#F3F6F5;--line:#D6E0DE;--teal:#0F766E;--teal-deep:#0A4F4B;--mint:#5EEAD4;--amber:#8A5A00;--amber-bg:#FDF3DC;--ok-bg:#E3F4EF;}
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp select {font-family:'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;}
/* Keep Streamlit's icon font intact, otherwise icon names print as text over the labels. */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined {font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;}
h1, h2, h3, h4 {font-family:'Bricolage Grotesque','Public Sans', system-ui, sans-serif !important; color:var(--ink); letter-spacing:-0.01em;}
.block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer {visibility:hidden;}

/* Hero */
.hero {background:var(--teal-deep); color:#E8F3F1; border-radius:16px; padding:1.7rem 1.9rem 1.5rem; margin-bottom:1.4rem;}
.hero-title {font-family:'Bricolage Grotesque','Public Sans',sans-serif; font-weight:700; font-size:2.15rem; line-height:1.1; color:#fff; margin:0;}
.hero-sub {margin:.55rem 0 1.25rem; max-width:46rem; font-size:1.02rem; color:#C5DEDA; line-height:1.5;}
.stages {display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;}
.stage {background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.16); border-left:3px solid rgba(255,255,255,.28); border-radius:10px; padding:.8rem 1rem;}
.stage.done {border-left-color:var(--mint); background:rgba(94,234,212,.10);}
.stage-name {font-size:.82rem; color:#A9CBC6; font-weight:500;}
.stage-value {font-family:'Bricolage Grotesque','Public Sans',sans-serif; font-size:1.2rem; font-weight:600; color:#fff; margin-top:.1rem;}
.stage-sub {font-size:.82rem; color:#A9CBC6; margin-top:.1rem;}
@media (max-width: 800px) {.stages {grid-template-columns:1fr;} .hero-title {font-size:1.7rem;}}

/* Sidebar */
[data-testid="stSidebar"] {background:#0B3B3C;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
.side-title {font-family:'Bricolage Grotesque','Public Sans',sans-serif; font-weight:600; font-size:1.05rem; color:#fff; margin-bottom:.6rem;}
.side-row {display:flex; justify-content:space-between; gap:10px; padding:.5rem 0; border-bottom:1px solid rgba(255,255,255,.12); font-size:.88rem; color:#B9D6D2;}
.side-row b {color:#fff; font-weight:600; text-align:right;}
.side-note {margin-top:1rem; font-size:.82rem; line-height:1.5; color:#9FC2BD;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"] {height:44px; padding:0 14px; font-weight:600; color:var(--muted);}
.stTabs [aria-selected="true"] {color:var(--teal);}
.stTabs [data-baseweb="tab-highlight"] {background-color:var(--teal);}

/* Metric cards */
[data-testid="stMetric"] {background:#fff; border:1px solid var(--line); border-radius:12px; padding:.9rem 1.1rem;}
[data-testid="stMetricLabel"] {color:var(--muted);}
[data-testid="stMetricValue"] {font-family:'Bricolage Grotesque','Public Sans',sans-serif; font-weight:700; color:var(--ink);}

/* Buttons and upload */
.stButton > button, .stDownloadButton > button {border-radius:9px; font-weight:600; padding:.55rem 1.1rem;}
[data-testid="stFileUploader"] section {border:1.5px dashed #9DB8B4; background:#F8FBFA; border-radius:12px;}

/* Callouts */
.callout {border-radius:10px; padding:.8rem 1rem; margin:.4rem 0 1rem; font-size:.95rem; line-height:1.5;}
.callout.warn {background:var(--amber-bg); color:var(--amber); border-left:4px solid #D99A1B;}
.callout.ok {background:var(--ok-bg); color:#0A4F4B; border-left:4px solid var(--teal);}
.callout.info {background:#EAF1F5; color:#23465C; border-left:4px solid #5B8FB0;}
</style>
"""


def callout(text, kind="info"):
    return f'<div class="callout {kind}">{text}</div>'


def model_status(root):
    """Return (trained_at, example_count) or None when no model has been trained."""
    path = root / "models" / "complaint_model.joblib"
    if not path.exists():
        return None
    trained_at = datetime.fromtimestamp(path.stat().st_mtime)
    count = None
    try:
        data = pd.read_csv(root / "data" / "training" / "combined_training_data.csv", usecols=["complaint"], encoding="utf-8-sig")
        count = len(data)
    except Exception:
        pass
    return trained_at, count


def _stage(title, value, sub, done):
    cls = "stage done" if done else "stage"
    return f'<div class="{cls}"><div class="stage-name">{escape(title)}</div><div class="stage-value">{escape(value)}</div><div class="stage-sub">{escape(sub)}</div></div>'


def hero_html(status, results_count):
    if status:
        trained_at, count = status
        when = trained_at.strftime("%d %b %Y")
        sub = (f"{count:,} examples, trained {when}" if count else f"Trained {when}")
        s1 = _stage("1  Train the model", "Model ready", sub, True)
    else:
        s1 = _stage("1  Train the model", "Not trained yet", "Upload labeled complaints first", False)
    if results_count:
        s2 = _stage("2  Analyze complaints", f"{results_count:,} analyzed", "Results are shown below", True)
        s3 = _stage("3  Export reports", "Excel and PDF ready", "Download from the results", True)
    else:
        s2 = _stage("2  Analyze complaints", "Waiting for a file", "Upload a CSV or Excel file", False)
        s3 = _stage("3  Export reports", "After analysis", "Excel and PDF downloads", False)
    return ('<div class="hero"><div class="hero-title">Customer Complaint Analyzer</div>'
            '<div class="hero-sub">Sort complaints into categories, find the issues that keep coming back, and export a report. '
            'Everything is processed on this computer.</div>'
            f'<div class="stages">{s1}{s2}{s3}</div></div>')


def sidebar_html(status):
    model = "Ready" if status else "Not trained"
    return ('<div class="side-title">Your data stays local</div>'
            '<div class="side-row"><span>Processing</span><b>This computer</b></div>'
            '<div class="side-row"><span>External AI API</span><b>Not used</b></div>'
            f'<div class="side-row"><span>Model</span><b>{model}</b></div>'
            '<div class="side-row"><span>Training columns</span><b>complaint, category</b></div>'
            '<div class="side-note">Local processing lowers exposure but does not replace device security or access controls.</div>')


def category_chart(summary, top_n=15):
    """Horizontal bar chart of complaint counts with value labels."""
    import matplotlib.pyplot as plt
    plot_df = summary.head(top_n).sort_values("count")
    labels = [(s if len(s) <= 46 else s[:45] + "…") for s in plot_df["category"].astype(str)]
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.42 * len(plot_df) + 0.6)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(labels, plot_df["count"], color=TEAL, height=0.62)
    ax.bar_label(bars, padding=4, color=INK, fontsize=9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK, labelsize=9)
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig