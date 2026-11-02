import re
import pandas as pd

COMPLAINT_CANDIDATES = ["complaint", "complaint_text", "description", "message", "issue", "customer_complaint", "remarks", "query", "text", "details"]
CATEGORY_CANDIDATES = ["category", "label", "complaint_category", "issue_category", "class", "target", "type", "group"]

def read_table(source):
    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".csv"):
        try:
            if hasattr(source, "seek"): source.seek(0)
            return pd.read_csv(source, encoding="utf-8-sig")
        except UnicodeDecodeError:
            if hasattr(source, "seek"): source.seek(0)
            return pd.read_csv(source, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        if hasattr(source, "seek"): source.seek(0)
        return pd.read_excel(source)
    raise ValueError("Unsupported file type. Use CSV or Excel (.xlsx/.xls).")

def _norm_col(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def _find_column(columns, candidates):
    pairs = [(_norm_col(c), c) for c in columns]
    for candidate in candidates:
        for normalized, original in pairs:
            if normalized == candidate: return original
    for candidate in candidates:
        for normalized, original in pairs:
            if candidate in normalized: return original
    return None

def find_complaint_column(columns): return _find_column(columns, COMPLAINT_CANDIDATES)
def find_category_column(columns): return _find_column(columns, CATEGORY_CANDIDATES)

def normalize_training_frame(df, complaint_col, category_col):
    result = df[[complaint_col, category_col]].copy()
    result.columns = ["complaint", "category"]
    result["complaint"] = result["complaint"].fillna("").astype(str).str.strip()
    result["category"] = result["category"].fillna("").astype(str).str.strip()
    return result

def make_summary(df, category_col="predicted_category"):
    if df.empty or category_col not in df.columns:
        return pd.DataFrame(columns=["category", "count", "percentage"])
    counts = df[category_col].fillna("Uncategorized").astype(str).value_counts()
    result = counts.rename_axis("category").reset_index(name="count")
    total = max(int(result["count"].sum()), 1)
    result["percentage"] = (result["count"] / total * 100).round(2)
    return result.sort_values("count", ascending=False).reset_index(drop=True)
