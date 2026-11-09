import pandas as pd
from complaint_analyzer.data_utils import find_complaint_column, find_category_column, make_summary

def test_find_columns():
    cols = ["Complaint Text", "Category", "extra"]
    assert find_complaint_column(cols) == "Complaint Text"
    assert find_category_column(cols) == "Category"

def test_summary_counts():
    df = pd.DataFrame({"predicted_category": ["A", "A", "B"]})
    summary = make_summary(df)
    assert int(summary["count"].sum()) == 3
    assert set(summary["category"]) == {"A", "B"}
