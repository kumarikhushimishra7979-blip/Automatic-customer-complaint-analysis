
import re
import pandas as pd


# --------------------------------------------------
# 1. POSSIBLE COLUMN NAMES
# --------------------------------------------------

COMPLAINT_CANDIDATES = [
    "complaint",
    "complaint_text",
    "description",
    "message",
    "issue",
    "customer_complaint",
    "remarks",
    "query",
    "text",
    "details"
]

CATEGORY_CANDIDATES = [
    "category",
    "label",
    "complaint_category",
    "issue_category",
    "class",
    "target",
    "type",
    "group"
]


# --------------------------------------------------
# 2. READ EXCEL OR CSV FILE
# --------------------------------------------------

def read_table(source):

    name = getattr(source, "name", str(source)).lower()

    try:

        if hasattr(source, "seek"):
            source.seek(0)

        if name.endswith(".csv"):

            try:
                df = pd.read_csv(
                    source,
                    encoding="utf-8-sig"
                )

            except UnicodeDecodeError:

                if hasattr(source, "seek"):
                    source.seek(0)

                df = pd.read_csv(
                    source,
                    encoding="latin-1"
                )

        elif name.endswith((".xlsx", ".xls")):

            df = pd.read_excel(source)

        else:

            raise ValueError(
                "Unsupported file type. "
                "Please upload CSV or Excel files."
            )

        if df.empty:
            raise ValueError(
                "The uploaded file contains no data."
            )

        if len(df.columns) == 0:
            raise ValueError(
                "No columns were found in the uploaded file."
            )

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        if df.empty or len(df.columns) == 0:
            raise ValueError(
                "The uploaded file contains no usable data."
            )

        return df

    except Exception as error:

        raise ValueError(
            f"Unable to read the uploaded file: {error}"
        ) from error


# --------------------------------------------------
# 3. NORMALIZE COLUMN NAMES
# --------------------------------------------------

def _norm_col(value):

    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value
    )

    return value.strip("_")


# --------------------------------------------------
# 4. IDENTIFY REQUIRED COLUMNS
# --------------------------------------------------

def _find_column(columns, candidates):

    pairs = [
        (_norm_col(column), column)
        for column in columns
    ]

    # First preference: exact match
    for candidate in candidates:

        for normalized, original in pairs:

            if normalized == candidate:
                return original

    # Second preference: partial match
    for candidate in candidates:

        for normalized, original in pairs:

            if candidate in normalized:
                return original

    return None


def find_complaint_column(columns):

    return _find_column(
        columns,
        COMPLAINT_CANDIDATES
    )


def find_category_column(columns):

    return _find_column(
        columns,
        CATEGORY_CANDIDATES
    )


# --------------------------------------------------
# 5. PREPARE TRAINING DATA
# --------------------------------------------------

def normalize_training_frame(
    df,
    complaint_col,
    category_col
):

    if complaint_col not in df.columns:
        raise ValueError(
            "Complaint column does not exist."
        )

    if category_col not in df.columns:
        raise ValueError(
            "Category column does not exist."
        )

    if complaint_col == category_col:
        raise ValueError(
            "Complaint and category columns "
            "must be different."
        )

    result = df[
        [complaint_col, category_col]
    ].copy()

    result.columns = [
        "complaint",
        "category"
    ]

    result["complaint"] = (
        result["complaint"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["category"] = (
        result["category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return result


# --------------------------------------------------
# 6. GENERATE CATEGORY SUMMARY
# --------------------------------------------------

def make_summary(
    df,
    category_col="predicted_category"
):

    required_columns = [
        "category",
        "count",
        "percentage",
        "rank"
    ]

    if df.empty or category_col not in df.columns:

        return pd.DataFrame(
            columns=required_columns
        )

    categories = (
        df[category_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Uncategorized")
    )

    counts = categories.value_counts()

    summary = (
        counts
        .rename_axis("category")
        .reset_index(name="count")
    )

    total = int(summary["count"].sum())

    if total == 0:
        return pd.DataFrame(
            columns=required_columns
        )

    # Calculate percentage
    summary["percentage"] = (
        summary["count"] / total * 100
    ).round(2)

    # Assign category rank
    summary["rank"] = (
        summary["count"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Sort from highest to lowest
    summary = summary.sort_values(
        by=["count", "category"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary