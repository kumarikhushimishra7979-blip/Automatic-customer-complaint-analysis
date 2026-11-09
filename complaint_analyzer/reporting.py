import io
import math
import re

import pandas as pd
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.utils import get_column_letter

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as ReportImage, KeepTogether
)


# Report palette based on the supplied reference screenshot.
NAVY = "173653"
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "F2F6FA"
MID_BLUE = "2F75B5"
WHITE = "FFFFFF"
DARK = "1F2937"
GREY = "667085"


def _clean_summary(summary):
    """Return a consistently named, ranked summary without changing the input."""
    if summary is None or summary.empty:
        return pd.DataFrame(columns=["category", "count", "percentage", "rank"])

    frame = summary.copy()

    # Accept the common column names used by the app and older summaries.
    rename = {}
    if "category" not in frame.columns:
        for candidate in ("predicted_category", "Issue Type", "issue_type", "COMPLAINT_SUBTYPE"):
            if candidate in frame.columns:
                rename[candidate] = "category"
                break
    if "count" not in frame.columns:
        for candidate in ("Complaint Count", "complaint_count", "frequency"):
            if candidate in frame.columns:
                rename[candidate] = "count"
                break
    if "percentage" not in frame.columns:
        for candidate in ("Percentage", "percent", "share"):
            if candidate in frame.columns:
                rename[candidate] = "percentage"
                break

    frame = frame.rename(columns=rename)

    if "category" not in frame.columns:
        raise ValueError("The summary must contain a category column.")
    if "count" not in frame.columns:
        raise ValueError("The summary must contain a count column.")

    frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0)
    total = float(frame["count"].sum())

    if "percentage" not in frame.columns:
        frame["percentage"] = (frame["count"] / total * 100) if total else 0
    else:
        frame["percentage"] = pd.to_numeric(
            frame["percentage"], errors="coerce"
        ).fillna(0)

    # Store percentages consistently as 0–100 values for display.
    if not frame.empty and frame["percentage"].max() <= 1:
        frame["percentage"] = frame["percentage"] * 100

    frame = frame.sort_values("count", ascending=False).reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame[["category", "count", "percentage", "rank"]]


def _safe_sheet_name(name):
    name = re.sub(r"[\[\]:*?/\\]", "_", str(name))
    return name[:31] or "Report"


def _style_data_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(name="Aptos", bold=True, color=WHITE)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 28

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if cell.row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=PALE_BLUE)

    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_len = max(
            (min(len(str(c.value or "")), 55) for c in list(column_cells)[:500]),
            default=12
        )
        ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 45)


def build_excel(results, summary, candidate_rows):
    """
    Create a formatted Excel workbook with a dashboard-style first sheet,
    detailed categorized complaints, category summary, and candidate groups.
    """
    summary_frame = _clean_summary(summary)
    results = results.copy() if results is not None else pd.DataFrame()
    candidate_rows = (
        candidate_rows.copy()
        if candidate_rows is not None
        else pd.DataFrame()
    )

    total = int(summary_frame["count"].sum()) if not summary_frame.empty else len(results)
    top3 = summary_frame.head(3)
    top3_count = int(top3["count"].sum()) if not top3.empty else 0
    top3_share = (top3_count / total * 100) if total else 0

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Dashboard-style report sheet.
        report_sheet = "Complaint Report"
        summary_frame.to_excel(
            writer, index=False, sheet_name=report_sheet, startrow=8, startcol=0
        )
        ws = writer.book[report_sheet]
        ws.sheet_view.showGridLines = False

        ws.merge_cells("A1:D1")
        ws["A1"] = "Customer Complaint Categorization Report"
        ws["A1"].font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
        ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 38

        ws.merge_cells("A2:D2")
        ws["A2"] = f"Based on {total:,} non-empty complaint entries"
        ws["A2"].font = Font(name="Aptos", size=11, italic=True, color=MID_BLUE)
        ws["A2"].alignment = Alignment(horizontal="center")

        metrics = [
            ("Total complaints", total),
            ("Top 3 categories (count)", top3_count),
            ("Top 3 categories (share)", top3_share / 100),
        ]
        for row_num, (label, value) in enumerate(metrics, start=4):
            ws.cell(row_num, 1, label)
            ws.cell(row_num, 2, value)
            for cell in ws[row_num][:2]:
                cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
            ws.cell(row_num, 1).font = Font(bold=True, color=NAVY)
            ws.cell(row_num, 1).alignment = Alignment(vertical="center")
            ws.cell(row_num, 2).alignment = Alignment(horizontal="right")
            ws.row_dimensions[row_num].height = 23
        ws["B6"].number_format = "0.0%"

        # Table header and body.
        header_row = 9
        for cell in ws[header_row]:
            cell.fill = PatternFill("solid", fgColor=NAVY)
            cell.font = Font(bold=True, color=WHITE)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[header_row].height = 25

        thin_blue = Side(style="thin", color="C8D8E8")
        for row in ws.iter_rows(min_row=header_row + 1, max_row=header_row + len(summary_frame), min_col=1, max_col=4):
            for cell in row:
                cell.border = Border(bottom=thin_blue)
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                if cell.row % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor=PALE_BLUE)
            row[2].number_format = '0.0"%"'
            row[1].number_format = '#,##0'
            row[3].alignment = Alignment(horizontal="center")

        total_row = header_row + len(summary_frame) + 1
        ws.cell(total_row, 1, "TOTAL")
        ws.cell(total_row, 2, total)
        ws.cell(total_row, 3, 100.0)
        for cell in ws[total_row][:4]:
            cell.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
            cell.font = Font(bold=True, color=NAVY)
            cell.border = Border(top=Side(style="medium", color=NAVY))
        ws.cell(total_row, 3).number_format = '0.0"%"'

        ws.column_dimensions["A"].width = 62
        ws.column_dimensions["B"].width = 20
        ws.column_dimensions["C"].width = 16
        ws.column_dimensions["D"].width = 12
        ws.freeze_panes = "A10"
        ws.auto_filter.ref = f"A9:D{header_row + len(summary_frame)}"

        # Horizontal bar chart, with largest categories at the bottom as in Excel bar charts.
        if not summary_frame.empty:
            chart = BarChart()
            chart.type = "bar"
            chart.style = 10
            chart.title = "Complaints by Issue Type"
            chart.y_axis.title = "Issue Type"
            chart.x_axis.title = "Number of Complaints"
            chart.height = 10
            chart.width = 19
            chart.legend = None
            chart.varyColors = True
            data_ref = Reference(
                ws, min_col=2, min_row=header_row, max_row=header_row + len(summary_frame)
            )
            cats_ref = Reference(
                ws, min_col=1, min_row=header_row + 1,
                max_row=header_row + len(summary_frame)
            )
            chart.add_data(data_ref, titles_from_data=True)
            chart.set_categories(cats_ref)
            chart.dataLabels = DataLabelList()
            chart.dataLabels.showVal = True
            chart.height = max(8, min(16, len(summary_frame) * 0.55))
            chart.width = 19
            ws.add_chart(chart, "F4")

        # Detail sheets.
        results.to_excel(writer, index=False, sheet_name="Categorized Complaints")
        summary_frame.to_excel(writer, index=False, sheet_name="Category Summary")
        candidate_rows.to_excel(writer, index=False, sheet_name="Candidate Groups")

        for sheet_name in ("Categorized Complaints", "Category Summary", "Candidate Groups"):
            _style_data_sheet(writer.book[sheet_name])

    output.seek(0)
    return output.getvalue()


def _make_pdf_chart(summary_frame):
    """Render a compact horizontal bar chart for embedding in the PDF."""
    if summary_frame.empty:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_frame = summary_frame.sort_values("count", ascending=True).tail(20)
    fig_height = max(3.2, min(8.0, len(plot_frame) * 0.32))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(plot_frame["category"].astype(str), plot_frame["count"])
    ax.set_xlabel("Number of Complaints")
    ax.set_ylabel("Issue Type")
    ax.set_title("Complaints by Issue Type", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    image_buffer.seek(0)
    return image_buffer


def build_pdf(results, summary, candidate_rows, source_name="uploaded file"):
    """Create a landscape PDF report with KPIs, ranked summary, and chart."""
    summary_frame = _clean_summary(summary)
    results = results if results is not None else pd.DataFrame()
    candidate_rows = candidate_rows if candidate_rows is not None else pd.DataFrame()

    total = int(summary_frame["count"].sum()) if not summary_frame.empty else len(results)
    top3_count = int(summary_frame.head(3)["count"].sum()) if not summary_frame.empty else 0
    top3_share = (top3_count / total * 100) if total else 0
    predicted_count = (
        int(results["predicted_category"].nunique())
        if "predicted_category" in results else 0
    )
    review_count = (
        int((results["prediction_status"] != "classified").sum())
        if "prediction_status" in results else 0
    )

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4),
        rightMargin=13 * mm, leftMargin=13 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#173653"),
        fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=4
    )
    small_style = ParagraphStyle(
        "SmallNote", parent=styles["Normal"], textColor=colors.HexColor("#526579"),
        fontSize=8, leading=11
    )

    story = [
        Paragraph("Customer Complaint Categorization Report", title_style),
        Paragraph(f"Source file: {source_name}", small_style),
        Spacer(1, 4 * mm),
    ]

    kpi_data = [
        ["Total complaints", "Top 3 categories (count)", "Top 3 categories (share)", "Predicted categories"],
        [f"{total:,}", f"{top3_count:,}", f"{top3_share:.1f}%", f"{predicted_count:,}"],
    ]
    kpi_table = Table(kpi_data, colWidths=[62 * mm] * 4, rowHeights=[10 * mm, 14 * mm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173653")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, 1), 15),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B7CDE0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7CDE0")),
    ]))
    story.extend([kpi_table, Spacer(1, 5 * mm)])

    story.append(Paragraph("Category Breakdown", styles["Heading2"]))
    table_data = [["Issue Type", "Complaint Count", "Percentage", "Rank"]]
    for _, row in summary_frame.head(30).iterrows():
        table_data.append([
            Paragraph(str(row["category"])[:140], small_style),
            f"{int(row['count']):,}",
            f"{row['percentage']:.1f}%",
            str(int(row["rank"])),
        ])

    category_table = Table(
        table_data, repeatRows=1,
        colWidths=[155 * mm, 32 * mm, 27 * mm, 18 * mm]
    )
    category_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173653")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D8E8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([category_table, Spacer(1, 5 * mm)])

    chart_buffer = _make_pdf_chart(summary_frame)
    if chart_buffer is not None:
        story.extend([
            Paragraph("Complaints by Issue Type", styles["Heading2"]),
            ReportImage(chart_buffer, width=245 * mm, height=95 * mm, kind="proportional"),
            Spacer(1, 3 * mm),
        ])

    story.extend([
        Paragraph("Review Indicators", styles["Heading2"]),
        Paragraph(
            f"Rows flagged for review or missing text: {review_count:,}<br/>"
            f"Rows in exploratory candidate groups: {len(candidate_rows):,}",
            styles["Normal"]
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            "Predictions are not verified facts. Low-confidence predictions and candidate groups "
            "must be reviewed before being treated as official categories.",
            small_style
        ),
    ])

    doc.build(story)
    output.seek(0)
    return output.getvalue()
