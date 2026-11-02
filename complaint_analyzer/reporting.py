import io
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_excel(results, summary, candidate_rows):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Categorized Complaints")
        summary.to_excel(writer, index=False, sheet_name="Category Summary")
        candidate_rows.to_excel(writer, index=False, sheet_name="Candidate Groups")
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                max_len = max((min(len(str(c.value or "")), 55) for c in list(col)[:500]), default=12)
                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 45)
    output.seek(0); return output.getvalue()

def build_pdf(results, summary, candidate_rows, source_name="uploaded file"):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Customer Complaint Analysis Report", styles["Title"]), Spacer(1, 5*mm),
             Paragraph(f"Source file: {source_name}", styles["Normal"]),
             Paragraph(f"Complaints processed: {len(results):,}", styles["Normal"]),
             Paragraph(f"Predicted categories: {results['predicted_category'].nunique() if 'predicted_category' in results else 0:,}", styles["Normal"]),
             Paragraph("Predictions are not verified facts. Review low-confidence cases and candidate groups.", styles["BodyText"]),
             Spacer(1, 5*mm), Paragraph("Category Summary", styles["Heading2"])]
    data = [["Category", "Count", "Percentage"]]
    for _, row in summary.head(25).iterrows():
        data.append([str(row.get("category", ""))[:100], str(row.get("count", "")), f"{row.get('percentage', 0)}%"])
    table = Table(data, repeatRows=1, colWidths=[150*mm, 25*mm, 30*mm])
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), colors.HexColor("#24476B")), ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"), ("GRID", (0,0),(-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#F2F5F8")]), ("VALIGN", (0,0),(-1,-1), "TOP"),
        ("FONTSIZE", (0,0),(-1,-1), 8), ("LEFTPADDING", (0,0),(-1,-1), 5)]))
    story.extend([table, Spacer(1, 6*mm), Paragraph("Review Indicators", styles["Heading2"])])
    review_count = int((results["prediction_status"] != "classified").sum()) if "prediction_status" in results else 0
    story.append(Paragraph(f"Rows flagged for review or missing text: {review_count:,}", styles["Normal"]))
    story.append(Paragraph(f"Rows in exploratory candidate groups: {len(candidate_rows):,}", styles["Normal"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Candidate groups are exploratory text clusters and must be reviewed before being treated as official categories.", styles["BodyText"]))
    doc.build(story); output.seek(0); return output.getvalue()
