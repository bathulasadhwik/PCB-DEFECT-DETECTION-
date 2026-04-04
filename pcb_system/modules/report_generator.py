import os
from datetime import datetime
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


def _safe_str(val):
    if val is None:
        return ""
    return str(val)


def generate_pdf_report(
    output_path: str,
    metadata: Dict,
    defects: List[Dict],
    gemini_text: str,
    repairs: List[Dict],
    original_image_path: str,
    heatmap_image_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontSize=10, leading=13))

    story = []

    story.append(Paragraph("PCB Inspection Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    now = metadata.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_data = [
        ["Timestamp", _safe_str(now)],
        ["Status", _safe_str(metadata.get("status"))],
        ["Risk Level", _safe_str(metadata.get("risk_level"))],
        ["Failure Probability", f"{_safe_str(metadata.get('failure_probability'))}%"],
    ]
    summary_table = Table(summary_data, colWidths=[5 * cm, 10 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Detected Defects", styles["Heading2"]))
    if defects:
        defect_rows = [["#", "Label", "Confidence", "BBox"]]
        for i, d in enumerate(defects, start=1):
            defect_rows.append(
                [
                    str(i),
                    _safe_str(d.get("label")),
                    _safe_str(d.get("confidence")),
                    _safe_str(d.get("bbox")),
                ]
            )
        defect_table = Table(defect_rows, colWidths=[1 * cm, 4 * cm, 3 * cm, 7 * cm])
        defect_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(defect_table)
    else:
        story.append(Paragraph("No defects detected.", styles["BodySmall"]))

    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Gemini AI Explanation", styles["Heading2"]))
    for para in (gemini_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles["BodySmall"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Repair Suggestions", styles["Heading2"]))
    if repairs:
        for item in repairs:
            line = f"- {_safe_str(item.get('defect'))}: {_safe_str(item.get('suggestion'))}"
            story.append(Paragraph(line, styles["BodySmall"]))
    else:
        story.append(Paragraph("No repair required.", styles["BodySmall"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Inspection Images", styles["Heading2"]))
    if original_image_path and os.path.exists(original_image_path):
        story.append(Paragraph("Original Image", styles["BodySmall"]))
        story.append(Image(original_image_path, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if heatmap_image_path and os.path.exists(heatmap_image_path):
        story.append(Paragraph("Heatmap Overlay", styles["BodySmall"]))
        story.append(Image(heatmap_image_path, width=14 * cm, height=7 * cm))

    doc.build(story)
    return output_path
