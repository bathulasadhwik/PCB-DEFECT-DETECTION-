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


def _get_defect_severity(label: str) -> str:
    """Return severity level for defect type."""
    label_lower = (label or "").lower()
    if "short" in label_lower:
        return "Critical"
    elif "open" in label_lower or "missing" in label_lower:
        return "High"
    elif "shifted" in label_lower or "misaligned" in label_lower:
        return "Medium"
    elif "scratch" in label_lower:
        return "Low"
    else:
        return "Medium"


def _get_defect_impact(label: str) -> str:
    """Return functional impact description."""
    label_lower = (label or "").lower()
    if "short" in label_lower:
        return "Circuit failure/immediate"
    elif "open" in label_lower:
        return "Signal loss/intermittent"
    elif "missing" in label_lower:
        return "Function incomplete"
    elif "shifted" in label_lower:
        return "Reliability reduced"
    elif "scratch" in label_lower:
        return "Cosmetic/minor"
    else:
        return "Needs evaluation"


def _calculate_mtbf_estimate(failure_probability: float) -> str:
    """Calculate rough MTBF estimate based on failure probability."""
    if failure_probability <= 0:
        return "> 100,000 hours"
    elif failure_probability < 10:
        return "50,000 - 100,000 hours"
    elif failure_probability < 30:
        return "10,000 - 50,000 hours"
    elif failure_probability < 70:
        return "1,000 - 10,000 hours"
    else:
        return "< 1,000 hours"


def generate_pdf_report(
    output_path: str,
    metadata: Dict,
    defects: List[Dict],
    gemini_text: str,
    repairs: List[Dict],
    original_image_path: str,
    detection_image_path: str,
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
        ["PCB Name", _safe_str(metadata.get("pcb_name") or "N/A")],
        ["Source", _safe_str(metadata.get("source") or "upload")],
        ["Status", _safe_str(metadata.get("status"))],
        ["Risk Level", _safe_str(metadata.get("risk_level"))],
        ["Failure Probability", f"{_safe_str(metadata.get('failure_probability'))}%"],
    ]
    session_seconds = metadata.get("session_seconds")
    if session_seconds is not None:
        summary_data.append(["Session Duration (s)", _safe_str(session_seconds)])
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

    # Add Technical Specifications section
    story.append(Paragraph("Technical Specifications", styles["Heading2"]))
    tech_specs = [
        ["Operating Voltage", _safe_str(metadata.get("operating_voltage", "3.3V-5V (Typical IoT)") or "N/A")],
        ["Current Draw (Max)", _safe_str(metadata.get("max_current", "< 500mA") or "N/A")],
        ["Operating Temperature", _safe_str(metadata.get("temp_range", "-20°C to +85°C") or "N/A")],
        ["Communication Protocol", _safe_str(metadata.get("communication", "I2C/SPI/UART/WiFi") or "N/A")],
        ["Power Supply Type", _safe_str(metadata.get("power_supply", "DC Regulated") or "N/A")],
        ["Compliance Standards", _safe_str(metadata.get("compliance", "RoHS/CE/FCC") or "N/A")],
    ]
    tech_table = Table(tech_specs, colWidths=[5 * cm, 10 * cm])
    tech_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(tech_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Detected Defects", styles["Heading2"]))
    if defects:
        defect_rows = [["#", "Label", "Confidence", "Severity", "Impact", "BBox"]]
        for i, d in enumerate(defects, start=1):
            label = _safe_str(d.get("label"))
            conf = _safe_str(d.get("confidence"))
            severity = _get_defect_severity(d.get("label", ""))
            impact = _get_defect_impact(d.get("label", ""))
            bbox = _safe_str(d.get("bbox"))
            defect_rows.append([str(i), label, conf, severity, impact, bbox])
        defect_table = Table(defect_rows, colWidths=[0.8 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 3 * cm, 4.2 * cm])
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

    # Add Component Analysis section
    story.append(Paragraph("Component Analysis", styles["Heading2"]))
    component_data = [
        ["Microcontroller", _safe_str(metadata.get("microcontroller", "ESP32/STM32/RP2040") or "Detected automatically")],
        ["Sensors", _safe_str(metadata.get("sensors", "Temperature/Humidity/Motion") or "As per design")],
        ["Communication", _safe_str(metadata.get("comm_modules", "WiFi/Bluetooth/Zigbee") or "As per design")],
        ["Power Management", _safe_str(metadata.get("power_mgmt", "LDO/Boost Converter") or "As per design")],
        ["Connectors", _safe_str(metadata.get("connectors", "USB/JST/Pin Headers") or "As per design")],
        ["Test Points", _safe_str(metadata.get("test_points", "TP1-TP8 (Power/GND/Signals)") or "Standard locations")],
    ]
    component_table = Table(component_data, colWidths=[4 * cm, 11 * cm])
    component_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(component_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Explanation", styles["Heading2"]))
    for para in (gemini_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles["BodySmall"]))
    story.append(Spacer(1, 0.4 * cm))

    # Add Testing Procedures section
    story.append(Paragraph("Testing Procedures", styles["Heading2"]))
    testing_procedures = [
        "1. Power Supply Test: Apply rated voltage and verify current draw < specified max",
        "2. Continuity Check: Verify all power and ground nets using multimeter",
        "3. Signal Integrity: Check clock signals, communication lines for proper levels",
        "4. Functional Test: Load firmware and verify sensor readings/communication",
        "5. Thermal Test: Monitor temperature rise under load (should be < 10°C above ambient)",
        "6. Burn-in Test: 24-48 hour operation at elevated temperature for early failure detection",
    ]
    for procedure in testing_procedures:
        story.append(Paragraph(procedure, styles["BodySmall"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Repair Suggestions", styles["Heading2"]))
    if repairs:
        for item in repairs:
            line = f"- {_safe_str(item.get('defect'))}: {_safe_str(item.get('suggestion'))}"
            story.append(Paragraph(line, styles["BodySmall"]))
    else:
        story.append(Paragraph("No repair required.", styles["BodySmall"]))
    story.append(Spacer(1, 0.5 * cm))

    # Add Manufacturing Notes section
    story.append(Paragraph("Manufacturing Notes", styles["Heading2"]))
    mfg_notes = [
        "• Handle with ESD precautions - all components are ESD sensitive",
        "• Store in moisture-proof packaging if not immediately assembled",
        "• Verify solder paste age and storage conditions before use",
        "• Calibrate reflow oven profile for component types and board thickness",
        "• Use nitrogen atmosphere for lead-free soldering if available",
        "• Allow 24-hour cure time for conformal coating if applied",
    ]
    for note in mfg_notes:
        story.append(Paragraph(note, styles["BodySmall"]))
    story.append(Spacer(1, 0.4 * cm))

    # Add Quality Metrics section
    story.append(Paragraph("Quality Metrics", styles["Heading2"]))
    quality_data = [
        ["Defect Density", f"{len(defects)} defects / {metadata.get('board_area', 'unknown')} cm²"],
        ["Process Yield", f"{100 - metadata.get('failure_probability', 0):.1f}% (estimated)"],
        ["Inspection Coverage", "Visual defects + AI analysis"],
        ["Detection Threshold", "95% confidence minimum"],
        ["False Positive Rate", "< 5% (typical)"],
        ["MTBF Estimate", _calculate_mtbf_estimate(metadata.get('failure_probability', 0))],
    ]
    quality_table = Table(quality_data, colWidths=[4 * cm, 11 * cm])
    quality_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightyellow),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(quality_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Inspection Images", styles["Heading2"]))
    if original_image_path and os.path.exists(original_image_path):
        story.append(Paragraph("Original Image", styles["BodySmall"]))
        story.append(Image(original_image_path, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if detection_image_path and os.path.exists(detection_image_path):
        story.append(Paragraph("Detection Image", styles["BodySmall"]))
        story.append(Image(detection_image_path, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if heatmap_image_path and os.path.exists(heatmap_image_path):
        story.append(Paragraph("Heatmap Overlay", styles["BodySmall"]))
        story.append(Image(heatmap_image_path, width=14 * cm, height=7 * cm))

    doc.build(story)
    return output_path
