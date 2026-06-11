from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def inline(text: str) -> str:
    text = escape(text).replace("`", "")
    while "**" in text:
        text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return text


def add_table(story, rows, styles) -> None:
    clean = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        clean.append([Paragraph(inline(cell), styles["Bodyx"]) for cell in cells])
    if len(clean) < 2:
        return
    widths = [6.8 * inch / len(clean[0])] * len(clean[0])
    table = Table(clean, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#233142")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d1dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 5))


def build(output_path: Path) -> Path:
    lines = Path("reports/TECHNICAL_REPORT.md").read_text(encoding="utf-8").splitlines()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bulletx", parent=styles["BodyText"], fontSize=8.3, leading=10.5, leftIndent=12, firstLineIndent=-7, spaceAfter=2))
    styles.add(ParagraphStyle(name="Codex", parent=styles["Code"], fontSize=7.5, leading=9, backColor=colors.HexColor("#f4f6f8")))

    story, table_lines, code_lines = [], [], []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(escape(item) for item in code_lines), styles["Codex"]))
                story.append(Spacer(1, 4))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            if set(stripped.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                continue
            table_lines.append(stripped)
            continue
        if table_lines:
            add_table(story, table_lines, styles)
            table_lines = []
        if not stripped:
            story.append(Spacer(1, 3))
        elif stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), styles["ReportTitle"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), styles["H2x"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), styles["H3x"]))
        elif stripped.startswith("- "):
            story.append(Paragraph("• " + inline(stripped[2:]), styles["Bulletx"]))
        elif stripped.startswith("> "):
            story.append(Paragraph("<i>" + inline(stripped[2:]) + "</i>", styles["Bodyx"]))
        else:
            story.append(Paragraph(inline(stripped), styles["Bodyx"]))
    if table_lines:
        add_table(story, table_lines, styles)

    for figure in ["outputs/figures/phase1_risk_return.png", "outputs/figures/phase1_correlation_heatmap.png"]:
        if Path(figure).exists():
            story.append(PageBreak())
            story.append(Paragraph(Path(figure).stem.replace("_", " ").title(), styles["H2x"]))
            story.append(Image(figure, width=6.5 * inch, height=3.7 * inch))
    doc.build(story)
    return output_path


if __name__ == "__main__":
    try:
        result = build(Path("reports/technical_report.pdf"))
    except PermissionError:
        result = build(Path("reports/technical_report_updated.pdf"))
    print(result.resolve())
