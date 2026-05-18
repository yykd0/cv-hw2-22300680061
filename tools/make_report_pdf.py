from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "HW2_report.md"
REPORT_PDF = ROOT / "report" / "HW2_report.pdf"


def register_fonts() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("CJK", str(font_path)))
            return "CJK"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def make_styles(font_name: str):
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "CJKNormal",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        firstLineIndent=0,
        spaceAfter=4,
    )
    h1 = ParagraphStyle(
        "CJKHeading1",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#111111"),
        spaceBefore=10,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "CJKHeading2",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#111111"),
        spaceBefore=8,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "CJKHeading3",
        parent=styles["Heading3"],
        fontName=font_name,
        fontSize=12,
        leading=18,
        spaceBefore=6,
        spaceAfter=4,
    )
    bullet = ParagraphStyle(
        "CJKBullet",
        parent=normal,
        leftIndent=12,
        firstLineIndent=-8,
    )
    return normal, h1, h2, h3, bullet


def flush_table(story, rows, font_name: str, normal) -> None:
    if not rows:
        return
    table_data = [[Paragraph(clean_inline(cell.strip()) or " ", normal) for cell in row] for row in rows]
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F2F5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8CDD4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 5))


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "")


def add_image(story, line: str, normal) -> bool:
    match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not match:
        return False
    alt, image_path = match.groups()
    path = Path(image_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        story.append(Paragraph(f"图像未找到：{clean_inline(image_path)}", normal))
        return True
    img = Image(str(path))
    max_w = A4[0] - 36 * mm
    max_h = 105 * mm
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    if alt:
        story.append(Paragraph(clean_inline(alt), normal))
    story.append(img)
    story.append(Spacer(1, 6))
    return True


def parse_markdown(md_text: str, font_name: str):
    normal, h1, h2, h3, bullet = make_styles(font_name)
    story = []
    table_rows = []
    for raw in md_text.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            table_rows.append(cells)
            continue
        flush_table(story, table_rows, font_name, normal)
        table_rows = []
        if not line:
            story.append(Spacer(1, 4))
        elif add_image(story, line, normal):
            pass
        elif line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), h2))
        elif line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), h3))
        elif line.startswith("- "):
            story.append(Paragraph("- " + clean_inline(line[2:]), bullet))
        elif line.startswith("> "):
            quote_style = ParagraphStyle(
                "Quote",
                parent=normal,
                leftIndent=10,
                textColor=colors.HexColor("#666666"),
                backColor=colors.HexColor("#F8F8F8"),
            )
            story.append(Paragraph(clean_inline(line[2:]), quote_style))
        else:
            story.append(Paragraph(clean_inline(line), normal))
    flush_table(story, table_rows, font_name, normal)
    return story


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(200 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def main() -> None:
    font_name = register_fonts()
    story = parse_markdown(REPORT_MD.read_text(encoding="utf-8"), font_name)
    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(REPORT_PDF)


if __name__ == "__main__":
    main()
