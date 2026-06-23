"""Generate a compact Chinese PDF report for HW3 Q1.

The final submitted report is the combined Word/PDF report. This script keeps
this public repository reproducible by producing a standalone Q1 PDF from the
local manifest and submission links.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "HW3_Q1_report.pdf"
CONFIG = ROOT / "configs" / "report_config.json"
LINKS = ROOT / "configs" / "submission_links.json"
MANIFEST = ROOT / "results" / "manifest.json"


def load_json(path: Path, default: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def register_fonts() -> str:
    for name in ["simsun.ttc", "simsun.ttf"]:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            pdfmetrics.registerFont(TTFont("SimSun", str(path), subfontIndex=0 if name.endswith(".ttc") else 0))
            return "SimSun"
    return "Helvetica"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font = register_fonts()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, firstLineIndent=21)
    head = ParagraphStyle("HeadCN", parent=styles["Heading1"], fontName=font, fontSize=14, leading=20)
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font, fontSize=16, leading=24)

    config = load_json(CONFIG, {})
    links = load_json(LINKS, {})
    manifest = load_json(MANIFEST, {"artifacts": []})

    story = [Paragraph("计算机视觉 HW3 题目一：2DGS 与 AIGC 三维资产生成", title)]
    story.append(Paragraph(f"作者：{config.get('author', 'Deng Kaiyuan')}（{config.get('student_id', '22300680061')}）", body))
    story.append(Paragraph(f"GitHub：{links.get('github_repo', '')}", body))
    story.append(Paragraph(f"权重与视频：{links.get('drive_folder', '')}", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("方法与实验设置", head))
    story.append(Paragraph("对象 A 使用 Mip-NeRF 360 bicycle 和 2D Gaussian Splatting 进行多视图重建；对象 B 使用 threestudio DreamFusion/SDS 从文本生成青花瓷茶壶；对象 C 使用 Magic123 从 AIGC 单图生成红色陶瓷龙摆件。", body))

    rows = [["对象", "方法", "输入", "输出证据"],
            ["A", "2D Gaussian Splatting", "Mip-NeRF 360 bicycle", "ply、novel-view 视频、PSNR/L1 指标"],
            ["B", "threestudio SDS", "porcelain teapot prompt", "checkpoint、turntable video"],
            ["C", "Magic123", "AIGC dragon image", "coarse checkpoint、OBJ、video"]]
    table = Table(rows, colWidths=[1.4*cm, 4.0*cm, 4.6*cm, 6.0*cm])
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("FONT", (0,0), (-1,-1), font, 8)]))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("产物状态", head))
    rows = [["key", "status", "paths"]]
    for item in manifest.get("artifacts", []):
        rows.append([item.get("key", ""), item.get("status", ""), "\n".join(item.get("paths", [])[:4])])
    table = Table(rows, colWidths=[5.0*cm, 2.0*cm, 9.0*cm])
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("FONT", (0,0), (-1,-1), font, 7)]))
    story.append(table)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
