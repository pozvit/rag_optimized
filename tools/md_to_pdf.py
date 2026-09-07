"""
Конвертер обезличенных текстовых документов (Markdown) в PDF для базы знаний RAG.

Зачем: исходные материалы были презентациями с графикой, логотипами и контактами.
Для RAG нужен «плоский» текст без графики, без названий компаний и без персональных данных.
Скрипт берёт .md из data/source_md/ и собирает текстовый PDF в data/raw/.

Запуск (из корня проекта):
    python tools/md_to_pdf.py
    python tools/md_to_pdf.py data/source_md data/raw
"""

import re
import sys
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

# Кириллические шрифты. Первый найденный путь выигрывает.
FONT_CANDIDATES = [
    ("DejaVuSans", "DejaVuSans-Bold",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("LiberationSans", "LiberationSans-Bold",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
]


def register_font() -> tuple:
    for regular, bold, reg_path, bold_path in FONT_CANDIDATES:
        if Path(reg_path).exists() and Path(bold_path).exists():
            pdfmetrics.registerFont(TTFont(regular, reg_path))
            pdfmetrics.registerFont(TTFont(bold, bold_path))
            return regular, bold
    raise RuntimeError(
        "Не найден TTF-шрифт с кириллицей. Установите fonts-dejavu-core "
        "или укажите свой путь в FONT_CANDIDATES."
    )


def escape(text: str) -> str:
    """Экранируем спецсимволы XML и переводим **жирный** в <b>."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


def build_styles(regular: str, bold: str) -> dict:
    base = dict(fontName=regular, fontSize=10.5, leading=15, alignment=TA_LEFT)
    return {
        "h1": ParagraphStyle("h1", fontName=bold, fontSize=17, leading=22, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13.5, leading=18, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=11.5, leading=16, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", spaceAfter=6, **base),
        "bullet": ParagraphStyle("bullet", spaceAfter=2, **base),
    }


def md_to_flowables(md_text: str, styles: dict) -> list:
    flow = []
    bullet_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            flow.append(ListFlowable(
                [ListItem(Paragraph(b, styles["bullet"]), leftIndent=14) for b in bullet_buffer],
                bulletType="bullet", bulletFontName=styles["bullet"].fontName,
                bulletFontSize=8, leftIndent=14, spaceAfter=6,
            ))
            bullet_buffer = []

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_bullets()
            continue
        if line.startswith("### "):
            flush_bullets()
            flow.append(Paragraph(escape(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            flush_bullets()
            flow.append(Paragraph(escape(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            flush_bullets()
            flow.append(Paragraph(escape(line[2:]), styles["h1"]))
        elif re.match(r"^\s*[-*]\s+", line):
            bullet_buffer.append(escape(re.sub(r"^\s*[-*]\s+", "", line)))
        elif re.match(r"^\s*\d+\.\s+", line):
            # нумерованный пункт оставляем обычным абзацем с сохранением номера
            flush_bullets()
            flow.append(Paragraph(escape(line.strip()), styles["body"]))
        else:
            flush_bullets()
            flow.append(Paragraph(escape(line.strip()), styles["body"]))

    flush_bullets()
    flow.append(Spacer(1, 4 * mm))
    return flow


def convert(md_path: Path, pdf_path: Path, styles: dict) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", md_text, flags=re.M)
    title = title_match.group(1).strip() if title_match else md_path.stem

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=title, author="Обезличенный материал", subject="RAG knowledge base",
    )
    doc.build(md_to_flowables(md_text, styles))


def main() -> None:
    src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/source_md")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    regular, bold = register_font()
    styles = build_styles(regular, bold)

    md_files = sorted(src_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"В {src_dir} нет .md файлов")

    for md_path in md_files:
        pdf_path = out_dir / (md_path.stem + ".pdf")
        convert(md_path, pdf_path, styles)
        print(f"{md_path.name} -> {pdf_path} ({pdf_path.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
