#!/usr/bin/env python3
import argparse
import base64
import html
import json
import re
from pathlib import Path

from export_notebook_outputs import TableParser, data_text, output_text

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NOISE_PATTERNS = (
    "Libraries loaded.",
    "Retrieving folder contents",
    "Processing file ",
    "Downloading...",
    "From (original):",
    "From (redirected):",
    "To: /",
    "100%|",
    "Download completed",
    "Unzipping ",
    "Done. Contents:",
    "Saved: fig",
    "DeprecationWarning:",
)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtle",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#1d3557"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#264653"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading3Custom",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=6,
            spaceAfter=5,
            textColor=colors.HexColor("#355070"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Quote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=13,
            leftIndent=18,
            rightIndent=10,
            textColor=colors.HexColor("#4f4f4f"),
            borderPadding=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MonoBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=10.5,
            leftIndent=10,
            rightIndent=10,
            backColor=colors.HexColor("#f5f5f5"),
            borderPadding=5,
            spaceAfter=8,
        )
    )
    return styles


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\$\$([^$]+)\$\$", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\$([^$]+)\$", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def markdown_to_story(text: str, story: list, styles):
    lines = text.splitlines()
    idx = 0
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            content = " ".join(line.strip() for line in paragraph_buffer).strip()
            if content:
                story.append(Paragraph(inline_markup(content), styles["Body"]))
            paragraph_buffer = []

    while idx < len(lines):
        raw = lines[idx]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            content = inline_markup(heading.group(2).strip())
            style_name = {
                1: "Heading1Custom",
                2: "Heading2Custom",
                3: "Heading3Custom",
            }.get(level, "Body")
            story.append(Paragraph(content, styles[style_name]))
            idx += 1
            continue

        if set(stripped) <= {"-"} and len(stripped) >= 3:
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#bcbcbc")))
            story.append(Spacer(1, 0.12 * inch))
            idx += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while idx < len(lines) and lines[idx].strip().startswith(">"):
                quote_lines.append(lines[idx].strip()[1:].strip())
                idx += 1
            story.append(Paragraph(inline_markup(" ".join(quote_lines)), styles["Quote"]))
            continue

        if stripped == "$$":
            flush_paragraph()
            idx += 1
            math_lines = []
            while idx < len(lines) and lines[idx].strip() != "$$":
                math_lines.append(lines[idx].rstrip())
                idx += 1
            story.append(Preformatted("\n".join(math_lines), styles["MonoBlock"]))
            if idx < len(lines) and lines[idx].strip() == "$$":
                idx += 1
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet:
            flush_paragraph()
            bullet_lines = []
            while idx < len(lines):
                match = re.match(r"^[-*+]\s+(.*)$", lines[idx].strip())
                if not match:
                    break
                bullet_lines.append(match.group(1))
                idx += 1
            for item in bullet_lines:
                story.append(Paragraph(f"• {inline_markup(item)}", styles["Body"]))
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            numbered_lines = []
            while idx < len(lines):
                match = re.match(r"^(\d+)\.\s+(.*)$", lines[idx].strip())
                if not match:
                    break
                numbered_lines.append((match.group(1), match.group(2)))
                idx += 1
            for number, item in numbered_lines:
                story.append(Paragraph(f"{number}. {inline_markup(item)}", styles["Body"]))
            continue

        paragraph_buffer.append(stripped)
        idx += 1

    flush_paragraph()


def is_meaningful_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not any(pattern in stripped for pattern in NOISE_PATTERNS)


def merge_header_rows(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    header = header_rows[0][:]
    for extra_row in header_rows[1:]:
        for index, value in enumerate(extra_row):
            if value:
                if index >= len(header):
                    header.extend([""] * (index + 1 - len(header)))
                header[index] = value
    return header


def html_table_to_flowables(html_text: str, styles, max_width: float):
    parser = TableParser()
    parser.feed(html_text)

    header = merge_header_rows(parser.header_rows)
    body = parser.body_rows[:]
    if not header and body:
        header = body.pop(0)
    if not header:
        return []

    rows = [header] + body
    max_cols = max(len(row) for row in rows)
    normalized = []
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        normalized.append([Paragraph(inline_markup(cell), styles["Body"]) for cell in padded[:max_cols]])

    col_width = max_width / max_cols
    table = Table(normalized, colWidths=[col_width] * max_cols, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ddebf7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f1f1f")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    flowables = []
    caption = " ".join(parser.caption.split())
    if caption:
        flowables.append(Paragraph(inline_markup(caption), styles["Caption"]))
    flowables.append(table)
    flowables.append(Spacer(1, 0.12 * inch))
    return flowables


def add_output_to_story(output: dict, story: list, styles, assets_dir: Path, image_index: int, max_width: float):
    output_type = output.get("output_type")

    if output_type == "stream":
        text = output_text(output)
        if is_meaningful_text(text):
            story.append(Preformatted(text.strip("\n"), styles["MonoBlock"]))
        return image_index

    data = output.get("data", {})

    if "text/html" in data:
        story.extend(html_table_to_flowables(data_text(data["text/html"]), styles, max_width))

    if "image/png" in data:
        image_index += 1
        image_name = f"figure_{image_index:02d}.png"
        image_path = assets_dir / image_name
        image_path.write_bytes(base64.b64decode(data_text(data["image/png"])))
        image = Image(str(image_path))
        image.drawHeight = image.drawHeight * min(1.0, max_width / image.drawWidth)
        image.drawWidth = image.drawWidth * min(1.0, max_width / image.drawWidth)
        story.append(KeepTogether([image, Spacer(1, 0.08 * inch)]))

    elif "text/plain" in data:
        text = data_text(data["text/plain"])
        if is_meaningful_text(text):
            story.append(Preformatted(text.strip("\n"), styles["MonoBlock"]))

    return image_index


def add_page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, str(doc.page))


def build_pdf(notebook_path: Path, pdf_path: Path):
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    assets_dir = pdf_path.with_suffix("")
    assets_dir.mkdir(parents=True, exist_ok=True)

    styles = build_styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=notebook_path.stem.replace("_", " "),
    )

    story = [
        Paragraph("SEARec Data Analysis", styles["TitleCenter"]),
        Paragraph("Notebook commentary and results only", styles["Subtle"]),
        Spacer(1, 0.12 * inch),
    ]

    image_index = 0
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            markdown_to_story("".join(cell.get("source", [])), story, styles)
            story.append(Spacer(1, 0.03 * inch))
            continue

        if cell_type != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        for output in outputs:
            image_index = add_output_to_story(
                output,
                story,
                styles,
                assets_dir,
                image_index,
                doc.width,
            )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return pdf_path, assets_dir, image_index


def main():
    parser = argparse.ArgumentParser(
        description="Create a PDF report from a notebook without code cells."
    )
    parser.add_argument("notebook", type=Path, help="Path to the input .ipynb notebook")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output PDF file (default: <notebook_stem>_comments_results.pdf)",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    pdf_path = args.output or notebook_path.with_name(
        f"{notebook_path.stem}_comments_results.pdf"
    )

    pdf_path, assets_dir, image_count = build_pdf(notebook_path, pdf_path)
    print(f"PDF saved to: {pdf_path}")
    print(f"Assets saved to: {assets_dir}")
    print(f"Embedded {image_count} figures.")


if __name__ == "__main__":
    main()
