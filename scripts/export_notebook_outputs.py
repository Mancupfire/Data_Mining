#!/usr/bin/env python3
import argparse
import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_caption = False
        self.in_row = False
        self.cell_tag = None
        self.current_cell = []
        self.current_row = []
        self.caption = ""
        self.header_rows = []
        self.body_rows = []
        self.section = None

    def handle_starttag(self, tag, attrs):
        if tag == "caption":
            self.in_caption = True
        elif tag in {"thead", "tbody"}:
            self.section = tag
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"th", "td"} and self.in_row:
            self.cell_tag = tag
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "caption":
            self.in_caption = False
        elif tag in {"thead", "tbody"}:
            self.section = None
        elif tag in {"th", "td"} and self.cell_tag == tag:
            text = "".join(self.current_cell).strip()
            self.current_row.append((tag, " ".join(text.split())))
            self.cell_tag = None
            self.current_cell = []
        elif tag == "tr" and self.in_row:
            cells = [text for _, text in self.current_row]
            if any(cells):
                if self.section == "thead":
                    self.header_rows.append(cells)
                else:
                    self.body_rows.append(cells)
            self.in_row = False
            self.current_row = []

    def handle_data(self, data):
        if self.in_caption:
            self.caption += data
        elif self.cell_tag:
            self.current_cell.append(data)


def markdown_escape(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", "<br>")


def html_table_to_markdown(html: str) -> str:
    parser = TableParser()
    parser.feed(html)

    header = None
    if parser.header_rows:
        header = parser.header_rows[0][:]
        if len(parser.header_rows) > 1:
            for extra_row in parser.header_rows[1:]:
                for index, value in enumerate(extra_row):
                    if value:
                        if index >= len(header):
                            header.extend([""] * (index + 1 - len(header)))
                        header[index] = value
    elif parser.body_rows:
        header = parser.body_rows[0]
        parser.body_rows = parser.body_rows[1:]

    if not header:
        return html

    width = len(header)
    rows = []
    for row in parser.body_rows:
        padded = row + [""] * (width - len(row))
        rows.append(padded[:width])

    lines = []
    caption = " ".join(parser.caption.split())
    if caption:
        lines.append(f"*{caption}*")
        lines.append("")

    header_line = "| " + " | ".join(markdown_escape(cell) for cell in header) + " |"
    sep_line = "| " + " | ".join(["---"] * width) + " |"
    lines.extend([header_line, sep_line])
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def first_source_line(cell: dict, max_len: int = 80) -> str:
    source = "".join(cell.get("source", []))
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_len]
    return "Code cell"


def output_text(output: dict) -> str:
    text = output.get("text", "")
    if isinstance(text, list):
        return "".join(text)
    return text


def data_text(value) -> str:
    if isinstance(value, list):
        return "".join(value)
    return value


def infer_image_name(cell_outputs: list[dict], default_stem: str) -> str:
    saved_pattern = re.compile(r"Saved:\s*([A-Za-z0-9._-]+)\.pdf\s*/\s*\.png")
    for output in cell_outputs:
        if output.get("output_type") != "stream":
            continue
        match = saved_pattern.search(output_text(output))
        if match:
            return f"{match.group(1)}.png"
    return f"{default_stem}.png"


def append_code_block(lines: list[str], text: str):
    stripped = text.strip("\n")
    if not stripped:
        return
    lines.append("~~~text")
    lines.append(stripped)
    lines.append("~~~")
    lines.append("")


def export_notebook(notebook_path: Path, output_dir: Path):
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rmd_path = output_dir / f"{notebook_path.stem}_outputs.Rmd"
    rmd_lines = [
        "---",
        f'title: "{notebook_path.stem} Outputs"',
        "output: html_document",
        "---",
        "",
        f"# {notebook_path.stem} Exported Outputs",
        "",
        f"Source notebook: `{notebook_path.name}`",
        "",
    ]

    image_count = 0
    text_count = 0

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        rmd_lines.append(f"## Cell {cell_index}")
        rmd_lines.append("")
        rmd_lines.append(f"`{first_source_line(cell)}`")
        rmd_lines.append("")

        for output_index, output in enumerate(outputs, start=1):
            output_type = output.get("output_type")

            if output_type == "stream":
                text = output_text(output)
                if text.strip():
                    rmd_lines.append(f"### Text Output {output_index}")
                    rmd_lines.append("")
                    append_code_block(rmd_lines, text)
                    text_count += 1
                continue

            data = output.get("data", {})
            handled = False

            if "image/png" in data:
                image_count += 1
                default_stem = f"cell_{cell_index:02d}_output_{output_index:02d}"
                image_name = infer_image_name(outputs, default_stem)
                image_path = images_dir / image_name
                png_data = data_text(data["image/png"])
                image_path.write_bytes(base64.b64decode(png_data))
                rmd_lines.append(f"### Image Output {output_index}")
                rmd_lines.append("")
                rmd_lines.append(f"![{image_name}](images/{image_name})")
                rmd_lines.append("")
                handled = True

            if "text/html" in data:
                html = data_text(data["text/html"])
                rmd_lines.append(f"### Rich Output {output_index}")
                rmd_lines.append("")
                rmd_lines.append(html_table_to_markdown(html))
                rmd_lines.append("")
                handled = True
                text_count += 1

            if not handled and "text/plain" in data:
                text = data_text(data["text/plain"])
                if text.strip():
                    rmd_lines.append(f"### Plain Output {output_index}")
                    rmd_lines.append("")
                    append_code_block(rmd_lines, text)
                    text_count += 1

        rmd_lines.append("")

    rmd_path.write_text("\n".join(rmd_lines).rstrip() + "\n", encoding="utf-8")
    return rmd_path, images_dir, image_count, text_count


def main():
    parser = argparse.ArgumentParser(
        description="Export embedded notebook images and non-image outputs."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported files (default: <notebook_stem>_exports)",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_dir = args.output_dir or notebook_path.with_name(f"{notebook_path.stem}_exports")
    rmd_path, images_dir, image_count, text_count = export_notebook(notebook_path, output_dir)

    print(f"Rmd saved to: {rmd_path}")
    print(f"Images saved to: {images_dir}")
    print(f"Exported {image_count} images and {text_count} non-image outputs.")


if __name__ == "__main__":
    main()
