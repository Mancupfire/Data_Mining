#!/usr/bin/env python3
import argparse
import base64
import json
import re
from pathlib import Path

from export_notebook_outputs import (
    append_code_block,
    data_text,
    first_source_line,
    html_table_to_markdown,
    output_text,
)


def sanitize_chunk_label(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-")
    return cleaned[:50] or fallback


def yaml_escape(text: str) -> str:
    return text.replace('"', '\\"')


def build_yaml(notebook_path: Path) -> list[str]:
    title = notebook_path.stem.replace("_", " ")
    return [
        "---",
        f'title: "{yaml_escape(title)}"',
        "output:",
        "  html_document:",
        "    toc: true",
        "    toc_depth: 3",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)",
        "```",
        "",
    ]


def export_full_rmd(notebook_path: Path, output_path: Path):
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    assets_dir = output_path.with_suffix("")
    images_dir = assets_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    lines = build_yaml(notebook_path)
    image_count = 0
    used_labels = set()

    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            content = source.rstrip()
            if content:
                lines.append(content)
                lines.append("")
            continue

        if cell_type != "code":
            continue

        chunk_label = sanitize_chunk_label(
            first_source_line(cell, max_len=60), f"cell-{cell_index}"
        )
        base_label = chunk_label
        suffix = 2
        while chunk_label in used_labels:
            chunk_label = f"{base_label}-{suffix}"
            suffix += 1
        used_labels.add(chunk_label)
        lines.append(f"```{{python {chunk_label}, eval=FALSE}}")
        lines.append(source.rstrip())
        lines.append("```")
        lines.append("")

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        lines.append(f"**Saved output from notebook cell {cell_index}:**")
        lines.append("")

        for output_index, output in enumerate(outputs, start=1):
            output_type = output.get("output_type")

            if output_type == "stream":
                text = output_text(output)
                if text.strip():
                    append_code_block(lines, text)
                continue

            data = output.get("data", {})

            if "text/html" in data:
                lines.append(html_table_to_markdown(data_text(data["text/html"])))
                lines.append("")

            if "image/png" in data:
                image_count += 1
                image_name = f"cell_{cell_index:02d}_output_{output_index:02d}.png"
                image_path = images_dir / image_name
                image_path.write_bytes(base64.b64decode(data_text(data["image/png"])))
                rel_path = image_path.relative_to(output_path.parent)
                lines.append(f"![Notebook output {cell_index}-{output_index}]({rel_path.as_posix()})")
                lines.append("")

            if "text/plain" in data and "image/png" not in data and "text/html" not in data:
                text = data_text(data["text/plain"])
                if text.strip():
                    append_code_block(lines, text)

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path, images_dir, image_count


def main():
    parser = argparse.ArgumentParser(description="Convert a Jupyter notebook to a full R Markdown document.")
    parser.add_argument("notebook", type=Path, help="Path to the input .ipynb notebook")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output .Rmd file (default: <notebook_stem>_full.Rmd)",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_path = args.output or notebook_path.with_name(f"{notebook_path.stem}_full.Rmd")

    rmd_path, images_dir, image_count = export_full_rmd(notebook_path, output_path)
    print(f"Rmd saved to: {rmd_path}")
    print(f"Images saved to: {images_dir}")
    print(f"Exported {image_count} embedded notebook images.")


if __name__ == "__main__":
    main()
