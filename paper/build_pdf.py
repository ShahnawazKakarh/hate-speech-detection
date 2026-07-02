#!/usr/bin/env python3
"""Compile the hate-speech-detection paper to PDF via WeasyPrint.

Combines:
  - paper/cover.html   (page 1 — cover)
  - paper/main.md      (Markdown body — converted to HTML)
  - paper/theme.css    (shared styling)
  - paper/figures/*    (embedded via file:// URLs)
  - paper/tables/*.md  (source of truth for table content)

Outputs:
  - paper/main.pdf

Usage:
  pip install weasyprint markdown pymdown-extensions
  python paper/build_pdf.py

If any font in theme.css fails to fetch from the internet, WeasyPrint falls
back to the local Georgia / Helvetica / Menlo — output looks near-identical.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
COVER_HTML = PAPER_DIR / "cover.html"
MAIN_MD = PAPER_DIR / "main.md"
THEME_CSS = PAPER_DIR / "theme.css"
OUT_PDF = PAPER_DIR / "main.pdf"


def _render_body_html(md_path: Path) -> str:
    import markdown

    text = md_path.read_text(encoding="utf-8")
    html = markdown.markdown(
        text,
        extensions=[
            "extra",  # tables, fenced code, footnotes, def lists
            "sane_lists",
            "smarty",  # smart quotes / em-dashes
            "toc",
            "attr_list",
            "md_in_html",
        ],
        output_format="html5",
    )
    return html


def _compose(body_html: str) -> str:
    cover = COVER_HTML.read_text(encoding="utf-8")

    # Strip cover.html down to its <section class="cover">...</section> block
    # so we don't emit two <html>/<head>.
    start = cover.find("<section")
    end = cover.rfind("</section>") + len("</section>")
    cover_section = cover[start:end] if start != -1 else ""

    doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Hate Speech Detection</title>
  <link rel=\"stylesheet\" href=\"file://{THEME_CSS.resolve()}\">
</head>
<body>
{cover_section}
<section class=\"content\">
{body_html}
</section>
</body>
</html>
"""
    return doc


def build() -> Path:
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise SystemExit(
            "weasyprint not installed. Run:\n"
            "    pip install weasyprint markdown pymdown-extensions"
        ) from e

    if not MAIN_MD.exists():
        raise SystemExit(
            f"{MAIN_MD} not found. Create paper/main.md first (translate from paper/main.tex)."
        )

    body_html = _render_body_html(MAIN_MD)
    full_html = _compose(body_html)

    # Persist the composed HTML for debugging
    debug_html = PAPER_DIR / "_build_debug.html"
    debug_html.write_text(full_html, encoding="utf-8")

    HTML(string=full_html, base_url=str(PAPER_DIR)).write_pdf(str(OUT_PDF))

    print(f"wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.1f} KB)")
    return OUT_PDF


if __name__ == "__main__":
    if shutil.which("weasyprint") is None:
        pass  # ok; we import python module
    build()
