"""PDF export via PyMuPDF Story (HTML subset → paginated PDF)."""
from __future__ import annotations

import re
from pathlib import Path


def html_to_pdf(html: str, out_path: str | Path) -> str:
    import pymupdf

    # Story supports a CSS subset; simplify styles for robust layout
    body = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
    css = ("body{font-family:sans-serif;font-size:8pt;} h1{font-size:15pt;} h2{font-size:11pt;margin-top:10pt;} "
           "h3{font-size:9pt;} table{border-collapse:collapse;} th,td{border:0.5px solid #999;padding:1pt 2pt;font-size:6.5pt;} "
           "pre{font-size:6pt;} .muted{color:#555;} .warn{background-color:#fff6d6;}")
    story = pymupdf.Story(html=body, user_css=css)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (28, 28, -28, -28)
    writer = pymupdf.DocumentWriter(str(out_path))
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return str(out_path)
