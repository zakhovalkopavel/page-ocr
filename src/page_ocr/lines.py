"""page_ocr.lines — Flatten OCR predictions into reading-order text lines."""
from __future__ import annotations

import re
from html.parser import HTMLParser


def lines_from_prediction(prediction: object) -> list[str]:
    """Flatten one page/strip prediction into reading-order text lines."""
    preamble: list[str] = []
    best_table = ""
    for block in getattr(prediction, "blocks", []) or []:
        if getattr(block, "skipped", False) or not getattr(block, "html", None):
            continue
        html = block.html
        if "<table>" in html.lower():
            if len(html) > len(best_table):
                best_table = html
            continue
        plain = _cell_text(html)
        if plain:
            preamble.append(plain)

    table_lines = table_html_to_lines(best_table) if best_table else []
    return preamble + table_lines


def table_html_to_lines(html: str) -> list[str]:
    """Flatten ``<table>`` HTML: one space-separated line per row.

    When a ``NO FORMULA NAME`` group-header row is followed by a field-label
    subheader, emit one merged header line so downstream parsers see the real
    value columns.
    """
    rows = [[_cell_text(c) for c in cells] for cells in _parse_table_rows(html)]
    rows = _merge_stacked_header_rows(rows)
    lines: list[str] = []
    for parts in rows:
        parts = [p for p in parts if p]
        if parts:
            lines.append(" ".join(parts))
    return lines


def _merge_stacked_header_rows(rows: list[list[str]]) -> list[list[str]]:
    if len(rows) < 2:
        return rows
    out: list[list[str]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        joined = " ".join(row).upper()
        is_main = "FORMULA" in joined and "NAME" in joined
        if is_main and i + 1 < len(rows):
            nxt = rows[i + 1]
            if _looks_like_subheader(nxt) and not _looks_like_data_row(nxt):
                fields = [c for c in nxt if c]
                out.append(["NO", "FORMULA", "NAME", *fields])
                i += 2
                continue
        out.append(row)
        i += 1
    return out


def _looks_like_data_row(cells: list[str]) -> bool:
    return bool(cells and re.fullmatch(r"\d{1,4}", (cells[0] or "").strip()))


def _looks_like_subheader(cells: list[str]) -> bool:
    joined = " ".join(cells).upper()
    if "FORMULA" in joined and "NAME" in joined:
        return False
    return any(
        tok in joined
        for tok in ("CODE", "S @", "LOG10", "KOW", "LEL", "UEL", "PPM", "T, C", "T,C")
    )


def _cell_text(raw: str) -> str:
    text = re.sub(r"<sub>\s*([^<]*?)\s*</sub>", r"\1", raw, flags=re.I)
    text = re.sub(r"<sup>\s*([^<]*?)\s*</sup>", r"\1", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    return re.sub(r"\s+", " ", text).strip()


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low == "tr":
            self._row = []
        elif low in ("td", "th") and self._row is not None:
            self._cell = []
            self._in_cell = True
        elif low == "br" and self._in_cell and self._cell is not None:
            self._cell.append(" ")
        elif low in ("sub", "sup") and self._in_cell and self._cell is not None:
            self._cell.append(f"<{low}>")

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell))
            self._cell = None
            self._in_cell = False
        elif low == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif low in ("sub", "sup") and self._in_cell and self._cell is not None:
            self._cell.append(f"</{low}>")

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._cell is not None:
            self._cell.append(data)


def _parse_table_rows(html: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    return parser.rows
