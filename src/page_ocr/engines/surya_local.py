"""page_ocr.engines.surya_local — Surya engine plugin."""
from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image

from page_ocr.render import render_page

_DPI = 400
_DEFAULT_LLAMA_BIN = Path("/opt/llama-cpp/llama-server")
# Defaults match surya.settings; used to locate a warm Hugging Face cache without
# importing Surya (and without phoning the Hub).
_GGUF_REPO_DIR = "models--datalab-to--surya-ocr-2.5-gguf"
_GGUF_MODEL_FILE = "surya-2.5.gguf"
_GGUF_MMPROJ_FILE = "surya-2.5-mmproj.gguf"
# Official datalab-to/surya-ocr-2.5-gguf weight files (mmproj is shared).
SURYA_QUANT_FILES: dict[str, str] = {
    "bf16": "surya-2.5.gguf",
    "q8": "surya-2.5-Q8_0.gguf",
}


def ocr_pdf_pages(
    pdf_path: Path | str,
    first_page: int,
    last_page: int,
    cache_dir: Path | str | None = None,
    dpi: int = _DPI,
    progress: Callable[[int, int], None] | None = None,
    *,
    cache_stem: str = "page",
) -> str:
    """OCR PDF pages ``first_page``…``last_page`` (1-based); pages separated by ``\\f``."""
    ensure_llama_env()
    predictor = _recognition_predictor()
    pages: list[str] = []
    for page in range(first_page, last_page + 1):
        if progress:
            progress(page, last_page)
        image = render_page(
            pdf_path, page, dpi=dpi, cache_dir=cache_dir, cache_stem=cache_stem
        )
        if page == first_page and first_page > 1:
            pages.extend([""] * (first_page - 1))
        pages.append(ocr_page_text(image, predictor=predictor))
    return "\f".join(pages) + "\f"


def ocr_pdf_page(
    pdf_path: Path | str,
    page: int,
    cache_dir: Path | str | None = None,
    dpi: int = _DPI,
    predictor: object | None = None,
    *,
    cache_stem: str = "page",
) -> str:
    """OCR one PDF page (1-based) to plain text (table HTML flattened when present)."""
    ensure_llama_env()
    image = render_page(
        pdf_path, page, dpi=dpi, cache_dir=cache_dir, cache_stem=cache_stem
    )
    return ocr_page_text(image, predictor=predictor)


def ocr_page_text(image: np.ndarray, predictor: object | None = None) -> str:
    """One rendered page → text lines (prefers table HTML flattened to lines)."""
    html = ocr_page_table_html(image, predictor=predictor)
    if not html:
        return ""
    return "\n".join(table_html_to_lines(html))


def ocr_page_table_html(image: np.ndarray, predictor: object | None = None) -> str:
    """Run Surya full-page OCR and return the first ``<table>…</table>`` HTML block."""
    picture = Image.fromarray(image if image.ndim == 2 else np.asarray(Image.fromarray(image).convert("RGB")))
    if picture.mode != "RGB":
        picture = picture.convert("RGB")
    recognizer = predictor or _recognition_predictor()
    prediction = recognizer([picture], full_page=True)[0]
    tables = [
        block.html
        for block in prediction.blocks
        if not block.skipped and block.html and "<table>" in block.html.lower()
    ]
    if tables:
        # Prefer the largest table (Table I body, not a tiny fragment).
        return max(tables, key=len)
    # Fallback: any block HTML that mentions a locator-like row.
    for block in prediction.blocks:
        if block.html and re.search(r"<td>\s*\d{1,4}\s*</td>", block.html, re.I):
            return block.html
    return ""


def _recognition_predictor():
    """One shared RecognitionPredictor (llama-server startup is expensive)."""
    # Env first, then import so Settings() picks up SURYA_GGUF_LOCAL_* / HF_HUB_OFFLINE.
    ensure_llama_env()
    from surya.recognition import RecognitionPredictor

    _prefer_cached_ggufs()  # patch singleton if Settings was already imported elsewhere
    return RecognitionPredictor()


def table_html_to_lines(html: str) -> list[str]:
    """Flatten Surya table HTML to ``build_nbs_index`` lines (fields separated by two spaces)."""
    rows = _parse_table_rows(html)
    lines: list[str] = []
    for cells in rows:
        if len(cells) < 4:
            continue
        locator = _plain(cells[0]).strip().rstrip(".")
        if not locator.isdigit():
            continue
        system = _formula_cell(cells[1])
        composition = _composition_cell(cells[2])
        temperature = _temperature_cell(cells[3])
        references = _plain(cells[4]).strip() if len(cells) > 4 else ""
        # Double spaces → column split in nbs_entry_index.
        parts = [locator, system, composition, temperature]
        line = "  ".join(p for p in parts if p)
        if references:
            line = f"{line}  {references}"
        lines.append(line)
    return lines


def apply_surya_quant(quant: str = "bf16") -> str:
    """Select the Surya GGUF weight file (``bf16`` or ``q8``).

    Sets ``SURYA_GGUF_MODEL_FILE`` and clears ``SURYA_GGUF_LOCAL_*`` so
    ``ensure_llama_env`` resolves the matching file from the HF cache (or
    downloads it when ``SURYA_HF_ONLINE=1``). Returns the GGUF filename.
    """
    key = quant.strip().lower()
    if key not in SURYA_QUANT_FILES:
        raise ValueError(
            f"unknown Surya quant {quant!r} (expected {tuple(SURYA_QUANT_FILES)})"
        )
    filename = SURYA_QUANT_FILES[key]
    os.environ["SURYA_GGUF_MODEL_FILE"] = filename
    # Drop sticky local paths from a previous quant so the cache lookup uses filename.
    os.environ.pop("SURYA_GGUF_LOCAL_MODEL_PATH", None)
    os.environ.pop("SURYA_GGUF_LOCAL_MMPROJ_PATH", None)
    return filename


def ensure_llama_env() -> None:
    """Point Surya at ``llama-server`` and prefer a warm GGUF cache (no Hub calls).

    Once both GGUFs exist under the Hugging Face cache (or
    ``SURYA_GGUF_LOCAL_*`` are set), Surya is pointed at those files and
    ``HF_HUB_OFFLINE=1`` is set so later runs never contact the Hub.

    Opt out with ``SURYA_HF_ONLINE=1`` (allows Hub checks / first-time download
    even when a cache is present). Explicit ``SURYA_GGUF_LOCAL_*`` always wins.
    Call ``apply_surya_quant("q8")`` first to prefer ``surya-2.5-Q8_0.gguf``.
    """
    which = shutil.which("llama-server")
    candidates = [
        Path(os.environ["LLAMA_CPP_BINARY"]) if os.environ.get("LLAMA_CPP_BINARY") else None,
        _DEFAULT_LLAMA_BIN,
        Path(which) if which else None,
    ]
    binary: Path | None = next((p for p in candidates if p is not None and p.is_file()), None)
    if binary is None:
        tried = [str(p) for p in candidates if p is not None]
        raise FileNotFoundError(
            "llama-server not found. Tried: "
            + (", ".join(tried) if tried else "(none)")
            + ". Run `python -m page_ocr install --llama` or set LLAMA_CPP_BINARY."
        )
    os.environ["LLAMA_CPP_BINARY"] = str(binary)
    lib_dir = str(binary.resolve().parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    _prefer_cached_ggufs()


def _hf_hub_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    return home / "hub"


def find_cached_gguf_paths(
    hub_root: Path | None = None,
    model_file: str = _GGUF_MODEL_FILE,
    mmproj_file: str = _GGUF_MMPROJ_FILE,
) -> tuple[Path, Path] | None:
    """Return resolved ``(model, mmproj)`` paths if both GGUFs are present on disk."""
    snapshots = (hub_root or _hf_hub_root()) / _GGUF_REPO_DIR / "snapshots"
    if not snapshots.is_dir():
        return None
    # Newest snapshot first (HF ref tip is usually the latest revision dir).
    for snap in sorted(snapshots.iterdir(), reverse=True):
        if not snap.is_dir():
            continue
        model = snap / model_file
        mmproj = snap / mmproj_file
        if not (model.exists() and mmproj.exists()):
            continue
        try:
            model_r, mmproj_r = model.resolve(), mmproj.resolve()
        except OSError:
            continue
        if model_r.is_file() and mmproj_r.is_file() and model_r.stat().st_size > 0 and mmproj_r.stat().st_size > 0:
            return model_r, mmproj_r
    return None


def _prefer_cached_ggufs() -> None:
    """Point Surya at local GGUFs when available; skip ``hf_hub_download``."""
    if os.environ.get("SURYA_HF_ONLINE", "").strip().lower() in {"1", "true", "yes"}:
        return

    model = os.environ.get("SURYA_GGUF_LOCAL_MODEL_PATH")
    mmproj = os.environ.get("SURYA_GGUF_LOCAL_MMPROJ_PATH")
    if model and mmproj:
        paths = (Path(model), Path(mmproj))
    else:
        found = find_cached_gguf_paths(
            model_file=os.environ.get("SURYA_GGUF_MODEL_FILE", _GGUF_MODEL_FILE),
            mmproj_file=os.environ.get("SURYA_GGUF_MMPROJ_FILE", _GGUF_MMPROJ_FILE),
        )
        if found is None:
            return
        paths = found
        os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = str(paths[0])
        os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = str(paths[1])

    # Only force offline when the caller did not already choose a value.
    if "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Surya loads Settings() at import time; patch the singleton if already live.
    try:
        from surya.settings import settings

        settings.SURYA_GGUF_LOCAL_MODEL_PATH = str(paths[0])
        settings.SURYA_GGUF_LOCAL_MMPROJ_PATH = str(paths[1])
    except ImportError:
        pass


def _formula_cell(raw: str) -> str:
    """``BF<sub>3</sub>-N<sub>2</sub>O`` → ``BF3-N2O``."""
    text = raw
    text = re.sub(r"<sub>\s*([^<]*?)\s*</sub>", r"\1", text, flags=re.I)
    text = re.sub(r"<sup>\s*([^<]*?)\s*</sup>", r"\1", text, flags=re.I)
    text = _plain(text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", "", text)
    return text


def _composition_cell(raw: str) -> str:
    text = _plain(raw).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("SERSOLIDSOL", "SER SOLID SOL")
    text = re.sub(r"(?<=\d)(?=APP\b)", " ", text, flags=re.I)
    text = re.sub(r"(?<=[\d.])(?=SER\b)", " ", text, flags=re.I)
    text = re.sub(r"(?<=\d)(?=LT\b)", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _temperature_cell(raw: str) -> str:
    text = _plain(raw).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    # ``-138.0±.25`` / ``-59.0APP`` → tokens the entry index already understands.
    text = re.sub(r"(?<=\d)\s*±\s*", " +", text)
    text = re.sub(r"(?<=\d|\.)(?=APP\b)", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _plain(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&nbsp;", " ").replace("&amp;", "&")


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
            # Keep tags so _formula_cell can promote subscripts to digits.
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


def recognition_predictor():
    """Public alias used by page_ocr.backend."""
    return _recognition_predictor()
