"""page_ocr.render — Source page → grey image (PDF, DjVu, or raster), cached."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"}


def render_page(
    path: Path | str,
    page: int = 1,
    dpi: int = 400,
    cache_dir: Path | str | None = None,
    cache_stem: str = "page",
) -> np.ndarray:
    """Grey page image (uint8, rows × columns) of ``page`` (1-based) at ``dpi``.

    ``path`` may be a PDF (``pdftoppm``), DjVu (``ddjvu``), or a raster image
    (opened directly; ``page`` must be 1). With ``cache_dir`` the PNG is kept as
    ``<cache_stem>-p<page>-<dpi>dpi.png`` and reused on the next call.
    """
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")

    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{cache_stem}-p{page}-{dpi}dpi.png"
        if cache_file.exists():
            return np.array(Image.open(cache_file).convert("L"))

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"source not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        image = _render_pdf(source, page, dpi)
    elif suffix == ".djvu":
        image = _render_djvu(source, page, dpi)
    elif suffix in _IMAGE_SUFFIXES:
        if page != 1:
            raise ValueError(f"raster images have a single page; got page={page}")
        image = Image.open(source).convert("L")
    else:
        raise ValueError(
            f"unsupported source type {suffix!r} for {source}; "
            "use .pdf, .djvu, or an image (.png, .jpg, …)"
        )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache_file)
    return np.array(image)


def _render_pdf(pdf_path: Path, page: int, dpi: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(dpi),
                "-gray",
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        produced = sorted(Path(tmp).glob("page*.png"))
        if not produced:
            raise RuntimeError(f"pdftoppm produced no image for page {page} of {pdf_path}")
        return Image.open(produced[0]).convert("L")


def _render_djvu(djvu_path: Path, page: int, dpi: int) -> Image.Image:
    # DjVuLibre ``-scale=N`` is N dpi relative to 100 dpi, so dpi=400 → -scale=400.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page.pgm"
        subprocess.run(
            [
                "ddjvu",
                "-format=pgm",
                f"-page={page}",
                f"-scale={dpi}",
                str(djvu_path),
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        if not out.is_file():
            raise RuntimeError(f"ddjvu produced no image for page {page} of {djvu_path}")
        return Image.open(out).convert("L")
