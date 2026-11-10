"""page_ocr — generic page image OCR (local container or cloud).

Surya is the current engine (``PAGE_OCR_ENGINE=surya``), not the product name.
"""
from page_ocr.config import OcrConfig, load_ocr_config, modal_tokens_present
from page_ocr.backend import ocr_page_image, prepare_local_predictor
from page_ocr.lines import lines_from_prediction, table_html_to_lines
from page_ocr.render import render_page

__all__ = [
    "OcrConfig",
    "load_ocr_config",
    "modal_tokens_present",
    "ocr_page_image",
    "prepare_local_predictor",
    "lines_from_prediction",
    "table_html_to_lines",
    "render_page",
]
