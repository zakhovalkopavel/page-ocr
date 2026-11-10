"""page_ocr.backend — Dispatch full-page OCR to local engine or Modal cloud."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from page_ocr.config import OcrConfig, load_ocr_config, modal_tokens_present
from page_ocr.lines import lines_from_prediction


def ocr_page_image(
    image: np.ndarray,
    *,
    config: OcrConfig | None = None,
    predictor: object | None = None,
    modal_dump_path: Path | str | None = None,
) -> list[str]:
    """OCR one page image with the configured backend; return reading-order lines."""
    cfg = config or load_ocr_config()
    if cfg.backend == "modal":
        return _ocr_page_modal(image, cfg, modal_dump_path=modal_dump_path)
    return _ocr_page_local(image, cfg, predictor=predictor)


def prepare_local_predictor(config: OcrConfig | None = None) -> object:
    """Boot local recognizer. Used once per parse run when backend=local."""
    cfg = config or load_ocr_config()
    if cfg.backend != "local":
        raise ValueError("prepare_local_predictor only applies when PAGE_OCR_BACKEND=local")
    if cfg.engine != "surya":
        raise ValueError(f"unsupported PAGE_OCR_ENGINE={cfg.engine!r} (only surya for now)")
    from page_ocr.engines.surya_local import (
        apply_surya_quant,
        ensure_llama_env,
        recognition_predictor,
    )

    _apply_local_model_env(cfg)
    apply_surya_quant(cfg.quant)
    ensure_llama_env()
    return recognition_predictor()


def _apply_local_model_env(cfg: OcrConfig) -> None:
    import os

    os.environ["PAGE_OCR_MODEL"] = cfg.model
    if cfg.model.rstrip("/") in {
        "datalab-to/surya-ocr-2.5-gguf",
        "datalab-to/surya-ocr-2.5",
    }:
        return
    os.environ["SURYA_HF_REPO"] = cfg.model


def _ocr_page_local(
    image: np.ndarray,
    cfg: OcrConfig,
    *,
    predictor: object | None,
) -> list[str]:
    if cfg.engine != "surya":
        raise ValueError(f"unsupported PAGE_OCR_ENGINE={cfg.engine!r}")
    from page_ocr.engines.surya_local import ensure_llama_env, recognition_predictor

    _apply_local_model_env(cfg)
    ensure_llama_env()
    picture = _as_rgb_pil(image)
    recognizer = predictor or recognition_predictor()
    prediction = recognizer([picture], full_page=True)[0]
    return lines_from_prediction(prediction)


def _ocr_page_modal(
    image: np.ndarray,
    cfg: OcrConfig,
    *,
    modal_dump_path: Path | str | None = None,
) -> list[str]:
    if not modal_tokens_present():
        raise RuntimeError(
            "PAGE_OCR_BACKEND=modal requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
            "in the environment (never commit them)."
        )
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            'modal package is not installed; pip install "modal>=1.6,<2"'
        ) from exc
    except RuntimeError as exc:
        if "3.14" in str(exc):
            raise RuntimeError(
                'modal 0.x does not support Python 3.14; install "modal>=1.6,<2"'
            ) from exc
        raise

    png = _image_to_png_bytes(image)
    Cls = modal.Cls.from_name(cfg.modal_app, cfg.modal_class)
    result: dict[str, Any] = Cls().ocr_page.remote(png)
    if modal_dump_path is not None:
        path = Path(modal_dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    elapsed = result.get("elapsed_ms")
    if elapsed is not None:
        print(
            f"[page-ocr] modal OCR {elapsed}ms "
            f"(engine={cfg.engine} quant={result.get('quant')} "
            f"gpu={result.get('gpu')} rev={result.get('deploy_rev')})",
            flush=True,
        )
    blocks = [
        SimpleNamespace(html=b.get("html"), skipped=bool(b.get("skipped")))
        for b in (result.get("blocks") or [])
    ]
    return lines_from_prediction(SimpleNamespace(blocks=blocks))


def _as_rgb_pil(image: np.ndarray) -> Image.Image:
    picture = Image.fromarray(
        image if image.ndim == 2 else np.asarray(Image.fromarray(image).convert("RGB"))
    )
    if picture.mode != "RGB":
        picture = picture.convert("RGB")
    return picture


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    picture = _as_rgb_pil(image)
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    return buf.getvalue()
