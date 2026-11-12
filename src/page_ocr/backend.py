"""page_ocr.backend — Dispatch full-page OCR to local engine or Modal cloud."""
from __future__ import annotations

import io
import json
import time
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
    raw_dump_path: Path | str | None = None,
    modal_dump_path: Path | str | None = None,
    dpi: int | None = None,
) -> list[str]:
    """OCR one page image with the configured backend; return reading-order lines.

    When ``raw_dump_path`` is set, write the backend payload as JSON (same shape
    for local and Modal: ``blocks`` with ``html`` / ``skipped``, plus provenance).
    ``modal_dump_path`` is accepted as an alias of ``raw_dump_path``.
    ``dpi`` is recorded in the raw dump when provided (render resolution).
    """
    dump = raw_dump_path if raw_dump_path is not None else modal_dump_path
    cfg = config or load_ocr_config()
    if cfg.backend == "modal":
        return _ocr_page_modal(image, cfg, raw_dump_path=dump, dpi=dpi)
    return _ocr_page_local(
        image, cfg, predictor=predictor, raw_dump_path=dump, dpi=dpi,
    )


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
    raw_dump_path: Path | str | None = None,
    dpi: int | None = None,
) -> list[str]:
    if cfg.engine != "surya":
        raise ValueError(f"unsupported PAGE_OCR_ENGINE={cfg.engine!r}")
    from page_ocr.engines.surya_local import ensure_llama_env, recognition_predictor

    _apply_local_model_env(cfg)
    ensure_llama_env()
    picture = _as_rgb_pil(image)
    recognizer = predictor or recognition_predictor()
    t0 = time.perf_counter()
    prediction = recognizer([picture], full_page=True)[0]
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if raw_dump_path is not None:
        payload: dict[str, Any] = {
            "backend": "local",
            "engine": cfg.engine,
            "model": cfg.model,
            "quant": cfg.quant,
            "elapsed_ms": elapsed_ms,
            "elapsed_s": round(elapsed_ms / 1000.0, 3),
            "blocks": _blocks_payload(prediction),
        }
        if dpi is not None:
            payload["dpi"] = int(dpi)
        _write_raw_dump(raw_dump_path, payload)
    return lines_from_prediction(prediction)


def _ocr_page_modal(
    image: np.ndarray,
    cfg: OcrConfig,
    *,
    raw_dump_path: Path | str | None = None,
    dpi: int | None = None,
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
    # Dual-quant deploy: PageOcr(quant=…) selects which warm GGUF worker to hit.
    worker = Cls(quant=cfg.quant)
    t0 = time.perf_counter()
    result: dict[str, Any] = worker.ocr_page.remote(png)
    client_elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if raw_dump_path is not None:
        payload = dict(result)
        payload.setdefault("backend", "modal")
        payload.setdefault("engine", cfg.engine)
        payload.setdefault("model", cfg.model)
        payload.setdefault("quant", cfg.quant)
        if dpi is not None:
            payload["dpi"] = int(dpi)
        # Server-side OCR time (from Modal worker) when present.
        if payload.get("elapsed_ms") is not None:
            payload["elapsed_s"] = round(float(payload["elapsed_ms"]) / 1000.0, 3)
        # Full client round-trip (queue + network + OCR).
        payload["client_elapsed_ms"] = client_elapsed_ms
        payload["client_elapsed_s"] = round(client_elapsed_ms / 1000.0, 3)
        _write_raw_dump(raw_dump_path, payload)
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


def _blocks_payload(prediction: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in getattr(prediction, "blocks", []) or []:
        out.append(
            {
                "html": getattr(block, "html", None),
                "skipped": bool(getattr(block, "skipped", False)),
            }
        )
    return out


def _write_raw_dump(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


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
