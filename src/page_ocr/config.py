"""page_ocr.config — Backend / model settings from the environment.

Env: ``PAGE_OCR_*`` / ``PAGE_OCR_MODAL_*`` / ``MODAL_TOKEN_*`` only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrConfig:
    """Resolved OCR settings for one run."""

    backend: str  # local | modal
    engine: str  # surya | …
    model: str
    quant: str  # bf16 | q8 when supported
    modal_app: str
    modal_class: str
    modal_gpu: str

    def summary(self) -> str:
        parts = [
            f"backend={self.backend}",
            f"engine={self.engine}",
            f"model={self.model}",
            f"quant={self.quant}",
        ]
        if self.backend == "modal":
            parts.append(f"modal_app={self.modal_app}")
            parts.append(f"modal_class={self.modal_class}")
            parts.append(f"modal_gpu={self.modal_gpu}")
        return " ".join(parts)


def _env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(
            f"{name} must be set in the environment (see page-ocr README / .env.example)"
        )
    return value


def load_ocr_config(*, quant_override: str | None = None) -> OcrConfig:
    """Load OCR config; ``quant_override`` wins over env quant."""
    backend = _env("PAGE_OCR_BACKEND").lower()
    if backend not in {"local", "modal"}:
        raise ValueError(f"PAGE_OCR_BACKEND must be 'local' or 'modal', got {backend!r}")
    engine = (os.environ.get("PAGE_OCR_ENGINE") or "surya").strip().lower()
    model = _env("PAGE_OCR_MODEL")
    quant = (quant_override or _env("PAGE_OCR_QUANT")).strip().lower()
    if quant not in {"bf16", "q8"}:
        raise ValueError(f"PAGE_OCR_QUANT must be 'bf16' or 'q8', got {quant!r}")

    modal_app = modal_class = modal_gpu = ""
    if backend == "modal":
        modal_app = _env("PAGE_OCR_MODAL_APP")
        modal_class = _env("PAGE_OCR_MODAL_CLASS")
        modal_gpu = _env("PAGE_OCR_MODAL_GPU")

    return OcrConfig(
        backend=backend,
        engine=engine,
        model=model,
        quant=quant,
        modal_app=modal_app,
        modal_class=modal_class,
        modal_gpu=modal_gpu,
    )


def modal_tokens_present() -> bool:
    """True when both Modal token env vars are non-empty (values never logged)."""
    tid = (os.environ.get("MODAL_TOKEN_ID") or "").strip()
    secret = (os.environ.get("MODAL_TOKEN_SECRET") or "").strip()
    return bool(tid and secret)
