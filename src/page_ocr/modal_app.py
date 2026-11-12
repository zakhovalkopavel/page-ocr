"""page_ocr.modal_app — Modal-hosted page OCR (engine=Surya for now).

Deploy:
  modal deploy -m page_ocr.modal_app

``PageOcr`` is parameterized by ``quant`` (``bf16`` | ``q8``). One deploy
registers both; the client picks at call time via ``Cls(quant=…).ocr_page``.
Each quant gets its own warm container (not both GGUFs in one VRAM).

Env: PAGE_OCR_MODAL_*, PAGE_OCR_MODEL, PAGE_OCR_QUANT (default quant),
PAGE_OCR_MODAL_LLAMA_*, …

Note: do **not** use ``from __future__ import annotations`` here — Modal's
``modal.parameter()`` needs real runtime types (``str``), not stringified
annotations (breaks under Python 3.14 / postponed evaluation).
"""
import io
import os
from typing import Any

import modal

QUANT_FILES = {
    "bf16": "surya-2.5.gguf",
    "q8": "surya-2.5-Q8_0.gguf",
}
SUPPORTED_QUANTS = tuple(QUANT_FILES)
MMPROJ_FILE = "surya-2.5-mmproj.gguf"


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set before deploying Modal OCR (see .env.example)")
    return value


# Read from the deploy client env, then bake into the image ENV so remote workers
# (which re-import this module) see the same values without a local .env.
_APP_NAME = _require_env("PAGE_OCR_MODAL_APP")
_GPU = _require_env("PAGE_OCR_MODAL_GPU")
_MODEL = _require_env("PAGE_OCR_MODEL")
_QUANT = _require_env("PAGE_OCR_QUANT").lower()
_LLAMA_RELEASE = _require_env("PAGE_OCR_MODAL_LLAMA_RELEASE")
_LLAMA_FLAVOR = _require_env("PAGE_OCR_MODAL_LLAMA_FLAVOR")  # e.g. ubuntu-cuda-12.8-x64
_CACHE = "/cache"

if _QUANT not in QUANT_FILES:
    raise RuntimeError(
        f"PAGE_OCR_QUANT must be one of {sorted(QUANT_FILES)}, got {_QUANT!r}"
    )

if _GPU.lower() in {"none", "cpu", "off", "0"}:
    raise RuntimeError(
        "PAGE_OCR_MODAL_GPU must be a Modal GPU id (e.g. T4). "
        "CPU-only Modal defeats the point of remote OCR acceleration."
    )

# CUDA llama-server on T4 (etc.) + forced llamacpp backend (never vLLM/Docker).
# surya-ocr 0.23 requires torch>=2.7 and pillow>=10.2,<11.
_DEPLOY_REV = "llamacpp-cuda-4-dual-quant"
_LLAMA_TAR = f"llama-{_LLAMA_RELEASE}-bin-{_LLAMA_FLAVOR}.tar.gz"
_CUDART_TAR = f"cudart-llama-{_LLAMA_RELEASE}-bin-{_LLAMA_FLAVOR}.tar.gz"
_image = (
    # Ubuntu 24.04: prebuilt llama.cpp CUDA bins need GLIBC ≥ 2.38 (22.04 is too old).
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "curl",
        "ca-certificates",
        "libgomp1",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.15.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "surya-ocr==0.23.0",
        "Pillow>=10.2,<11",
        "numpy",
        "huggingface_hub",
    )
    .run_commands(
        # CUDA llama-server + matching cudart/cublas libs from the same release.
        f"curl -fsSL "
        f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_RELEASE}/{_LLAMA_TAR} "
        f"| tar -xz -C /opt && "
        f"curl -fsSL "
        f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_RELEASE}/{_CUDART_TAR} "
        f"| tar -xz -C /opt && "
        f"rm -rf /opt/llama-cpp && "
        f"mv /opt/llama-{_LLAMA_RELEASE} /opt/llama-cpp && "
        f"cp -a /opt/cudart-llama-{_LLAMA_RELEASE}-bin-{_LLAMA_FLAVOR}/. /opt/llama-cpp/ && "
        f"rm -rf /opt/cudart-llama-{_LLAMA_RELEASE}-bin-{_LLAMA_FLAVOR} && "
        f"ln -sf /opt/llama-cpp/llama-server /usr/local/bin/llama-server && "
        f"llama-server --version",
    )
    .env(
        {
            "LLAMA_CPP_BINARY": "/opt/llama-cpp/llama-server",
            "LD_LIBRARY_PATH": "/opt/llama-cpp:/usr/local/cuda/lib64",
            "HF_HOME": _CACHE,
            "HUGGINGFACE_HUB_CACHE": f"{_CACHE}/huggingface",
            "HF_HUB_CACHE": f"{_CACHE}/huggingface/hub",
            "SURYA_INFERENCE_PARALLEL": "1",
            "SURYA_INFERENCE_BACKEND": "llamacpp",
            "LLAMA_CPP_NGL": "99",
            "SURYA_HF_ONLINE": "1",
            "PAGE_OCR_DEPLOY_REV": _DEPLOY_REV,
            "PAGE_OCR_MODAL_APP": _APP_NAME,
            "PAGE_OCR_MODAL_GPU": _GPU,
            "PAGE_OCR_MODAL_LLAMA_RELEASE": _LLAMA_RELEASE,
            "PAGE_OCR_MODAL_LLAMA_FLAVOR": _LLAMA_FLAVOR,
            "PAGE_OCR_MODEL": _MODEL,
            "PAGE_OCR_QUANT": _QUANT,
        }
    )
)

_volume = modal.Volume.from_name("page-ocr-cache", create_if_missing=True)

app = modal.App(_APP_NAME, image=_image)


def _apply_model_env(model: str, quant: str) -> None:
    """Point Surya at CUDA llama-server + the configured HF GGUF repo / quant file."""
    quant = quant.strip().lower()
    if quant not in QUANT_FILES:
        raise ValueError(f"PAGE_OCR_QUANT must be one of {sorted(QUANT_FILES)}, got {quant!r}")
    os.environ["PAGE_OCR_MODEL"] = model
    os.environ["PAGE_OCR_QUANT"] = quant
    os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
    os.environ.setdefault("LLAMA_CPP_BINARY", "/opt/llama-cpp/llama-server")
    os.environ.setdefault("LLAMA_CPP_NGL", "99")
    repo = model.strip()
    if "/" in repo:
        os.environ["SURYA_GGUF_REPO"] = repo
    os.environ["SURYA_GGUF_MODEL_FILE"] = QUANT_FILES[quant]
    os.environ["SURYA_GGUF_MMPROJ_FILE"] = MMPROJ_FILE
    os.environ.setdefault("SURYA_HF_ONLINE", "1")


@app.cls(
    gpu=_GPU,
    timeout=1800,
    scaledown_window=300,
    volumes={_CACHE: _volume},
)
class PageOcr:
    """Warm page OCR recognizer (Surya) on Modal GPU.

    Parameter ``quant`` selects which GGUF this container loads (``bf16`` or
    ``q8``). Call as ``PageOcr(quant=\"q8\").ocr_page.remote(png)``.
    """

    quant: str = modal.parameter(default=_QUANT)

    @modal.enter()
    def load(self) -> None:
        model = (os.environ.get("PAGE_OCR_MODEL") or _MODEL).strip()
        quant = (self.quant or _QUANT).strip().lower()
        _apply_model_env(model, quant)
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings

        # Force llamacpp even though a GPU is present (auto would pick vLLM/Docker).
        settings.SURYA_INFERENCE_BACKEND = "llamacpp"
        settings.LLAMA_CPP_BINARY = os.environ["LLAMA_CPP_BINARY"]
        settings.LLAMA_CPP_NGL = int(os.environ.get("LLAMA_CPP_NGL", "99"))
        settings.SURYA_INFERENCE_KEEP_ALIVE = True
        settings.SURYA_GGUF_REPO = os.environ.get("SURYA_GGUF_REPO", settings.SURYA_GGUF_REPO)
        settings.SURYA_GGUF_MODEL_FILE = os.environ["SURYA_GGUF_MODEL_FILE"]
        settings.SURYA_GGUF_MMPROJ_FILE = os.environ["SURYA_GGUF_MMPROJ_FILE"]

        self.model_id = model
        self.quant_id = quant
        self.deploy_rev = os.environ.get("PAGE_OCR_DEPLOY_REV", _DEPLOY_REV)
        self.gpu = os.environ.get("PAGE_OCR_MODAL_GPU", _GPU)
        # Eager-start llama-server + load GGUF here (not on the first page call).
        manager = SuryaInferenceManager(method="llamacpp", lazy=False)
        self.predictor = RecognitionPredictor(manager=manager)

    @modal.method()
    def ocr_page(self, image_bytes: bytes) -> dict[str, Any]:
        """OCR a single page image; returns serializable block HTML for the client."""
        import time

        from PIL import Image

        t0 = time.perf_counter()
        picture = Image.open(io.BytesIO(image_bytes))
        if picture.mode != "RGB":
            picture = picture.convert("RGB")
        prediction = self.predictor([picture], full_page=True)[0]
        blocks: list[dict[str, Any]] = []
        for block in getattr(prediction, "blocks", []) or []:
            blocks.append(
                {
                    "html": getattr(block, "html", None),
                    "skipped": bool(getattr(block, "skipped", False)),
                }
            )
        return {
            "model": self.model_id,
            "quant": self.quant_id,
            "deploy_rev": self.deploy_rev,
            "gpu": self.gpu,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "blocks": blocks,
        }

    @modal.method()
    def health(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "quant": self.quant_id,
            "supported_quants": list(SUPPORTED_QUANTS),
            "deploy_rev": self.deploy_rev,
            "gpu": self.gpu,
            "backend": "llamacpp",
            "status": "ok",
        }
