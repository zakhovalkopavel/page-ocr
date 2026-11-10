"""page_ocr.cpu — CPU-friendly defaults for local Surya / llama-server."""
from __future__ import annotations

import os
from pathlib import Path


def logical_cpu_count() -> int:
    base = Path("/sys/devices/system/cpu")
    if base.is_dir():
        online = sorted(base.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:]))
        if online:
            return max(1, len(online))
    return max(1, os.cpu_count() or 4)


def physical_cpu_count() -> int:
    seen: set[str] = set()
    base = Path("/sys/devices/system/cpu")
    if base.is_dir():
        for sib in base.glob("cpu[0-9]*/topology/thread_siblings_list"):
            try:
                seen.add(sib.read_text(encoding="utf-8").strip())
            except OSError:
                continue
        if seen:
            return max(1, len(seen))
    return max(1, logical_cpu_count() // 2)


def apply_cpu_defaults() -> dict[str, str]:
    """Set Surya/llama env knobs when the caller has not already set them."""
    threads = str(logical_cpu_count())
    applied: dict[str, str] = {}

    def _setdefault(key: str, value: str) -> None:
        if not os.environ.get(key, "").strip():
            os.environ[key] = value
        applied[key] = os.environ[key]

    _setdefault("SURYA_INFERENCE_PARALLEL", "1")
    extra = os.environ.get("LLAMA_CPP_EXTRA_ARGS", "").strip()
    if not extra:
        os.environ["LLAMA_CPP_EXTRA_ARGS"] = f"--threads {threads} --threads-batch {threads}"
    applied["LLAMA_CPP_EXTRA_ARGS"] = os.environ["LLAMA_CPP_EXTRA_ARGS"]
    applied["SURYA_INFERENCE_PARALLEL"] = os.environ["SURYA_INFERENCE_PARALLEL"]
    applied["logical_cpus"] = threads
    applied["physical_cpus"] = str(physical_cpu_count())
    return applied
