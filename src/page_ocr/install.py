"""page_ocr.install — Provision this package on a server for local or Modal use.

Run: ``python -m page_ocr install --help``
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep in sync with modal_app / docker defaults unless overridden by env.
_DEFAULT_LLAMA_RELEASE = "b12061"
_DEFAULT_LLAMA_PREFIX = Path("/opt/llama-cpp")
_DEFAULT_MODEL = "datalab-to/surya-ocr-2.5-gguf"
_DEFAULT_QUANT = "bf16"

_QUANT_FILES = {
    "bf16": "surya-2.5.gguf",
    "q8": "surya-2.5-Q8_0.gguf",
}
_MMPROJ_FILE = "surya-2.5-mmproj.gguf"

_APT_PACKAGES = (
    "poppler-utils",  # pdftoppm
    "djvulibre-bin",  # ddjvu
    "curl",
    "ca-certificates",
    "libgomp1",
)

_PIP_BASE = (
    "numpy",
    "Pillow>=10.2,<11",
    "huggingface_hub",
)
_PIP_LOCAL = (
    "surya-ocr==0.23.0",
)
_PIP_MODAL = (
    "modal>=1.6,<2",
)


def _repo_root() -> Path:
    # src/page_ocr/install.py → repo root (…/page-ocr)
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check, env=env)


def _pip(*args: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "--upgrade", *args])


def install_python_deps(*, local: bool, modal: bool) -> None:
    _pip(*_PIP_BASE)
    if local:
        # Torch CPU wheel from the official index (surya needs torch).
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--timeout",
                "600",
                "torch==2.15.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        _pip("--timeout", "600", *_PIP_LOCAL)
    if modal:
        _pip(*_PIP_MODAL)


def install_system_deps() -> None:
    """Apt packages needed to render PDF / DjVu pages."""
    if shutil.which("apt-get") is None:
        raise SystemExit(
            "apt-get not found; install poppler-utils and djvulibre-bin with your OS package manager"
        )
    apt = ["apt-get"]
    if os.geteuid() != 0 and shutil.which("sudo"):
        apt = ["sudo", "apt-get"]
    _run([*apt, "update"])
    _run([*apt, "install", "-y", "--no-install-recommends", *_APT_PACKAGES])


def install_llama(
    *,
    release: str,
    prefix: Path,
    flavor: str = "ubuntu-x64",
) -> Path:
    """Download a prebuilt llama-server into ``prefix`` and symlink onto PATH when possible."""
    prefix = prefix.expanduser().resolve()
    tar_name = f"llama-{release}-bin-{flavor}.tar.gz"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{tar_name}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="page-ocr-llama-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / tar_name
        _run(["curl", "-fsSL", "-o", str(archive), url])
        _run(["tar", "-xz", "-f", str(archive), "-C", str(tmp_path)])
        extracted = tmp_path / f"llama-{release}"
        if not extracted.is_dir():
            candidates = [p for p in tmp_path.iterdir() if p.is_dir()]
            if not candidates:
                raise RuntimeError(f"llama.cpp archive had no directory: {archive}")
            extracted = candidates[0]
        if prefix.exists():
            shutil.rmtree(prefix)
        shutil.move(str(extracted), str(prefix))

    binary = prefix / "llama-server"
    if not binary.is_file():
        raise RuntimeError(f"llama-server missing after install: {binary}")
    binary.chmod(binary.stat().st_mode | 0o111)

    link = Path("/usr/local/bin/llama-server")
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(binary)
        print(f"linked {link} → {binary}", flush=True)
    except OSError:
        print(
            f"could not link {link}; set LLAMA_CPP_BINARY={binary}",
            flush=True,
        )

    print(f"LLAMA_CPP_BINARY={binary}", flush=True)
    _run([str(binary), "--version"], check=False)
    return binary


def install_models(*, model: str, quant: str) -> None:
    """Download Surya GGUF weights into the Hugging Face cache."""
    quant = quant.lower()
    if quant not in _QUANT_FILES:
        raise SystemExit(f"quant must be one of {sorted(_QUANT_FILES)}, got {quant!r}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub missing; run with --local first") from exc

    weight = _QUANT_FILES[quant]
    print(f"downloading {model} / {weight}", flush=True)
    hf_hub_download(repo_id=model, filename=weight)
    print(f"downloading {model} / {_MMPROJ_FILE}", flush=True)
    hf_hub_download(repo_id=model, filename=_MMPROJ_FILE)
    print("models cached under HF hub cache", flush=True)


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (comments / blanks skipped)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip("'").strip('"')
    return values


def find_host_env(*, explicit: Path | None = None) -> Path | None:
    """Host project ``.env`` when this repo is vendored (``…/vendor/page-ocr``).

    Order: ``explicit`` → ``PAGE_OCR_HOST_ENV`` → ``<repo>/../../.env`` → ``<repo>/../.env``.
    """
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env_override = (os.environ.get("PAGE_OCR_HOST_ENV") or "").strip()
    if env_override:
        path = Path(env_override)
        return path if path.is_file() else None
    root = _repo_root()
    for candidate in (root.parent.parent / ".env", root.parent / ".env"):
        if candidate.is_file():
            return candidate
    return None


def sync_dotenv_from_host(
    *,
    host_env: Path | None = None,
    dest: Path | None = None,
    template: Path | None = None,
) -> Path | None:
    """Write this package's ``.env`` from a host ``.env`` using ``.env.example`` keys.

    For each key in the template, take the host value when present, else the
    template default. Comments from the template are preserved. Returns the
    written path, or ``None`` when there is no host file / template.
    """
    root = _repo_root()
    source = host_env if host_env is not None else find_host_env()
    example = template or (root / ".env.example")
    target = dest or (root / ".env")
    if source is None or not example.is_file():
        return None

    host_values = _parse_env_file(source)
    lines_out: list[str] = [
        f"# Generated from host {source} (keys from {example.name}). Do not commit.",
        "",
    ]
    wrote = 0
    for raw in example.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines_out.append(raw)
            continue
        key, _, default = stripped.partition("=")
        key = key.strip()
        if key in host_values:
            lines_out.append(f"{key}={host_values[key]}")
            wrote += 1
        else:
            lines_out.append(raw)
    # Keys present in host but not in the template that this package cares about.
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if key in host_values and not any(line.startswith(f"{key}=") for line in lines_out):
            lines_out.append(f"{key}={host_values[key]}")
            wrote += 1

    target.write_text("\n".join(lines_out).rstrip() + "\n", encoding="utf-8")
    print(f"synced {wrote} env value(s) from {source} → {target}", flush=True)
    return target


def write_env_example_hint() -> None:
    root = _repo_root()
    env_path = root / ".env"
    example = root / ".env.example"
    if env_path.is_file():
        print(f"using {env_path}", flush=True)
    elif example.is_file():
        print(
            f"no .env yet; sync from the host project .env on install, "
            f"or copy {example} → .env",
            flush=True,
        )


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from ``.env`` into ``os.environ`` (does not override set vars)."""
    env_path = path or (_repo_root() / ".env")
    if not env_path.is_file():
        return None
    for key, value in _parse_env_file(env_path).items():
        if key not in os.environ:
            os.environ[key] = value
    return env_path


def deploy_modal(*, env_file: Path | None = None) -> None:
    """Deploy the page-ocr Modal app (``modal deploy -m page_ocr.modal_app``).

    Loads ``.env`` first. Requires ``PAGE_OCR_MODAL_*``, ``PAGE_OCR_MODEL``,
    ``PAGE_OCR_QUANT``, and ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``.
    """
    loaded = load_dotenv(env_file)
    if loaded is not None:
        print(f"loaded env from {loaded}", flush=True)
    else:
        print("no .env found; using process environment", flush=True)

    required = (
        "PAGE_OCR_MODAL_APP",
        "PAGE_OCR_MODAL_GPU",
        "PAGE_OCR_MODEL",
        "PAGE_OCR_QUANT",
        "PAGE_OCR_MODAL_LLAMA_RELEASE",
        "PAGE_OCR_MODAL_LLAMA_FLAVOR",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    )
    missing = [name for name in required if not (os.environ.get(name) or "").strip()]
    if missing:
        raise SystemExit(
            "Modal deploy needs these env vars (see .env.example): " + ", ".join(missing)
        )

    try:
        import modal  # noqa: F401
    except ImportError:
        print("modal package missing; installing…", flush=True)
        install_python_deps(local=False, modal=True)

    root = _repo_root()
    src = root / "src"
    env = os.environ.copy()
    # Ensure ``page_ocr.modal_app`` is importable for the Modal CLI.
    path_parts = [str(src)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    print("deploying Modal app page_ocr.modal_app …", flush=True)
    _run(
        [sys.executable, "-m", "modal", "deploy", "-m", "page_ocr.modal_app"],
        env=env,
    )
    print(
        f"Modal deploy done (app={os.environ['PAGE_OCR_MODAL_APP']}, "
        f"gpu={os.environ['PAGE_OCR_MODAL_GPU']})",
        flush=True,
    )


def _add_install_flags(inst: argparse.ArgumentParser) -> None:
    inst.add_argument(
        "--all",
        action="store_true",
        help="package + local Python deps + system renderers + llama-server + GGUF models "
        "(does not deploy Modal; pass --deploy)",
    )
    inst.add_argument(
        "--no-package",
        dest="package",
        action="store_false",
        help="skip pip install -e of this repo",
    )
    inst.set_defaults(package=True)
    inst.add_argument(
        "--local",
        action="store_true",
        help="install torch (CPU) + surya-ocr for PAGE_OCR_BACKEND=local",
    )
    inst.add_argument(
        "--modal",
        action="store_true",
        help="install the Modal Python client",
    )
    inst.add_argument(
        "--deploy",
        action="store_true",
        help="deploy the Modal OCR container (modal deploy -m page_ocr.modal_app)",
    )
    inst.add_argument(
        "--system",
        action="store_true",
        help="apt-install poppler-utils + djvulibre-bin (PDF / DjVu render)",
    )
    inst.add_argument(
        "--llama",
        action="store_true",
        help="download llama-server into --llama-prefix (default /opt/llama-cpp)",
    )
    inst.add_argument(
        "--models",
        action="store_true",
        help="download Surya GGUF weights into the Hugging Face cache",
    )
    inst.add_argument(
        "--llama-prefix",
        type=Path,
        default=Path(os.environ.get("LLAMA_CPP_PREFIX") or _DEFAULT_LLAMA_PREFIX),
        help="install prefix for llama-server (default: /opt/llama-cpp)",
    )
    inst.add_argument(
        "--llama-release",
        default=os.environ.get("PAGE_OCR_LLAMA_RELEASE")
        or os.environ.get("PAGE_OCR_MODAL_LLAMA_RELEASE")
        or _DEFAULT_LLAMA_RELEASE,
        help="llama.cpp GitHub release tag (default: b12061)",
    )
    inst.add_argument(
        "--llama-flavor",
        default=os.environ.get("PAGE_OCR_LLAMA_FLAVOR") or "ubuntu-x64",
        help="prebuilt binary flavor (default: ubuntu-x64 for CPU servers)",
    )
    inst.add_argument(
        "--model",
        default=os.environ.get("PAGE_OCR_MODEL") or _DEFAULT_MODEL,
        help="Hugging Face repo id for GGUF weights",
    )
    inst.add_argument(
        "--quant",
        default=os.environ.get("PAGE_OCR_QUANT") or _DEFAULT_QUANT,
        choices=sorted(_QUANT_FILES),
        help="which Surya GGUF quant to download",
    )
    inst.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="env file for Modal deploy (default: <repo>/.env after host sync)",
    )
    inst.add_argument(
        "--from-env",
        type=Path,
        default=None,
        help="host project .env to sync into this package .env (default: auto-detect vendor parent)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="page-ocr",
        description="page-ocr tools (install / Modal deploy for reuse in any project)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    inst = sub.add_parser(
        "install",
        help="Install the package and optional local/Modal OCR runtime on this machine",
    )
    _add_install_flags(inst)

    dep = sub.add_parser(
        "deploy",
        help="Deploy the Modal OCR container (same as: install --modal --deploy)",
    )
    dep.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="env file (default: <repo>/.env after host sync)",
    )
    dep.add_argument(
        "--from-env",
        type=Path,
        default=None,
        help="host project .env to sync into this package .env (default: auto-detect vendor parent)",
    )
    return p


def _sync_env_before_run(args: argparse.Namespace) -> None:
    """Fill ``<repo>/.env`` from the host project when vendored / ``--from-env`` is set."""
    host = find_host_env(explicit=getattr(args, "from_env", None))
    if host is None:
        return
    sync_dotenv_from_host(host_env=host)


def run_install(args: argparse.Namespace) -> int:
    _sync_env_before_run(args)
    load_dotenv()
    if args.all:
        args.package = True
        args.local = True
        args.modal = True
        args.system = True
        args.llama = True
        args.models = True
        # --all does not deploy; cloud deploy needs tokens and is opt-in via --deploy

    deploy = bool(getattr(args, "deploy", False))
    if deploy:
        args.modal = True

    if not any(
        (
            args.package,
            args.local,
            args.modal,
            args.system,
            args.llama,
            args.models,
            deploy,
        )
    ):
        print(
            "nothing selected; pass --local / --system / --llama / --models / --modal "
            "/ --deploy / --all",
            flush=True,
        )
        return 2

    if args.package:
        root = _repo_root()
        if _tree_writable(root / "src"):
            _run([sys.executable, "-m", "pip", "install", "--upgrade", "-e", str(root)])
        else:
            print(
                f"skip pip install -e {root} (read-only tree); using PYTHONPATH=src",
                flush=True,
            )

    if args.local or args.modal:
        install_python_deps(local=args.local, modal=args.modal)

    if args.system:
        install_system_deps()

    if args.llama:
        binary = install_llama(
            release=args.llama_release,
            prefix=args.llama_prefix,
            flavor=args.llama_flavor,
        )
        os.environ.setdefault("LLAMA_CPP_BINARY", str(binary))

    if args.models:
        install_models(model=args.model, quant=args.quant)

    if deploy:
        deploy_modal(env_file=getattr(args, "env_file", None))

    write_env_example_hint()
    print("page-ocr install done", flush=True)
    return 0


def _tree_writable(path: Path) -> bool:
    """True when ``path`` can accept new files (false for docker ``:ro`` mounts)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".page_ocr_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def run_deploy(args: argparse.Namespace) -> int:
    """``page-ocr deploy`` → install Modal client if needed, then deploy the app.

    Skips ``pip install -e`` when the repo tree is read-only (compose mounts
    ``vendor/page-ocr:ro``). ``deploy_modal`` puts ``src/`` on ``PYTHONPATH``.
    """
    _sync_env_before_run(args)
    root = _repo_root()
    if _tree_writable(root / "src"):
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "-e", str(root)])
    else:
        print(
            f"skip pip install -e {root} (read-only tree); using PYTHONPATH=src",
            flush=True,
        )
    # Only ensure the Modal client — do not upgrade numpy/HF (breaks surya pins
    # in the shared python image).
    try:
        import modal  # noqa: F401
    except ImportError:
        install_python_deps(local=False, modal=True)
    deploy_modal(env_file=args.env_file)
    print("page-ocr deploy done", flush=True)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "install":
        raise SystemExit(run_install(args))
    if args.command == "deploy":
        raise SystemExit(run_deploy(args))
    parser.error(f"unknown command {args.command!r}")
