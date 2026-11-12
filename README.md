# page-ocr

Generic **page image → text/blocks** OCR library (local container or Modal cloud).

**Engine-pluggable:** Surya is the current backend (`PAGE_OCR_ENGINE=surya`), not the product name.

Sources: **PDF**, **DjVu**, or a **raster image**. Used by **`source-extract`** (and any other project that vendors this repo).

## Install (any server / project)

From this repo root (works as a git submodule):

```bash
chmod +x install.sh
# Vendored under <host>/vendor/page-ocr: install/deploy sync PAGE_OCR_* /
# MODAL_TOKEN_* from the host .env into ./.env automatically.
# Standalone: cp .env.example .env, or pass --from-env /path/to/host.env
./install.sh --all                 # package + local OCR runtime + PDF/DjVu tools + models
# or pick pieces:
./install.sh --local --system      # Python deps + pdftoppm/ddjvu
./install.sh --llama --models      # llama-server under /opt/llama-cpp + GGUF cache
./install.sh --modal --deploy      # Modal client + deploy OCR container
./install.sh deploy                # same: install client and `modal deploy`
```

Same after editable install:

```bash
pip install -e .
page-ocr install --all
page-ocr deploy                    # modal deploy -m page_ocr.modal_app
python -m page_ocr install --help
```

Host project helper: `make sync-vendor-env` (also from `make setup`).

## Layout

```text
src/page_ocr/
  config.py       # PAGE_OCR_*
  backend.py      # ocr_page_image local | modal
  lines.py        # prediction → lines / table HTML flatten
  render.py       # PDF / DjVu / image → grey page
  install.py      # `page-ocr install` / `python -m page_ocr install`
  cpu.py
  modal_app.py    # Modal deploy (PageOcr class)
  engines/surya_local.py
install.sh        # bootstrap wrapper for servers
```

## Env (primary)

| Variable | Notes |
| --- | --- |
| `PAGE_OCR_BACKEND` | `local` \| `modal` |
| `PAGE_OCR_ENGINE` | default `surya` |
| `PAGE_OCR_MODEL` / `PAGE_OCR_QUANT` | model id, `bf16` \| `q8` |
| `PAGE_OCR_MODAL_APP` / `_CLASS` / `_GPU` | when backend=modal |
| `PAGE_OCR_MODAL_LLAMA_RELEASE` / `_FLAVOR` | Modal image bake |
| `LLAMA_CPP_BINARY` | override llama-server path (default `/opt/llama-cpp/llama-server`) |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | deploy/call |

## Example

```bash
cp .env.example .env   # fill PAGE_OCR_* and optional Modal tokens
# local
PAGE_OCR_BACKEND=local python -c "from page_ocr import load_ocr_config; print(load_ocr_config().summary())"
# Modal container (needs MODAL_TOKEN_* + PAGE_OCR_MODAL_*)
./install.sh deploy
# equivalent: page-ocr deploy   /   modal deploy -m page_ocr.modal_app
```

This directory is its **own git repository** (submodule path `vendor/page-ocr`).
