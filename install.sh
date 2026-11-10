#!/usr/bin/env bash
# Bootstrap page-ocr on a server (usable as a submodule from any project).
# Examples:
#   ./install.sh --all                      # local OCR runtime
#   ./install.sh --local --system --llama --models
#   ./install.sh --modal --deploy           # Modal client + deploy container
#   ./install.sh deploy                     # shorthand: client + modal deploy
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade -e "$ROOT"
# Allow ``./install.sh deploy`` as well as ``./install.sh --deploy`` / install flags.
if [[ "${1:-}" == "deploy" ]]; then
  shift
  exec python3 -m page_ocr deploy "$@"
fi
exec python3 -m page_ocr install "$@"
