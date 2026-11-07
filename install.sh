#!/usr/bin/env bash
# Optional helper: create a virtual environment in .venv and install the requirements.
#   bash install.sh            (PYTHON=python3.12 bash install.sh to pick the interpreter)
#   .venv/bin/python scripts/serve.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py="${PYTHON:-python3}"
"$py" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "installed. start the server with: .venv/bin/python scripts/serve.py"
