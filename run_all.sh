#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$ROOT_DIR/figures"
cd "$ROOT_DIR/src"

figures=(
  figure1.py
  figure2.py
  figure3.py
  figure4.py
  figure5.py
  figure6.py
  figure7.py
  figure8.py
  figure9.py
  figure10.py
  figure11.py
  figure12.py
  figure13.py
  figure14.py
)

for figure in "${figures[@]}"; do
  echo "Running ${figure}..."
  "$PYTHON_BIN" "$figure"
done

echo "All figures have been generated in $ROOT_DIR/figures."
