#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

UNSAFE=0
for arg in "$@"; do
  case "$arg" in
    --unsafe-fixes) UNSAFE=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

require_module ruff
require_module black

RUFF_ARGS=(check --fix --show-fixes)
if [[ "$UNSAFE" == "1" ]]; then RUFF_ARGS+=(--unsafe-fixes); fi

echo "== Ruff autofix =="
python -m ruff "${RUFF_ARGS[@]}" "${PYTHON_TARGETS[@]}"

echo "== Black format =="
python -m black "${PYTHON_TARGETS[@]}"

echo "== Ruff verify remaining lint =="
python -m ruff check --output-format=full "${PYTHON_TARGETS[@]}"

echo "Formatting/fixes complete."
