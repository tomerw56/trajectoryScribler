#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SKIP_TESTS=0
COVERAGE=0
FIX=0
for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    --coverage) COVERAGE=1 ;;
    --fix) FIX=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

for tool in black ruff mypy pytest; do assert_python_module "$tool"; done
if [[ "$COVERAGE" == "1" ]]; then assert_python_module pytest_cov; fi

if [[ "$FIX" == "1" ]]; then
  echo "== Safe auto-fix =="
  invoke_python_module ruff check --fix --show-fixes "${PYTHON_TARGETS[@]}"
  invoke_python_module black "${PYTHON_TARGETS[@]}"
fi

echo "== Compile check =="
python -m compileall -q trajectory_app tests app.py example_custom_solver.py

echo "== Black check =="
if ! python -m black --check --diff "${PYTHON_TARGETS[@]}"; then
  if [[ "$FIX" != "1" ]]; then
    echo "Black formatting is required. Run: ./scripts/format.sh or ./scripts/check.sh --fix" >&2
  fi
  exit 1
fi

echo "== Ruff lint =="
invoke_python_module ruff check --output-format=full "${PYTHON_TARGETS[@]}"

echo "== Mypy =="
invoke_python_module mypy --show-error-codes --show-column-numbers "${PYTHON_TARGETS[@]}"

if [[ "$SKIP_TESTS" != "1" ]]; then
  echo "== Pytest =="
  if [[ "$COVERAGE" == "1" ]]; then
    invoke_python_module pytest -q --cov=trajectory_app --cov-report=term-missing --cov-report=xml
  else
    invoke_python_module pytest -q
  fi
fi

echo "All checks passed."
