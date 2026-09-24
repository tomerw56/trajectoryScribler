#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_TARGETS=(trajectory_app tests app.py example_custom_solver.py)

require_module() {
  local module="$1"
  if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)"; then
    echo "Missing development tool '$module'. Run: python -m pip install -r requirements-dev.txt" >&2
    exit 2
  fi
}
