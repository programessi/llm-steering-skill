#!/usr/bin/env bash
# Select the Python interpreter used by project scripts.
#
# Priority:
#   1. explicit PYTHON=/path/to/python from the caller
#   2. project-local .venv310
#   3. project-local .venv
#   4. python on PATH

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv310/bin/python" ]]; then
    PYTHON="$ROOT/.venv310/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi
