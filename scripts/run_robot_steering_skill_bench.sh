#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv310/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

cd "$ROOT"
exec "$PYTHON" experiments/run_robot_steering_skill_bench.py \
  --out "$ROOT/runs/robot_steering_skill_bench" \
  "$@"

