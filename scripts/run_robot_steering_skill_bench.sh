#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_select_python.sh"

cd "$ROOT"
exec "$PYTHON" experiments/run_robot_steering_skill_bench.py \
  --out "$ROOT/runs/robot_steering_skill_bench" \
  "$@"
