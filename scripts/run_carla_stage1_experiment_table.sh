#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
source "$ROOT/scripts/_select_python.sh"

cd "$WORKSPACE"
exec "$PYTHON" "$ROOT/experiments/run_carla_stage1_experiment_table.py" \
  --out "$ROOT/runs/carla_stage1_experiment_table" \
  --host "${CARLA_HOST:-127.0.0.1}" \
  --port "${CARLA_PORT:-2000}" \
  --town "${CARLA_TOWN:-}" \
  --width "${CARLA_WIDTH:-1280}" \
  --height "${CARLA_HEIGHT:-720}" \
  --fps "${CARLA_FPS:-20}" \
  --timeout-s "${CARLA_TIMEOUT_S:-60}" \
  --lane-departure-limit-m "${CARLA_LANE_DEPARTURE_LIMIT_M:-5.0}" \
  "$@"
