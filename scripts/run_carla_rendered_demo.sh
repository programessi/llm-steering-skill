#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON="$ROOT/.venv310/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python venv not found:" >&2
  echo "  $PYTHON" >&2
  exit 1
fi

cd "$WORKSPACE"
exec "$PYTHON" "$ROOT/experiments/run_carla_rendered_demo.py" \
  --out "$ROOT/runs/carla_rendered_demo" \
  --host "${CARLA_HOST:-127.0.0.1}" \
  --port "${CARLA_PORT:-2000}" \
  --town "${CARLA_TOWN:-Town04}" \
  --spawn-index "${CARLA_SPAWN_INDEX:-0}" \
  --horizon "${CARLA_HORIZON:-240}" \
  --width "${CARLA_WIDTH:-1280}" \
  --height "${CARLA_HEIGHT:-720}" \
  --fps "${CARLA_FPS:-20}" \
  --timeout-s "${CARLA_TIMEOUT_S:-60}" \
  "$@"
