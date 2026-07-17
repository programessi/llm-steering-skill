#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_select_python.sh"

cd "$ROOT"
exec "$PYTHON" "$ROOT/experiments/check_carla_rpc.py" \
  --host "${CARLA_HOST:-127.0.0.1}" \
  --port "${CARLA_PORT:-2000}" \
  --timeout-s "${CARLA_TIMEOUT_S:-30}" \
  "$@"
