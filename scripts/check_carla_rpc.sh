#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv310/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python venv not found:" >&2
  echo "  $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" "$ROOT/experiments/check_carla_rpc.py" \
  --host "${CARLA_HOST:-127.0.0.1}" \
  --port "${CARLA_PORT:-2000}" \
  --timeout-s "${CARLA_TIMEOUT_S:-30}" \
  "$@"
