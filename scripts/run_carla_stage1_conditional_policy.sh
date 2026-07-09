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
export STAGE1_LLM_BASE_URL="${STAGE1_LLM_BASE_URL:-${OPENAI_BASE_URL:-${X2_AGENT_LLM_BASE_URL:-${AXONHUB_BASE_URL:-https://ai.zxcoding.top/v1}}}}"
export STAGE1_LLM_API_KEY="${STAGE1_LLM_API_KEY:-${OPENAI_API_KEY:-${X2_AGENT_LLM_API_KEY:-${AXONHUB_API_KEY:-}}}}"
export STAGE1_LLM_MODEL="${STAGE1_LLM_MODEL:-${OPENAI_MODEL:-${X2_AGENT_LLM_MODEL:-gpt-5.5}}}"

exec "$PYTHON" "$ROOT/experiments/run_carla_stage1_conditional_policy.py" \
  --out "$ROOT/runs/carla_stage1_conditional_policy" \
  --host "${CARLA_HOST:-127.0.0.1}" \
  --port "${CARLA_PORT:-2000}" \
  --town "${CARLA_TOWN:-}" \
  --width "${CARLA_WIDTH:-1280}" \
  --height "${CARLA_HEIGHT:-720}" \
  --fps "${CARLA_FPS:-20}" \
  --timeout-s "${CARLA_TIMEOUT_S:-60}" \
  --lane-departure-limit-m "${CARLA_LANE_DEPARTURE_LIMIT_M:-5.0}" \
  "$@"
