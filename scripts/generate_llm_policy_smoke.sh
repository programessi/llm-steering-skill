#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv310/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

cd "$ROOT"
export STAGE1_LLM_BASE_URL="${STAGE1_LLM_BASE_URL:-${OPENAI_BASE_URL:-${X2_AGENT_LLM_BASE_URL:-${AXONHUB_BASE_URL:-https://ai.zxcoding.top/v1}}}}"
export STAGE1_LLM_API_KEY="${STAGE1_LLM_API_KEY:-${OPENAI_API_KEY:-${X2_AGENT_LLM_API_KEY:-${AXONHUB_API_KEY:-}}}}"
export STAGE1_LLM_MODEL="${STAGE1_LLM_MODEL:-${OPENAI_MODEL:-${X2_AGENT_LLM_MODEL:-gpt-5.5}}}"

if [[ -z "${STAGE1_LLM_API_KEY}" ]]; then
  echo "Set STAGE1_LLM_API_KEY, OPENAI_API_KEY, X2_AGENT_LLM_API_KEY, or AXONHUB_API_KEY." >&2
  exit 1
fi

exec "$PYTHON" experiments/generate_llm_policy_smoke.py \
  --out "$ROOT/runs/llm_policy_smoke" \
  "$@"
