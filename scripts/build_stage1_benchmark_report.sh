#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_select_python.sh"

cd "$ROOT"
exec "$PYTHON" experiments/build_stage1_benchmark_report.py "$@"
