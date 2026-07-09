#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$ROOT/third_party/carla}"
CARLA_BIN="$CARLA_ROOT/CarlaUE4.sh"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_QUALITY="${CARLA_QUALITY:-Low}"
CARLA_RENDER_OFFSCREEN="${CARLA_RENDER_OFFSCREEN:-0}"

if [[ ! -x "$CARLA_BIN" ]]; then
  echo "CARLA server binary not found or not executable:" >&2
  echo "  $CARLA_BIN" >&2
  echo "Run scripts/download_carla_server.sh, set CARLA_ROOT, or place CARLA under:" >&2
  echo "  $CARLA_ROOT" >&2
  exit 1
fi

args=(-quality-level="$CARLA_QUALITY" -nosound -carla-rpc-port="$CARLA_PORT")
if [[ "$CARLA_RENDER_OFFSCREEN" == "1" ]]; then
  args=(-RenderOffScreen "${args[@]}")
fi

exec env VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
  __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}" \
  SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}" \
  "$CARLA_BIN" "${args[@]}" "$@"
