#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$ROOT/third_party/downloads"
INSTALL_DIR="$ROOT/third_party/carla"
ARCHIVE="$DOWNLOAD_DIR/CARLA_0.9.15.tar.gz"
PARTIAL="$ARCHIVE.part"
URL="${CARLA_SERVER_URL:-https://carla-releases.b-cdn.net/Linux/CARLA_0.9.15.tar.gz}"

mkdir -p "$DOWNLOAD_DIR" "$INSTALL_DIR"

if [[ -f "$ARCHIVE" ]]; then
  if tar -tzf "$ARCHIVE" >/dev/null 2>&1; then
    echo "Existing CARLA archive passed gzip/tar validation:"
    echo "  $ARCHIVE"
  else
    BACKUP="$ARCHIVE.bad.$(date +%Y%m%d%H%M%S)"
    echo "Existing CARLA archive is incomplete or corrupt; keeping it as:"
    echo "  $BACKUP"
    mv "$ARCHIVE" "$BACKUP"
  fi
fi

echo "Downloading CARLA server package:"
echo "  $URL"
echo "to:"
echo "  $ARCHIVE"
if [[ ! -f "$ARCHIVE" ]]; then
  curl --http1.1 -L --fail --retry 12 --retry-delay 5 --retry-all-errors \
    --continue-at - \
    --output "$PARTIAL" \
    "$URL"
  mv "$PARTIAL" "$ARCHIVE"
fi

echo "Validating archive:"
echo "  $ARCHIVE"
tar -tzf "$ARCHIVE" >/dev/null

echo "Extracting to:"
echo "  $INSTALL_DIR"
tar -xzf "$ARCHIVE" -C "$INSTALL_DIR"

echo "CARLA server installed at:"
echo "  $INSTALL_DIR"
echo "Start it with:"
echo "  $ROOT/scripts/start_carla_server.sh"
