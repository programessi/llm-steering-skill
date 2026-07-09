#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="$ROOT/third_party/downloads"
INSTALL_DIR="$ROOT/third_party/carla"
ARCHIVE="$DOWNLOAD_DIR/CARLA_0.9.15.tar.gz"
PARTIAL="$ARCHIVE.part"
SEGMENT_DIR="$DOWNLOAD_DIR/CARLA_0.9.15.segments"
URL="${CARLA_SERVER_URL:-https://carla-releases.b-cdn.net/Linux/CARLA_0.9.15.tar.gz}"
TOTAL_BYTES="${CARLA_SERVER_BYTES:-8386636048}"
JOBS="${CARLA_SERVER_JOBS:-16}"

mkdir -p "$DOWNLOAD_DIR" "$INSTALL_DIR" "$SEGMENT_DIR"

segment_name() {
  printf "%s/segment_%03d" "$SEGMENT_DIR" "$1"
}

check_size() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] && [[ "$(stat -c%s "$path")" -eq "$expected" ]]
}

download_segment() {
  local i="$1"
  local start="$2"
  local end="$3"
  local expected=$((end - start + 1))
  local seg
  local part
  local tmp
  seg="$(segment_name "$i")"
  part="$seg.part"
  tmp="$seg.download"

  if check_size "$seg" "$expected"; then
    echo "segment $i already complete ($expected bytes)"
    return 0
  fi

  if [[ -f "$seg" ]]; then
    mv "$seg" "$seg.bad.$(date +%Y%m%d%H%M%S)"
  fi
  if [[ -f "$part" ]]; then
    mv "$part" "$part.bad.$(date +%Y%m%d%H%M%S)"
  fi
  if [[ -f "$tmp" ]]; then
    mv "$tmp" "$tmp.bad.$(date +%Y%m%d%H%M%S)"
  fi

  echo "segment $i downloading bytes $start-$end"
  curl --http1.1 --silent --show-error -L --fail \
    --retry 12 --retry-delay 5 --retry-all-errors \
    --range "$start-$end" \
    --output "$tmp" \
    "$URL"

  check_size "$tmp" "$expected"
  mv "$tmp" "$seg"
  echo "segment $i complete"
}

if [[ -f "$ARCHIVE" ]]; then
  if tar -tzf "$ARCHIVE" >/dev/null 2>&1; then
    echo "Existing CARLA archive passed validation:"
    echo "  $ARCHIVE"
    exit 0
  fi
  mv "$ARCHIVE" "$ARCHIVE.bad.$(date +%Y%m%d%H%M%S)"
fi

if [[ -f "$PARTIAL" ]]; then
  mv "$PARTIAL" "$PARTIAL.bad.$(date +%Y%m%d%H%M%S)"
fi

echo "Downloading CARLA server package in $JOBS parallel segments:"
echo "  $URL"
echo "total bytes:"
echo "  $TOTAL_BYTES"

chunk=$(((TOTAL_BYTES + JOBS - 1) / JOBS))
pids=()
for ((i = 0; i < JOBS; i++)); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if [[ "$start" -ge "$TOTAL_BYTES" ]]; then
    continue
  fi
  if [[ "$end" -ge "$TOTAL_BYTES" ]]; then
    end=$((TOTAL_BYTES - 1))
  fi
  download_segment "$i" "$start" "$end" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

assembled="$ARCHIVE.assemble"
: > "$assembled"
for ((i = 0; i < JOBS; i++)); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  if [[ "$start" -ge "$TOTAL_BYTES" ]]; then
    continue
  fi
  if [[ "$end" -ge "$TOTAL_BYTES" ]]; then
    end=$((TOTAL_BYTES - 1))
  fi
  seg="$(segment_name "$i")"
  expected=$((end - start + 1))
  check_size "$seg" "$expected"
  cat "$seg" >> "$assembled"
done

mv "$assembled" "$ARCHIVE"

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
