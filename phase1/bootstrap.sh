#!/usr/bin/env bash
# Knight Vision phase1 bootstrap — unified entry for paired captures.
#
# Usage:
#   bootstrap.sh [--recapture-bg] [--bg-duration N] [run_phase1 live flags...]
#
# Examples:
#   bootstrap.sh --recapture-bg --viewer --duration 60
#   bootstrap.sh --viewer --duration 60        # skip bg, reuse cached
set -euo pipefail

PHASE1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=("$PHASE1_DIR/run.sh" "$PHASE1_DIR/run_phase1.py")

RECAPTURE_BG=0
BG_DURATION=30
LIVE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recapture-bg) RECAPTURE_BG=1; shift ;;
    --bg-duration)  BG_DURATION="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,9p' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) LIVE_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$RECAPTURE_BG" -eq 1 ]]; then
  echo
  echo "=== Background recapture (${BG_DURATION}s) ==="
  echo "Step OUT of the sensor FOV — leave the room empty."
  read -rp "Press Enter when the FOV is clear... "
  echo
  echo ">>> Capturing background..."
  "${RUN[@]}" --mode background --duration "$BG_DURATION" --lidar
  echo ">>> Background saved to phase1/data/background_model.npz"
  echo
  read -rp "Step back into position. Press Enter to start live capture... "
fi

echo
echo "=== Live capture ==="
exec "${RUN[@]}" --mode live --lidar "${LIVE_ARGS[@]}"
