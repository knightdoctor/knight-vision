#!/usr/bin/env bash
# Knight Vision phase1 bootstrap — unified entry for paired captures.
#
# Usage:
#   bootstrap.sh [--recapture-bg] [--bg-duration N] [run_phase1 live flags...]
#
# Examples:
#   bootstrap.sh --recapture-bg              # bg + session, M1 defaults
#   bootstrap.sh                             # skip bg, M1 defaults
#   bootstrap.sh --duration 90               # override session duration
#
# M1-grade defaults baked into LIVE_DEFAULTS below: viewer on, peak-pick
# RR method, raw-frames saved, radar sidecar recording, 30-min session.
# Recordings inside the session are triggered from the viewer (REC/STOP
# buttons / R/S keys); each REC press writes its own run_dir. Any flag
# repeated in the user's CLI overrides the default (argparse takes last).
set -euo pipefail

PHASE1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=("$PHASE1_DIR/run.sh" "$PHASE1_DIR/run_phase1.py")

# M1-grade defaults — see header comment. Edit here, not per-launch.
LIVE_DEFAULTS=(
    --viewer
    --rr-method peak-pick
    --save-raw
    --record-radar
    --duration 1800
)

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
echo "=== Live capture (M1 defaults + user overrides) ==="
echo "    defaults: ${LIVE_DEFAULTS[*]}"
if [[ ${#LIVE_ARGS[@]} -gt 0 ]]; then
  echo "    overrides: ${LIVE_ARGS[*]}"
fi
exec "${RUN[@]}" --mode live --lidar "${LIVE_DEFAULTS[@]}" "${LIVE_ARGS[@]}"
