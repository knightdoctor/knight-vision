#!/usr/bin/env bash
# Knight Vision MVP simulation — Week 1 smoke test orchestrator.
#
# Builds the scene, renders 30 s of synthetic 15-BPM breathing through
# stereo IR, computes depth + point clouds, runs the FFT pipeline, and
# asserts the recovered RR is within ±2 BPM of ground truth.
#
# Expected total runtime on a modern Mac: ~3-8 minutes (Eevee rendering
# dominates).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
PYTHON="${PYTHON:-$REPO/.venv-local/bin/python}"

# Knobs — override via env for exploration.
RR_BPM="${RR_BPM:-15}"
AMPLITUDE_MM="${AMPLITUDE_MM:-5}"
DURATION_S="${DURATION_S:-30}"
FPS="${FPS:-15}"
TOLERANCE="${TOLERANCE:-2}"

SCENE="$HERE/scene.blend"
FRAMES="$HERE/frames"
DEPTH="$HERE/depth_out"
REPORT="$HERE/smoke_report.json"

echo "=== [1/4] Build scene ==="
"$BLENDER" --background --python "$HERE/scene_build.py" -- \
    --out "$SCENE"

echo ""
echo "=== [2/4] Render ${DURATION_S}s @ ${FPS} fps (${RR_BPM} BPM, ${AMPLITUDE_MM} mm) ==="
rm -rf "$FRAMES"
"$BLENDER" --background --python "$HERE/render.py" -- \
    --scene "$SCENE" \
    --rr-bpm "$RR_BPM" --amplitude-mm "$AMPLITUDE_MM" \
    --duration "$DURATION_S" --fps "$FPS" \
    --out "$FRAMES"

echo ""
echo "=== [3/4] Depth + point clouds ==="
rm -rf "$DEPTH"
"$PYTHON" "$HERE/depth.py" --frames "$FRAMES" --out "$DEPTH"

echo ""
echo "=== [4/4] Smoke test ==="
"$PYTHON" "$HERE/smoke_test.py" \
    --depth-dir "$DEPTH" \
    --frames-dir "$FRAMES" \
    --ground-truth-bpm "$RR_BPM" \
    --tolerance "$TOLERANCE" \
    --out "$REPORT"
