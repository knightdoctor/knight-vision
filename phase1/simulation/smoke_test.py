"""End-to-end smoke test: synthetic 15-BPM breathing → recovered RR.

Validates that the Phase 1 algorithm stack (extract_rr_from_signal) ports
to depth derived from passive stereo IR. Path:

  depth.py's per-frame point clouds
    → extract chest-region centroid_z per frame
    → feed centroid_z series into phase1.respiratory.extract_rr_from_signal
    → assert recovered RR within ±tolerance of --ground-truth-bpm.

Note: this is the *thin-adapter* smoke test specified in
mvp_simulation_spec_2026-05-19.md §"Algorithm pipeline integration".
The full Phase1Pipeline (background subtraction + DBSCAN + chest band
selection) is exercised in a follow-up step in Week 2; for the smoke
test, the synthetic scene's chest centroid is known by construction
so we can short-circuit to the FFT and validate it picks the right
frequency on synthetic stereo depth.

Exit codes:
    0 — PASS (|recovered − ground_truth| ≤ tolerance)
    1 — FAIL
    2 — error (no data / invalid input)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal


def chest_centroid_z(cloud: np.ndarray, intr: dict) -> float:
    """Mean Z of points falling inside a chest-axis-aligned box.

    Shared-frame: X right, Y up, Z forward. Chest centre in shared frame =
    (chest_world[0], chest_world[2], chest_world[1]) — Blender (x,y,z) →
    shared (x, z, y) permutation that depth.py applies.
    """
    if cloud.shape[0] == 0:
        return float("nan")
    cx, cy_blender, cz_blender = intr["chest_centre_world"]
    chest_shared = (cx, cz_blender, cy_blender)
    half_xy = 0.12   # 24 cm chest box (X and Y)
    half_z  = 0.20   # 40 cm depth window
    mask = (
        (np.abs(cloud[:, 0] - chest_shared[0]) <= half_xy)
        & (np.abs(cloud[:, 1] - chest_shared[1]) <= half_xy)
        & (np.abs(cloud[:, 2] - chest_shared[2]) <= half_z)
    )
    if not mask.any():
        return float("nan")
    return float(np.mean(cloud[mask, 2]))


def load_clouds(cloud_dir: Path) -> tuple[list[Path], list[np.ndarray]]:
    paths = sorted(cloud_dir.glob("*.npy"))
    return paths, [np.load(p) for p in paths]


def load_ground_truth(frames_dir: Path) -> dict:
    gt = frames_dir / "ground_truth.csv"
    if not gt.exists():
        return {}
    rows = list(csv.DictReader(gt.open()))
    return {
        "n_frames": len(rows),
        "duration_s": float(rows[-1]["t_s"]) if rows else 0.0,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", type=Path, required=True,
                    help="Output dir from depth.py (contains cloud/, depth/)")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="Renders dir from render.py (for ground_truth.csv "
                         "and intrinsics.json). Defaults to "
                         "<depth-dir>/../frames")
    ap.add_argument("--intrinsics", type=Path, default=None,
                    help="intrinsics.json path; falls back to alongside scene.blend")
    ap.add_argument("--ground-truth-bpm", type=float, default=15.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON report path")
    args = ap.parse_args()

    cloud_dir = args.depth_dir.resolve() / "cloud"
    if not cloud_dir.exists():
        sys.exit(f"ERROR: {cloud_dir} not found — run depth.py first")
    frames_dir = (args.frames_dir.resolve() if args.frames_dir
                  else (args.depth_dir.parent / "frames").resolve())

    intr_path = (args.intrinsics
                 or frames_dir / "intrinsics.json"
                 or frames_dir.parent / "intrinsics.json")
    for cand in (intr_path, frames_dir / "intrinsics.json",
                 frames_dir.parent / "intrinsics.json"):
        if cand.exists():
            intr = json.loads(cand.read_text())
            break
    else:
        sys.exit("ERROR: intrinsics.json not found")

    gt = load_ground_truth(frames_dir)
    if gt.get("n_frames", 0) < 30:
        sys.exit(f"ERROR: ground_truth.csv has {gt.get('n_frames',0)} frames, "
                 f"need ≥ 30")
    fps = gt["n_frames"] / gt["duration_s"]

    paths, clouds = load_clouds(cloud_dir)
    if len(clouds) != gt["n_frames"]:
        print(f"WARN: cloud count {len(clouds)} != gt frames {gt['n_frames']}",
              file=sys.stderr)

    cz_series = np.array([chest_centroid_z(c, intr) for c in clouds],
                         dtype=float)
    n_valid = int(np.sum(np.isfinite(cz_series)))
    if n_valid < 30:
        sys.exit(f"ERROR: only {n_valid} frames had chest-box points; "
                 "scene/depth chain not producing usable data")

    cfg = replace(KVConfig(), rr_method="peak-pick")
    r = extract_rr_from_signal(cz_series, fps, cfg)

    recovered = float(r["rr_bpm"])
    snr = float(r["snr"])
    conf = str(r["confidence"])
    delta = recovered - args.ground_truth_bpm
    passed = abs(delta) <= args.tolerance

    report = {
        "ground_truth_bpm":  args.ground_truth_bpm,
        "tolerance_bpm":     args.tolerance,
        "recovered_bpm":     round(recovered, 3),
        "delta_bpm":         round(delta, 3),
        "snr":               round(snr, 3),
        "confidence":        conf,
        "rr_method":         cfg.rr_method,
        "fps":               round(fps, 3),
        "n_frames":          gt["n_frames"],
        "n_chest_valid":     n_valid,
        "cz_std_mm":         round(float(np.nanstd(cz_series) * 1000.0), 3),
        "cz_span_mm":        round(
            float((np.nanmax(cz_series) - np.nanmin(cz_series)) * 1000.0), 3),
        "pass":              passed,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"# wrote {args.out}", file=sys.stderr)

    headline = (
        f"{'✅' if passed else '⛔'} smoke test "
        f"recovered {recovered:.2f} BPM (gt {args.ground_truth_bpm}, "
        f"Δ {delta:+.2f}) · SNR {snr:.2f} · {conf} · "
        f"cz_std {report['cz_std_mm']} mm"
    )
    print(headline)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
