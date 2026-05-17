"""Pipeline D — fixed 3D ROI baseline for retrospective replay.

Removes the clustering + chest-band machinery entirely. The operator
defines (or auto-derives) a static 3D box; the pipeline computes mean
Z of residual points inside that box, frame by frame, then FFTs the cz
series for RR.

Ablation purpose: isolate how much of A/B/C's accuracy is doing real
work vs. responding to noise from DBSCAN cluster jitter and chest
y-band selection. If Pipeline D matches or beats B/C on a clean run,
the clustering layer adds complexity without value (for stationary
subjects in known positions).

Usage:
  phase1/run.sh pipeline_d_replay.py runs/<TS> \\
      [--auto-roi]                  # default: chest-sized box around median
      [--x-lo X --x-hi X ...]       # explicit box
      [--min-points N]              # min residuals inside box to compute cz
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal, settled_median


def derive_auto_roi(residuals_npz, half_x=0.40, half_y=0.25,
                    half_z=0.20, sample_frames=200) -> dict:
    """Compute a chest-sized ROI centred on the recording's spatial median."""
    keys = sorted(residuals_npz.files)
    import random
    sample = random.sample(keys, min(sample_frames, len(keys)))
    xs = np.concatenate([residuals_npz[k][:, 0] for k in sample])
    ys = np.concatenate([residuals_npz[k][:, 1] for k in sample])
    zs = np.concatenate([residuals_npz[k][:, 2] for k in sample])
    cx, cy, cz = float(np.median(xs)), float(np.median(ys)), float(np.median(zs))
    # Y bias: median is usually at abdomen height; shift the box up to
    # capture chest (positive Y in shared frame).
    cy_chest = cy + half_y * 0.5
    return {
        "x_lo": cx - half_x, "x_hi": cx + half_x,
        "y_lo": cy_chest - half_y, "y_hi": cy_chest + half_y,
        "z_lo": cz - half_z,  "z_hi": cz + half_z,
        "centre_xyz": (cx, cy, cz),
    }


def replay(run_dir: Path, roi: dict, min_points: int = 30) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    data = np.load(run_dir / "residuals.npz")
    keys = sorted(data.files)
    n = len(keys)
    fps = meta["frames_seen"] / meta["wall_seconds"]

    cz_series = []
    n_box = 0
    box_pts_total = 0
    for k in keys:
        pts = data[k]
        in_box = (
            (pts[:, 0] >= roi["x_lo"]) & (pts[:, 0] <= roi["x_hi"]) &
            (pts[:, 1] >= roi["y_lo"]) & (pts[:, 1] <= roi["y_hi"]) &
            (pts[:, 2] >= roi["z_lo"]) & (pts[:, 2] <= roi["z_hi"])
        )
        box_pts = pts[in_box]
        if len(box_pts) >= min_points:
            cz_series.append(float(box_pts[:, 2].mean()))
            n_box += 1
            box_pts_total += len(box_pts)
        else:
            cz_series.append(float("nan"))

    cz_arr = np.array(cz_series, dtype=float)

    # Whole-recording FFT (matches the way offline_replay reports a single
    # final number; for sliding-window comparison, use the windowed call).
    cfg = KVConfig()
    full = extract_rr_from_signal(cz_arr, fps, cfg)

    # Sliding-window per-window history (matches PR J live behaviour)
    step_n = max(1, int(cfg.rr_window_seconds * (1 - cfg.rr_window_overlap) * fps))
    win_n  = max(64, int(cfg.rr_window_seconds * fps))
    rr_history = []
    for end in range(step_n, len(cz_arr) + 1, step_n):
        sig = cz_arr[max(0, end - win_n):end]
        r = extract_rr_from_signal(sig, fps, cfg)
        rr_history.append({
            "rr_bpm": float(r["rr_bpm"]),
            "snr":    float(r["snr"]),
            "confidence": str(r["confidence"]),
        })
    settled = settled_median(rr_history, cfg)

    valid = ~np.isnan(cz_arr)
    return {
        "run":           run_dir.name,
        "fps":           round(fps, 2),
        "frames":        n,
        "frames_in_box": n_box,
        "frame_coverage": round(100 * n_box / n, 1),
        "avg_pts_in_box": round(box_pts_total / max(1, n_box), 1),
        "cz_span_mm":    round((cz_arr[valid].max() - cz_arr[valid].min()) * 1000, 1)
                          if valid.any() else None,
        "cz_std_mm":     round(cz_arr[valid].std() * 1000, 2) if valid.any() else None,
        "roi":           {k: round(v, 3) if isinstance(v, float) else
                          [round(x, 3) for x in v]
                          for k, v in roi.items()},
        "full_recording_fft": {
            "rr_bpm": round(float(full["rr_bpm"]), 2),
            "snr":    round(float(full["snr"]), 2),
            "conf":   str(full["confidence"]),
        },
        "sliding_settled": {
            "rr_bpm": round(settled["rr_bpm"], 2),
            "snr":    round(settled["snr"], 2),
            "conf":   settled["confidence"],
            "note":   settled["settle_note"],
            "windows":settled["window_n"],
            "total":  settled["windows_total"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--x-lo", type=float, default=None)
    ap.add_argument("--x-hi", type=float, default=None)
    ap.add_argument("--y-lo", type=float, default=None)
    ap.add_argument("--y-hi", type=float, default=None)
    ap.add_argument("--z-lo", type=float, default=None)
    ap.add_argument("--z-hi", type=float, default=None)
    ap.add_argument("--min-points", type=int, default=30)
    args = ap.parse_args()

    for arg in args.run_dirs:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = _HERE / "runs" / p
        if not (p / "residuals.npz").exists():
            print(f"=== {p.name}: SKIPPED (no residuals.npz) ===")
            continue

        # Build ROI: explicit if any flag given, else auto from this run
        if all(getattr(args, k) is not None
               for k in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")):
            roi = {
                "x_lo": args.x_lo, "x_hi": args.x_hi,
                "y_lo": args.y_lo, "y_hi": args.y_hi,
                "z_lo": args.z_lo, "z_hi": args.z_hi,
            }
        else:
            roi = derive_auto_roi(np.load(p / "residuals.npz"))

        result = replay(p, roi, min_points=args.min_points)
        print("=" * 72)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
