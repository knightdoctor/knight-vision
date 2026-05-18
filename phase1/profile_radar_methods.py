"""Profile radar aggregation methods (mean_z / median_z / min_z) across
all paired (radar + Polar) recordings to test whether min_z is universally
more robust than mean_z."""
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/phil/knight-vision")
from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal

RUNS_DIR = Path("/home/phil/knight-vision/phase1/runs")
DATASETS = [
    # label, radar npz path, polar gt.csv run dir
    ("Run 7 Phil  (14:52 smoke)",  "/tmp/radar_smoke3.npz",
     "20260518_145226"),
    ("Run 8 Phil  (15:03 smoke)",  "/tmp/radar_smoke_repro2b.npz",
     "20260518_150354"),
    ("Run 9 Phil  (15:12 sidecar)", "20260518_151241/radar.npz",
     "20260518_151241"),
    ("Run 10 partner (16:01)",     "20260518_160119/radar.npz",
     "20260518_160119"),
    ("Run 11 partner (16:08)",     "20260518_160843/radar.npz",
     "20260518_160843"),
    ("Run 12 partner (16:19)",     "20260518_161944/radar.npz",
     "20260518_161944"),
]

cfg = KVConfig()


def load_radar(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = RUNS_DIR / path
    if not p.exists():
        return None, None
    d = np.load(p)
    if "timestamps_rel" in d.files:
        ts = np.asarray(d["timestamps_rel"], dtype=float)
    elif "timestamps" in d.files:
        ts = np.asarray(d["timestamps"], dtype=float)
    else:
        return None, None
    keys = sorted(k for k in d.files if k.startswith("f"))
    frames = [d[k] for k in keys]
    return ts, frames


def polar_median_rr(run_dir: str):
    p = RUNS_DIR / run_dir / "gt.csv"
    if not p.exists():
        return None, 0
    rrs = []
    for r in csv.DictReader(open(p)):
        if r.get("source") == "polar_h10" and r.get("rr"):
            try:
                rrs.append(float(r["rr"]))
            except ValueError:
                pass
    return (float(statistics.median(rrs)) if rrs else None), len(rrs)


def bpm_via_aggregation(frames, ts, aggregator) -> tuple:
    """aggregator: callable(pts) -> float. Returns (bpm, snr)."""
    if not frames or len(frames) < 30:
        return None, None
    cz = np.array([aggregator(f) if len(f) else float("nan")
                   for f in frames], dtype=float)
    valid = ~np.isnan(cz)
    if valid.sum() < 30:
        return None, None
    cz[~valid] = float(np.nanmean(cz))
    span = float(ts[-1] - ts[0]) if len(ts) > 1 else 60.0
    fps = (len(frames) - 1) / span if span > 0 else 10.0
    r = extract_rr_from_signal(cz, fps, cfg)
    return float(r["rr_bpm"]), float(r["snr"])


METHODS = {
    "mean_z":   lambda pts: float(pts[:, 2].mean()),
    "median_z": lambda pts: float(np.median(pts[:, 2])),
    "min_z":    lambda pts: float(pts[:, 2].min()),
}


def main():
    print(f"{'Run':<28}| {'Polar':>7} | "
          + " | ".join(f"{m:>15}" for m in METHODS) + " |   best")
    print("-" * 110)
    abs_deltas_by_method = {m: [] for m in METHODS}
    for label, radar_path, polar_dir in DATASETS:
        ts, frames = load_radar(radar_path)
        polar, n_polar = polar_median_rr(polar_dir)
        if frames is None:
            print(f"{label:<28}| (no radar data)")
            continue
        polar_str = f"{polar:7.2f}" if polar else "    ---"
        cols = []
        deltas = {}
        for m, fn in METHODS.items():
            bpm, snr = bpm_via_aggregation(frames, ts, fn)
            if bpm is None:
                cols.append(f"{'---':>15}")
                continue
            if polar is not None:
                d = bpm - polar
                deltas[m] = d
                abs_deltas_by_method[m].append(abs(d))
                cols.append(f"{bpm:5.2f} (Δ{d:+5.2f})")
            else:
                cols.append(f"{bpm:5.2f}  (—) ")
        best = min(deltas, key=lambda k: abs(deltas[k])) if deltas else "-"
        print(f"{label:<28}| {polar_str} | "
              + " | ".join(cols) + f" | {best}")
    print()
    print("Mean abs Δ vs Polar across paired runs:")
    for m, ds in abs_deltas_by_method.items():
        if ds:
            print(f"  {m:>10} : {statistics.mean(ds):.2f} BPM "
                  f"(n={len(ds)}, max {max(ds):.2f})")


if __name__ == "__main__":
    main()
