"""Profile per-window vs full-record vs PR M settled-median vs Polar GT
across the M1 paired-run cohort. Tests whether PR M's SNR-weighted
selection systematically biases the final estimate."""
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/phil/knight-vision")
from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal

RUNS = [
    ("Run 4", "20260516_165647"),
    ("Run 5", "20260516_170653"),
    ("Run 6", "20260517_072337"),
    ("Run 7", "20260518_145226"),
    ("Run 8", "20260518_150354"),
    ("Run 9", "20260518_151241"),
]
RUNS_DIR = Path("/home/phil/knight-vision/phase1/runs")
cfg = KVConfig()


def polar_median(run_dir: Path):
    p = run_dir / "gt.csv"
    if not p.exists():
        return None
    rrs = []
    for r in csv.DictReader(open(p)):
        if r.get("source") == "polar_h10" and r.get("rr"):
            try:
                rrs.append(float(r["rr"]))
            except ValueError:
                pass
    return float(statistics.median(rrs)) if rrs else None


def full_record_fft(run_dir: Path, fps: float) -> float:
    cz = np.load(run_dir / "centroid_z.npy")
    r = extract_rr_from_signal(cz, fps, cfg)
    return float(r["rr_bpm"]), float(r["snr"])


def main():
    print(f"{'Run':<7} | {'Polar':>6} | {'PR M (live)':>11} | "
          f"{'median windows':>14} | {'full FFT':>10} | "
          f"{'Δ PR M':>7} | {'Δ med':>7} | {'Δ full':>7}")
    print("-" * 100)
    for tag, ts in RUNS:
        rd = RUNS_DIR / ts
        m = json.loads((rd / "meta.json").read_text())
        fps = m["frames_seen"] / m["wall_seconds"]
        polar = polar_median(rd)
        pr_m = m["result_final"]["rr_bpm"] if m.get("result_final") else None
        windows = m.get("rr_windows", [])
        win_med = float(statistics.median([w["rr_bpm"] for w in windows])) if windows else None
        full_bpm, full_snr = full_record_fft(rd, fps)
        polar_str = f"{polar:6.2f}" if polar else "  ----"
        d_prm = f"{pr_m - polar:+.2f}" if (polar and pr_m) else "  ---"
        d_med = f"{win_med - polar:+.2f}" if (polar and win_med) else "  ---"
        d_full = f"{full_bpm - polar:+.2f}" if polar else "  ---"
        pr_m_str = f"{pr_m:11.2f}" if pr_m else "      ---"
        win_str = f"{win_med:14.2f}" if win_med else "         ---"
        print(f"{tag:<7} | {polar_str} | {pr_m_str} | {win_str} | "
              f"{full_bpm:10.2f} | {d_prm:>7} | {d_med:>7} | {d_full:>7}")


if __name__ == "__main__":
    main()
