"""HR/2 tracking on the LiDAR centroid_z spectrum — task #64.

Tests the cardiac-BCG-at-HR/2 hypothesis stated provisionally in
`apnoea_pipeline_architecture.md`. Protocol: subject elevates HR via
brief exercise then sits still while recording; HR drops from ~100 →
~70 over the recording. Polar H10 gives ground-truth HR throughout.

For each sliding window of the LiDAR centroid_z signal:
1. Detrend + FFT (same recipe as respiratory.py).
2. Find the dominant peak inside a wider "cardiac" band (default
   25-60 BPM, which spans HR/2 across HR 50-120).
3. Match the window's centre-time to the closest Polar HR sample and
   compute HR/2 as the expected cardiac-BCG frequency.
4. Report (window_centre_s, cardiac_peak_bpm, polar_hr, polar_hr_over_2)
   per window and the overall correlation / RMS-tracking-error.

Verdict criteria (per apnoea doc open-question box):
- **Confirms HR/2 BCG hypothesis** if the centroid_z cardiac peak tracks
  HR/2 with R^2 ≥ 0.7 across the HR ramp (and absolute |peak − HR/2| < 3
  BPM on average). The peak should descend from ~50 BPM at HR=100 toward
  ~35 BPM at HR=70.
- **Refutes** if the peak stays fixed (e.g. 30-33 BPM regardless of HR)
  or shows no statistically significant correlation with HR. In that
  case the contamination has a different physical origin (sensor
  artifact, environmental vibration, postural micro-tremor) and the
  cardiac-defence sections of the apnoea doc need revising.

Usage:
    phase1/run.sh phase1/hr2_tracking.py <run_dir> \
        [--cardiac-lo 25] [--cardiac-hi 60] \
        [--window-seconds 20] [--step-seconds 5] \
        [--out phase1/notes/hr2_tracking_<runname>.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def polar_hr_track(gt_csv: Path) -> list[tuple[float, float]]:
    """Return [(t_s_from_recording_start, hr_bpm), ...] for polar_h10 rows."""
    out = []
    rows = list(csv.DictReader(gt_csv.open()))
    if not rows:
        return out
    t0 = datetime.fromisoformat(rows[0]["ts_iso"])
    for r in rows:
        if r.get("source") != "polar_h10":
            continue
        try:
            hr = float(r["hr"]) if r.get("hr") else None
        except ValueError:
            hr = None
        if hr is None:
            continue
        ts = (datetime.fromisoformat(r["ts_iso"]) - t0).total_seconds()
        out.append((ts, hr))
    return out


def cardiac_peak_bpm(signal: np.ndarray, fps: float,
                     band_bpm: tuple) -> tuple[float, float] | None:
    """Hanning + zero-padded FFT; return (peak_bpm, in_band_snr) in cardiac band."""
    if len(signal) < 32:
        return None
    sig = detrend(signal, type="linear") * np.hanning(len(signal))
    n_fft = int(2 ** np.ceil(np.log2(len(sig) * 16)))
    spec = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    band = (freq * 60.0 >= band_bpm[0]) & (freq * 60.0 <= band_bpm[1])
    if not band.any():
        return None
    peak_idx = int(np.argmax(spec[band]))
    peak_bpm = float(freq[band][peak_idx] * 60.0)
    snr = float(spec[band].max() / (spec[band].mean() + 1e-12))
    return peak_bpm, snr


def nearest_hr(t_s: float, hr_track: list[tuple[float, float]]) -> float | None:
    if not hr_track:
        return None
    ts = np.array([t for t, _ in hr_track])
    hrs = np.array([h for _, h in hr_track])
    idx = int(np.argmin(np.abs(ts - t_s)))
    if abs(ts[idx] - t_s) > 15.0:   # > 15 s gap → unreliable
        return None
    return float(hrs[idx])


def run(run_dir: Path, cardiac_lo: float, cardiac_hi: float,
        window_s: float, step_s: float) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    cz   = np.load(run_dir / "centroid_z.npy")
    fps  = meta["frames_seen"] / meta["wall_seconds"]
    gt_p = run_dir / "gt.csv"
    hr_track = polar_hr_track(gt_p) if gt_p.exists() else []

    n_win = int(window_s * fps)
    n_step = max(1, int(step_s * fps))
    rows = []
    for i_end in range(n_win, len(cz) + 1, n_step):
        slice_ = cz[i_end - n_win : i_end]
        # NaN-interp inside the window so detrend doesn't choke
        if np.isnan(slice_).any():
            valid = ~np.isnan(slice_)
            if valid.sum() < 32:
                continue
            idx = np.arange(len(slice_))
            slice_ = slice_.copy()
            slice_[~valid] = np.interp(idx[~valid], idx[valid], slice_[valid])
        peak = cardiac_peak_bpm(slice_, fps, (cardiac_lo, cardiac_hi))
        if peak is None:
            continue
        t_centre = (i_end - n_win / 2) / fps
        hr = nearest_hr(t_centre, hr_track)
        rows.append({
            "t_centre_s":   round(t_centre, 1),
            "cardiac_bpm":  round(peak[0], 2),
            "cardiac_snr":  round(peak[1], 2),
            "polar_hr":     round(hr, 1) if hr is not None else None,
            "polar_hr_2":   round(hr / 2.0, 2) if hr is not None else None,
        })

    # Correlation + tracking-error stats (drop windows without matched HR)
    paired = [r for r in rows if r["polar_hr_2"] is not None]
    summary: dict = {
        "n_windows":         len(rows),
        "n_paired":          len(paired),
        "hr_track_n":        len(hr_track),
        "hr_track_range":    (round(min(h for _, h in hr_track), 1),
                              round(max(h for _, h in hr_track), 1))
                              if hr_track else None,
    }
    if paired:
        cz_peak = np.array([r["cardiac_bpm"] for r in paired])
        hr_half = np.array([r["polar_hr_2"] for r in paired])
        delta   = cz_peak - hr_half
        if len(paired) >= 3 and np.std(hr_half) > 0.1:
            r_val = float(np.corrcoef(cz_peak, hr_half)[0, 1])
        else:
            r_val = None
        summary.update({
            "cardiac_peak_range": (round(cz_peak.min(), 2),
                                   round(cz_peak.max(), 2)),
            "mean_delta":         round(float(delta.mean()), 2),
            "rms_delta":          round(float(np.sqrt(np.mean(delta ** 2))), 2),
            "abs_delta_median":   round(float(np.median(np.abs(delta))), 2),
            "correlation_r":      round(r_val, 3) if r_val is not None else None,
            "r_squared":          round(r_val ** 2, 3) if r_val is not None else None,
        })

    return {"meta": meta, "fps": fps, "rows": rows, "summary": summary,
            "params": {"cardiac_lo": cardiac_lo, "cardiac_hi": cardiac_hi,
                       "window_s": window_s, "step_s": step_s}}


def fmt_md(run_dir: Path, result: dict) -> str:
    s = result["summary"]
    p = result["params"]
    rows = result["rows"]

    lines = [
        f"# HR/2 cardiac-tracking — {run_dir.name}",
        "",
        f"*Generated by `phase1/hr2_tracking.py` (task #64).*",
        "",
        f"- centroid_z duration: {result['meta'].get('wall_seconds', 0):.1f}s · "
        f"corrected fps {result['fps']:.2f}",
        f"- analysis: rolling {p['window_s']:.0f}s window, "
        f"step {p['step_s']:.0f}s, cardiac band "
        f"{p['cardiac_lo']:.0f}-{p['cardiac_hi']:.0f} BPM",
        f"- Polar HR samples: n={s['hr_track_n']}"
        + (f", range {s['hr_track_range'][0]}-{s['hr_track_range'][1]} BPM"
           if s['hr_track_range'] else ""),
        f"- windows analysed: {s['n_windows']} ({s['n_paired']} matched to Polar HR)",
        "",
    ]
    if s.get("correlation_r") is not None:
        verdict = (
            "**HR/2 hypothesis CONFIRMED**" if (s["r_squared"] >= 0.7
                                                and s["abs_delta_median"] <= 3.0)
            else "**HR/2 hypothesis REFUTED**" if s["r_squared"] < 0.3
            else "**HR/2 hypothesis EQUIVOCAL** — partial tracking; needs longer or higher-Δ-HR recording"
        )
        lines += [
            "## Verdict",
            "",
            verdict,
            "",
            f"- correlation r = {s['correlation_r']}, R² = {s['r_squared']}",
            f"- mean Δ (cardiac peak − HR/2) = {s['mean_delta']:+.2f} BPM",
            f"- |Δ| median = {s['abs_delta_median']:.2f} BPM, "
            f"RMS Δ = {s['rms_delta']:.2f} BPM",
            f"- cardiac peak range: "
            f"{s['cardiac_peak_range'][0]}-{s['cardiac_peak_range'][1]} BPM",
            "",
        ]
    elif s["n_paired"] == 0:
        lines += ["## Verdict",
                  "",
                  "Cannot evaluate — no windows matched to Polar HR samples.",
                  ""]

    lines += [
        "## Per-window detail",
        "",
        "| Window centre (s) | Cardiac peak (BPM) | Cardiac SNR | Polar HR | Polar HR/2 | Δ (peak − HR/2) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        d = (round(r["cardiac_bpm"] - r["polar_hr_2"], 2)
             if r["polar_hr_2"] is not None else "—")
        lines.append(
            f"| {r['t_centre_s']} | {r['cardiac_bpm']} | {r['cardiac_snr']} | "
            f"{r['polar_hr'] if r['polar_hr'] is not None else '—'} | "
            f"{r['polar_hr_2'] if r['polar_hr_2'] is not None else '—'} | {d} |"
        )
    return "\n".join(lines) + "\n"


def plot(run_dir: Path, result: dict, out_png: Path) -> None:
    rows = result["rows"]
    if not rows:
        return
    t = np.array([r["t_centre_s"] for r in rows])
    cz_peak = np.array([r["cardiac_bpm"] for r in rows])
    hr2 = np.array([r["polar_hr_2"] if r["polar_hr_2"] is not None else np.nan
                    for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(t, cz_peak, "o-", color="#a44",
            label="LiDAR cardiac peak (centroid_z FFT)")
    ax.plot(t, hr2, "x-", color="#247",
            label="Polar HR / 2")
    ax.set_xlabel("time (s, recording-relative)")
    ax.set_ylabel("BPM")
    ax.set_title(f"{run_dir.name}: cardiac peak vs Polar HR/2 over time")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    paired = ~np.isnan(hr2)
    if paired.any():
        ax.scatter(hr2[paired], cz_peak[paired], color="#a44", s=30)
        lo = float(min(hr2[paired].min(), cz_peak[paired].min()))
        hi = float(max(hr2[paired].max(), cz_peak[paired].max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x (perfect tracking)")
        ax.set_xlabel("Polar HR / 2 (BPM)")
        ax.set_ylabel("LiDAR cardiac peak (BPM)")
        ax.set_title("Scatter: tracking quality")
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    print(f"# plot → {out_png}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cardiac-lo", type=float, default=25.0)
    ap.add_argument("--cardiac-hi", type=float, default=60.0)
    ap.add_argument("--window-seconds", type=float, default=20.0,
                    dest="window_seconds")
    ap.add_argument("--step-seconds", type=float, default=5.0,
                    dest="step_seconds")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    p = Path(args.run_dir).expanduser()
    if not p.is_absolute():
        cands = [Path.cwd() / args.run_dir, _HERE / "runs" / args.run_dir]
        for c in cands:
            if c.exists():
                p = c; break

    result = run(p, args.cardiac_lo, args.cardiac_hi,
                 args.window_seconds, args.step_seconds)
    md = fmt_md(p, result)
    print(md)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"# wrote → {args.out}", file=sys.stderr)

    plot_path = args.plot or (p / "hr2_tracking.png")
    plot(p, result, plot_path)


if __name__ == "__main__":
    main()
