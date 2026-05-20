"""Per-run multimodal logger.

Given a paired-capture run_dir produced by `bootstrap.sh ... --save-raw
--record-radar`, compute the per-modality numbers Phil's session protocol
asks for and emit a markdown block ready to append to the session log:

  - Polar median RR + median HR + HR range
  - LiDAR peak-pick RR + centroid RR + Δ between them (cardiac quality
    indicator from PR Z)
  - Radar (whole-frame mean_z naive) BPM + in-band SNR
  - cz_std + cz_span (sanity check that signal is real breathing motion)

Usage (Jetson, via phase1/run.sh):
    phase1/run.sh phase1/per_run_log.py <run_dir> [--label "Run α"] \
        [--append phase1/notes/multimodal_2026-05-20.md]

Usage (Mac, against a pulled-back run dir):
    .venv-local/bin/python phase1/per_run_log.py runs/20260520_xxxxx
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
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal, settled_median


_RR_BAND_BPM = (10.0, 30.0)


def polar_stats(gt_csv: Path) -> dict:
    """Median RR/HR + HR range from polar_h10 rows."""
    rrs, hrs = [], []
    with gt_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("source") != "polar_h10":
                continue
            rr = row.get("rr", "").strip()
            hr = row.get("hr", "").strip()
            if rr:
                try: rrs.append(float(rr))
                except ValueError: pass
            if hr:
                try: hrs.append(float(hr))
                except ValueError: pass
    return {
        "rr_median": float(np.median(rrs)) if rrs else None,
        "rr_n":      len(rrs),
        "hr_median": float(np.median(hrs)) if hrs else None,
        "hr_min":    float(np.min(hrs))    if hrs else None,
        "hr_max":    float(np.max(hrs))    if hrs else None,
        "hr_n":      len(hrs),
    }


def lidar_methods(run_dir: Path, fps: float) -> dict:
    """Re-score centroid_z under both rr_method settings + settled-median."""
    cz = np.load(run_dir / "centroid_z.npy")
    out = {}
    cfg_base = KVConfig()
    for method in ("peak-pick", "centroid"):
        cfg = replace(cfg_base, rr_method=method)
        history = []
        compute_every = max(1, int(round(
            cfg.rr_window_seconds * (1.0 - cfg.rr_window_overlap) * fps)))
        for i in range(compute_every, len(cz) + 1, compute_every):
            r = extract_rr_from_signal(cz[:i], fps, cfg)
            history.append({"rr_bpm": float(r["rr_bpm"]),
                            "snr": float(r["snr"]),
                            "confidence": str(r["confidence"])})
        settled = settled_median(history, cfg)
        out[method] = {
            "rr_bpm":     round(settled["rr_bpm"], 2),
            "snr":        round(settled["snr"], 2),
            "confidence": settled["confidence"],
            "settled":    settled["settled"],
            "settle_note": settled["settle_note"],
        }
    out["cz_std"]  = float(np.nanstd(cz))
    out["cz_span"] = float(np.nanmax(cz) - np.nanmin(cz))
    out["n_samples"] = int(len(cz))
    return out


def radar_naive(run_dir: Path) -> dict | None:
    """Whole-frame mean_z FFT (no spatial filter). Returns None if no radar.npz."""
    npz_p = run_dir / "radar.npz"
    if not npz_p.exists():
        return None
    d = np.load(npz_p)
    ts_key = "timestamps_rel" if "timestamps_rel" in d.files else "timestamps"
    ts = d[ts_key]
    frame_keys = sorted(k for k in d.files if k.startswith("f"))
    if not frame_keys or len(ts) < 32:
        return {"error": "too few frames"}

    mz = np.array([
        float(d[k][:, 2].mean()) if len(d[k]) else float("nan")
        for k in frame_keys
    ])
    valid = ~np.isnan(mz)
    if valid.sum() < 32:
        return {"error": f"only {int(valid.sum())} non-empty frames"}

    # NaN-interp
    idx = np.arange(len(mz))
    if (~valid).any():
        mz[~valid] = np.interp(idx[~valid], idx[valid], mz[valid])

    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 10.0

    # Detrend + Hanning + zero-pad FFT (same recipe as respiratory.py)
    from scipy.signal import detrend
    sig = detrend(mz, type="linear") * np.hanning(len(mz))
    n_fft = int(2 ** np.ceil(np.log2(len(sig) * 16)))
    spec = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    band = (freq * 60.0 >= _RR_BAND_BPM[0]) & (freq * 60.0 <= _RR_BAND_BPM[1])
    if not band.any():
        return {"error": "band empty"}
    peak_idx = int(np.argmax(spec[band]))
    peak_pow = float(spec[band].max())
    mean_pow = float(spec[band].mean())
    return {
        "rr_bpm": round(float(freq[band][peak_idx] * 60.0), 2),
        "snr":    round(peak_pow / (mean_pow + 1e-12), 2),
        "fps":    round(fps, 2),
        "n_frames": len(frame_keys),
        "n_empty":  int((~valid).sum()),
    }


def fmt_block(run_dir: Path, label: str | None = None) -> str:
    meta = json.loads((run_dir / "meta.json").read_text())
    fps  = meta["frames_seen"] / meta["wall_seconds"]
    gt_p = run_dir / "gt.csv"
    polar = polar_stats(gt_p) if gt_p.exists() else {}

    lidar = lidar_methods(run_dir, fps)
    radar = radar_naive(run_dir)

    polar_rr = polar.get("rr_median")
    lp_d = (round(lidar["peak-pick"]["rr_bpm"] - polar_rr, 2)
            if polar_rr is not None else None)
    lc_d = (round(lidar["centroid"]["rr_bpm"]  - polar_rr, 2)
            if polar_rr is not None else None)
    rd_d = (round(radar["rr_bpm"] - polar_rr, 2)
            if (radar and "rr_bpm" in radar and polar_rr is not None) else None)
    method_split = round(
        lidar["centroid"]["rr_bpm"] - lidar["peak-pick"]["rr_bpm"], 2)

    name = label or run_dir.name
    hdr = f"### {name} — `{run_dir.name}`"

    lines = [
        hdr, "",
        f"- duration: {meta.get('wall_seconds', 0):.1f}s · "
        f"{meta.get('frames_seen', 0)} frames · "
        f"corrected fps {fps:.2f}",
    ]

    if polar:
        lines.append(
            f"- **Polar:** RR median **{polar['rr_median']:.2f}** "
            f"(n={polar['rr_n']}) · HR median "
            f"{polar['hr_median']:.1f} BPM "
            f"(range {polar['hr_min']:.0f}-{polar['hr_max']:.0f}, n={polar['hr_n']})"
        )
    else:
        lines.append("- **Polar:** no gt.csv")

    lines += [
        "",
        "| Modality | Method | BPM | SNR | Conf | Δ vs Polar |",
        "|---|---|---:|---:|:---:|---:|",
    ]
    if polar_rr is not None:
        lines.append(f"| Polar H10 | ground truth | **{polar_rr:.2f}** | n/a | — | 0 |")
    lines += [
        f"| LiDAR (peak-pick) | PR Z safe default | **{lidar['peak-pick']['rr_bpm']}** "
        f"| {lidar['peak-pick']['snr']} | {lidar['peak-pick']['confidence']} "
        f"| {lp_d if lp_d is not None else '—'} |",
        f"| LiDAR (centroid)  | PR I, cardiac-quality indicator | {lidar['centroid']['rr_bpm']} "
        f"| {lidar['centroid']['snr']} | {lidar['centroid']['confidence']} "
        f"| {lc_d if lc_d is not None else '—'} |",
    ]
    if radar and "rr_bpm" in radar:
        lines.append(
            f"| Radar (whole-frame mean_z) | naive FFT | **{radar['rr_bpm']}** "
            f"| {radar['snr']} | — | {rd_d if rd_d is not None else '—'} |"
        )
    elif radar:
        lines.append(f"| Radar | — | — | — | — | ERR: {radar.get('error','?')} |")
    else:
        lines.append("| Radar | no radar.npz sidecar | — | — | — | — |")

    lines += [
        "",
        f"- **Method Δ** (centroid − peak-pick): **{method_split:+.2f} BPM** "
        f"— cardiac contamination indicator (large Δ ⇒ centroid biased high)",
        f"- cz_std = {lidar['cz_std']*1000:.2f} mm · "
        f"cz_span = {lidar['cz_span']*1000:.2f} mm · n_samples = {lidar['n_samples']}",
    ]
    if radar and "rr_bpm" in radar:
        lines.append(
            f"- radar: {radar['n_frames']} frames @ {radar['fps']} fps · "
            f"{radar['n_empty']} empty detections"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default=None,
                    help="Optional friendly run label (e.g. \"Run α\")")
    ap.add_argument("--append", type=Path, default=None,
                    help="Markdown file to append the block to (creates if absent)")
    args = ap.parse_args()

    p = Path(args.run_dir).expanduser()
    if not p.is_absolute():
        # Try repo-relative then phase1/runs/<name>
        cands = [Path.cwd() / args.run_dir, _HERE / "runs" / args.run_dir]
        for c in cands:
            if c.exists():
                p = c; break
    block = fmt_block(p, args.label)
    print(block)
    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a") as f:
            f.write(block + "\n")
        print(f"# appended → {args.append}", file=sys.stderr)


if __name__ == "__main__":
    main()
