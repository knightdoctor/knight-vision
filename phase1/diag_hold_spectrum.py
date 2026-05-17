"""Spectrum diagnostic: chest centroid power, normal-breathing vs breath-hold.

Reusable replay tool for any recording captured with the breath-hold
validation protocol (phase1/protocols/breath_hold_validation.md). Takes
a run directory, auto-detects the hold phase from manual gt.csv markers
(rr=0 marks hold start; next manual rr>0 marks recovery), runs the live
pipeline's FFT on the normal-#1 and hold slices, plots both spectra
overlaid, and prints a one-line conclusion classifying any 30-33 BPM
peak as cardiac BCG (persists during hold) or respiratory harmonic
(disappears).

First applied to Run 5 (20260516_170653) — found the persistent 30-33
BPM peak is cardiac BCG at HR/2 (HR median 64 -> 32 BPM 2nd subharmonic),
NOT a respiratory harmonic.

Usage:
  phase1/run.sh diag_hold_spectrum.py <run_dir>
                                      [--hold-start SEC --hold-end SEC]
                                      [--band-lo BPM --band-hi BPM]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal


def auto_detect_hold(run_dir: Path) -> tuple:
    """Read gt.csv, find first manual rr=0 (hold start) and next manual rr>0
    (recovery). Returns (hold_start_s, hold_end_s) relative to recording
    start, or (None, None) if markers absent."""
    gt = run_dir / "gt.csv"
    if not gt.exists():
        return None, None
    rows = list(csv.DictReader(open(gt)))
    if not rows:
        return None, None
    from datetime import datetime
    parse = lambda s: datetime.fromisoformat(s)
    t0 = parse(rows[0]["ts_iso"])
    hold_start = None
    hold_end   = None
    for r in rows:
        if r.get("source") != "manual":
            continue
        try:
            rr_val = float(r["rr"]) if r["rr"] else None
        except ValueError:
            continue
        if rr_val is None:
            continue
        t = (parse(r["ts_iso"]) - t0).total_seconds()
        if hold_start is None and rr_val == 0:
            hold_start = t
        elif hold_start is not None and hold_end is None and rr_val > 0:
            hold_end = t
            break
    return hold_start, hold_end


def median_hr_from_polar(run_dir: Path) -> float:
    """Median HR from polar_h10 entries in gt.csv (used for cardiac
    subharmonic reference line on the plot)."""
    gt = run_dir / "gt.csv"
    if not gt.exists():
        return None
    hrs = []
    for r in csv.DictReader(open(gt)):
        if r.get("source") == "polar_h10" and r.get("hr"):
            try:
                hrs.append(float(r["hr"]))
            except ValueError:
                pass
    return float(np.median(hrs)) if hrs else None


def run(run_dir: Path, hold_start_s: float, hold_end_s: float,
        band_lo: float, band_hi: float) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    cz   = np.load(run_dir / "centroid_z.npy")
    fps  = meta["frames_seen"] / meta["wall_seconds"]
    print(f"Run: {run_dir.name}")
    print(f"fps_measured: {fps:.3f}; total frames: {len(cz)}; "
          f"duration: {len(cz)/fps:.1f}s")
    print(f"Hold phase: {hold_start_s:.1f}s .. {hold_end_s:.1f}s "
          f"({hold_end_s - hold_start_s:.1f}s hold)\n")

    cfg = KVConfig()
    i_hold_start = int(hold_start_s * fps)
    i_hold_end   = int(hold_end_s   * fps)
    normal_slice = cz[:i_hold_start]
    hold_slice   = cz[i_hold_start:i_hold_end]
    if len(normal_slice) < 64 or len(hold_slice) < 64:
        sys.exit("ERROR: phase slices too short for FFT (need >= 64 samples).")

    r_normal = extract_rr_from_signal(normal_slice, fps, cfg)
    r_hold   = extract_rr_from_signal(hold_slice,   fps, cfg)
    print(f"Normal phase  RR={r_normal['rr_bpm']:5.2f} BPM  "
          f"SNR={r_normal['snr']:5.2f}  conf={r_normal['confidence']}")
    print(f"Hold phase    RR={r_hold['rr_bpm']:5.2f} BPM  "
          f"SNR={r_hold['snr']:5.2f}  conf={r_hold['confidence']}")

    def band_power(result, lo_bpm, hi_bpm):
        f = result["freq_axis"]
        p = result["power"]
        m = (f * 60 >= lo_bpm) & (f * 60 <= hi_bpm)
        return float(p[m].sum())

    p_norm  = band_power(r_normal, band_lo, band_hi)
    p_hold  = band_power(r_hold,   band_lo, band_hi)
    p_norm_tot = float(r_normal["power"].sum())
    p_hold_tot = float(r_hold["power"].sum())
    norm_frac  = p_norm / p_norm_tot if p_norm_tot else 0.0
    hold_frac  = p_hold / p_hold_tot if p_hold_tot else 0.0
    print(f"\n{band_lo:.0f}-{band_hi:.0f} BPM fraction of total spectrum power:")
    print(f"  normal phase: {norm_frac*100:.2f}%")
    print(f"  hold phase:   {hold_frac*100:.2f}%")
    if norm_frac > 0:
        ratio = hold_frac / norm_frac
        print(f"  hold/normal ratio: {ratio:.2f}x")
    else:
        ratio = None
        print(f"  hold/normal ratio: n/a (no power in band during normal)")

    hr_med = median_hr_from_polar(run_dir)

    # ── Overlay plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    f_norm = r_normal["freq_axis"] * 60.0
    f_hold = r_hold["freq_axis"]   * 60.0
    norm_pwr = r_normal["power"] / r_normal["power"].max()
    hold_pwr = r_hold["power"]   / r_hold["power"].max()
    ax.plot(f_norm, norm_pwr, label=f"Normal breathing ({len(normal_slice)/fps:.0f}s)",
            color="#2d7", linewidth=1.4)
    ax.plot(f_hold, hold_pwr, label=f"Breath hold ({len(hold_slice)/fps:.0f}s)",
            color="#a44", linewidth=1.4)
    ax.axvspan(band_lo, band_hi, alpha=0.12, color="#888",
               label=f"{band_lo:.0f}-{band_hi:.0f} BPM (investigation target)")
    if hr_med is not None:
        ax.axvline(hr_med, color="#444", linestyle=":", linewidth=0.8,
                   label=f"{hr_med:.0f} BPM (HR median)")
        ax.axvline(hr_med / 2, color="#444", linestyle=":", linewidth=0.8,
                   alpha=0.4,
                   label=f"HR/2 = {hr_med/2:.0f} BPM (cardiac subharmonic)")
    ax.set_xlim(0, 120)
    ax.set_xlabel("BPM (frequency * 60)")
    ax.set_ylabel("Normalised power (each spectrum self-scaled)")
    ax.set_title(f"{run_dir.name}: chest centroid spectrum, normal vs hold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    out = run_dir / "diag_hold_vs_normal.png"
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"\nspectrum overlay -> {out}")

    # ── One-line conclusion ─────────────────────────────────────────────
    print()
    if norm_frac == 0:
        print(f"CONCLUSION: no power in {band_lo:.0f}-{band_hi:.0f} BPM "
              f"band during normal phase -- artifact not reproducing in "
              f"this run.")
        return
    if ratio >= 0.7:
        cause = "cardiac BCG / HR-related"
        if hr_med is not None and band_lo <= hr_med / 2 <= band_hi:
            cause += f" (HR/2 = {hr_med/2:.0f} BPM falls in target band)"
        print(f"CONCLUSION: {band_lo:.0f}-{band_hi:.0f} BPM peak PERSISTS "
              f"during breath hold (hold/normal ratio {ratio:.2f}x) -- "
              f"NOT respiratory; consistent with {cause}.")
    elif ratio < 0.3:
        print(f"CONCLUSION: {band_lo:.0f}-{band_hi:.0f} BPM peak "
              f"DISAPPEARS during breath hold (hold/normal ratio "
              f"{ratio:.2f}x) -- respiratory harmonic; consider "
              f"harmonic-rejection or sharper analysis band.")
    else:
        print(f"CONCLUSION: {band_lo:.0f}-{band_hi:.0f} BPM peak PARTIALLY "
              f"persists during hold (hold/normal ratio {ratio:.2f}x) -- "
              f"ambiguous; likely mix of cardiac and respiratory-harmonic.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir",
                    help="Path to run directory containing centroid_z.npy, "
                         "meta.json, and gt.csv with manual hold markers.")
    ap.add_argument("--hold-start", type=float, default=None,
                    help="Hold-phase start in seconds (overrides gt.csv "
                         "auto-detect).")
    ap.add_argument("--hold-end", type=float, default=None,
                    help="Hold-phase end in seconds (overrides gt.csv "
                         "auto-detect).")
    ap.add_argument("--band-lo", type=float, default=30.0,
                    help="Target band low BPM for the cardiac/harmonic "
                         "investigation (default 30).")
    ap.add_argument("--band-hi", type=float, default=33.0,
                    help="Target band high BPM (default 33).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        # Allow short form "runs/<TS>"; also resolve relative to phase1/
        for cand in (Path.cwd() / args.run_dir,
                     _HERE / args.run_dir):
            if cand.exists():
                run_dir = cand
                break
    if not (run_dir / "centroid_z.npy").exists():
        sys.exit(f"ERROR: no centroid_z.npy in {run_dir}")

    # Phase markers: explicit overrides win, else auto-detect, else fail.
    if args.hold_start is not None and args.hold_end is not None:
        hs, he = args.hold_start, args.hold_end
    else:
        hs_auto, he_auto = auto_detect_hold(run_dir)
        hs = args.hold_start if args.hold_start is not None else hs_auto
        he = args.hold_end   if args.hold_end   is not None else he_auto
        if hs is None or he is None:
            sys.exit("ERROR: hold phase not detected from gt.csv manual "
                     "markers and not supplied via --hold-start/--hold-end. "
                     "Per protocol, hold start = first manual rr=0 entry, "
                     "hold end = next manual rr>0 entry.")
    run(run_dir, hs, he, args.band_lo, args.band_hi)


if __name__ == "__main__":
    main()
