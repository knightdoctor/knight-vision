"""Harmonic-locking diagnostic for Run #3 (20260516_081815).

Replays the saved centroid_z, computes the FFT spectrum at t = 10, 30, 60s
and end-of-run, and reports the top in-band peaks.

Goal: determine whether the early 20-26 BPM reads were
  (a) harmonic-locking — a 20 BPM peak present throughout the recording
      that the algorithm picked instead of the true ~10 BPM fundamental, or
  (b) physiological — a real 20 BPM signal that genuinely fades, replaced
      by the slower true breathing rate as Phil settled in.

If (a): fix is "prefer lowest in-band peak above noise threshold".
If (b): no algorithm change needed — early reads were real.

Usage: phase1/run.sh harmonic_diagnostic.py <run_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig


def find_top_peaks(freq_hz, power, band_lo_hz=0.10, band_hi_hz=2.00,
                   n_top=5, min_separation_bpm=2.0):
    """Find the N strongest peaks in band, separated by min_separation_bpm."""
    mask = (freq_hz >= band_lo_hz) & (freq_hz <= band_hi_hz)
    f_in   = freq_hz[mask]
    p_in   = power[mask]
    bpm_in = f_in * 60.0

    # Local maxima: power > both neighbours
    peaks = []
    for i in range(1, len(p_in) - 1):
        if p_in[i] > p_in[i-1] and p_in[i] > p_in[i+1]:
            peaks.append((float(bpm_in[i]), float(p_in[i])))
    peaks.sort(key=lambda x: -x[1])

    # Enforce min separation
    kept = []
    for bpm, pw in peaks:
        if all(abs(bpm - k_bpm) > min_separation_bpm for k_bpm, _ in kept):
            kept.append((bpm, pw))
        if len(kept) >= n_top:
            break
    return kept


def spectrum_at_time(cz, fps, t_seconds, cfg):
    """Compute FFT spectrum on cz[:t_seconds*fps] (cumulative signal)."""
    n = int(t_seconds * fps)
    if n > len(cz):
        n = len(cz)
    sig = cz[:n].copy()

    # Same pre-processing as extract_rr_from_signal
    valid = ~np.isnan(sig)
    if valid.sum() < 30:
        return None, None, n
    sig[~valid] = float(np.nanmean(sig))   # mean-fill NaNs
    sig = sig - np.mean(sig)               # detrend

    # FFT with same zero-padding as live pipeline
    n_fft = int(cfg.fft_window_length * cfg.fft_zero_pad_factor)
    n_fft = max(n_fft, n)
    sig_padded = np.zeros(n_fft)
    sig_padded[:n] = sig
    fft_vals = np.fft.rfft(sig_padded * np.hanning(n_fft))
    power = np.abs(fft_vals) ** 2
    freq_hz = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    return freq_hz, power, n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cz-file", default="centroid_z.npy",
                    help="cz array filename inside run_dir (default centroid_z.npy)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = _HERE / "runs" / run_dir

    meta = json.loads((run_dir / "meta.json").read_text())
    cz   = np.load(run_dir / args.cz_file)
    fps  = meta["frames_seen"] / meta["wall_seconds"]
    print(f"(using cz file: {args.cz_file})")

    print(f"\nRun: {run_dir.name}")
    print(f"fps (corrected): {fps:.3f}  |  total frames: {len(cz)}  "
          f"|  duration: {len(cz)/fps:.1f}s")
    print(f"Original reported: rr_bpm={meta['result_final']['rr_bpm']:.2f}  "
          f"conf={meta['result_final']['confidence']}")

    cfg = KVConfig()
    time_points = [10, 30, 60, len(cz) / fps]

    print("\n" + "=" * 78)
    print("Top in-band (6-120 BPM) peaks at each time point")
    print("=" * 78)
    print(f"{'t (s)':>6}  {'n frames':>9}  {'top peaks (BPM, rel.power)':<60}")
    print("-" * 78)

    spectra = []
    for t in time_points:
        freq, power, n = spectrum_at_time(cz, fps, t, cfg)
        if freq is None:
            print(f"{t:>6.1f}  {n:>9}  (insufficient data)")
            continue
        peaks = find_top_peaks(freq, power, 0.10, 2.00, n_top=5)
        if not peaks:
            print(f"{t:>6.1f}  {n:>9}  (no peaks)")
            continue
        max_p = peaks[0][1]
        peaks_str = "  ".join(f"{b:5.1f}@{p/max_p:.2f}" for b, p in peaks)
        print(f"{t:>6.1f}  {n:>9}  {peaks_str}")
        spectra.append((t, freq, power, peaks))

    # Conclusion
    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)

    # Look for 20 BPM peak (18-22 BPM) at each time point
    twenty_strength = []
    ten_strength    = []
    for t, freq, power, peaks in spectra:
        twenty_str = "absent"
        ten_str    = "absent"
        for bpm, pw in peaks:
            if 18 <= bpm <= 22 and twenty_str == "absent":
                twenty_str = f"present at {bpm:.1f} BPM rank {peaks.index((bpm,pw))+1}"
            if 8 <= bpm <= 12 and ten_str == "absent":
                ten_str = f"present at {bpm:.1f} BPM rank {peaks.index((bpm,pw))+1}"
        twenty_strength.append((t, twenty_str))
        ten_strength.append((t, ten_str))

    print("\n~20 BPM peak across recording:")
    for t, s in twenty_strength:
        print(f"  t={t:>6.1f}s : {s}")
    print("\n~10 BPM peak across recording:")
    for t, s in ten_strength:
        print(f"  t={t:>6.1f}s : {s}")

    # Stacked plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(spectra), 1, figsize=(10, 2.2 * len(spectra)),
                                 sharex=True)
        if len(spectra) == 1:
            axes = [axes]
        for ax, (t, freq, power, peaks) in zip(axes, spectra):
            bpm_axis = freq * 60.0
            in_band = (bpm_axis >= 6) & (bpm_axis <= 120)
            ax.plot(bpm_axis[in_band], power[in_band], color="#2d7", linewidth=0.8)
            for bpm, pw in peaks[:3]:
                ax.axvline(bpm, color="#a44", alpha=0.4, linewidth=0.5)
                ax.text(bpm, pw, f"{bpm:.1f}", fontsize=8, color="#a44",
                        ha="center", va="bottom")
            ax.set_title(f"t = {t:.0f}s  (n={int(t*fps)} samples)", fontsize=10)
            ax.set_xlim(6, 120)
            ax.set_ylabel("power")
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("BPM")
        plt.tight_layout()
        stem = Path(args.cz_file).stem
        out_png = run_dir / f"harmonic_diagnostic_{stem}.png"
        plt.savefig(out_png, dpi=110)
        print(f"\nSpectra saved → {out_png}")
    except Exception as e:
        print(f"\n(plot failed: {e})")


if __name__ == "__main__":
    main()
