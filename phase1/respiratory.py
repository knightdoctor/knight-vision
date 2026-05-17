"""
phase1/respiratory.py
=====================
Respiratory rate estimation from a time-series of subject point clouds.

Algorithm
---------
1.  For each frame, compute the **centroid z** of the subject cluster.
    Vertical displacement of the thorax/sternum is a robust proxy for
    respiratory chest-rise in upright and semi-recumbent postures.

2.  Detrend the signal (remove DC + linear drift).

3.  Apply a Hanning window to the most recent ``fft_window_length``
    samples, then zero-pad to ``fft_window_length × fft_zero_pad_factor``
    for sub-bin frequency interpolation.

4.  Compute the real FFT power spectrum and find the dominant peak
    within the physiological RR band (``rr_freq_min``–``rr_freq_max`` Hz,
    default 0.1–2.0 Hz → 6–120 BPM).

5.  Return RR in BPM, SNR (peak power / mean band power), and a simple
    three-tier confidence label.

Usage
-----
    from phase1.respiratory import extract_rr

    result = extract_rr(cluster_frames, fps=10.0, config=cfg)
    print(f"RR = {result['rr_bpm']:.1f} BPM  SNR={result['snr']:.1f}  "
          f"Conf={result['confidence']}")
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from scipy.signal import detrend


# Minimum number of non-NaN samples required for a meaningful estimate
_MIN_GOOD_SAMPLES = 20


def extract_rr(
    cluster_frames: List[np.ndarray],
    fps: float,
    config,
) -> Dict[str, Union[float, str, np.ndarray]]:
    """Estimate respiratory rate from a sequence of subject cluster frames.

    Parameters
    ----------
    cluster_frames : list of np.ndarray
        Time-ordered sequence of (N_i, 3) subject cluster point clouds.
        Each array is one frame; N_i may vary between frames.
    fps : float
        Frame rate at which the clusters were captured (Hz).
    config : KVConfig
        Pipeline configuration.  Reads ``fft_window_length``,
        ``fft_zero_pad_factor``, ``rr_freq_min``, ``rr_freq_max``,
        ``snr_high_threshold``, ``snr_medium_threshold``.

    Returns
    -------
    dict with keys:
        ``rr_bpm`` (float)         — estimated respiratory rate (breaths/min).
        ``snr`` (float)            — peak power / mean in-band power.
        ``confidence`` (str)       — 'HIGH', 'MEDIUM', or 'LOW'.
        ``signal`` (np.ndarray)    — centroid-z time series used.
        ``freq_axis`` (np.ndarray) — FFT frequency axis (Hz).
        ``power`` (np.ndarray)     — FFT power spectrum (full band).
    """
    if len(cluster_frames) < _MIN_GOOD_SAMPLES:
        return _null_result()

    centroids_z = np.array(
        [frame[:, 2].mean() for frame in cluster_frames], dtype=float
    )
    return extract_rr_from_signal(centroids_z, fps, config)


def extract_rr_from_signal(
    centroid_z: np.ndarray,
    fps: float,
    config,
) -> Dict[str, Union[float, str, np.ndarray]]:
    """Estimate respiratory rate from a uniform centroid-z time series.

    Accepts NaN values (frames where the subject was not detected);
    NaNs are filled by linear interpolation before the FFT so the
    sample rate stays consistent with *fps*.

    Parameters
    ----------
    centroid_z : np.ndarray
        1-D array of per-frame centroid-z values (metres).  Must be sampled
        at a uniform rate of *fps* Hz.  NaN where subject was undetected.
    fps : float
        Sample rate of the signal (Hz) — must match the capture frame rate.
    config : KVConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Same keys as :func:`extract_rr`.
    """
    if centroid_z.size == 0:
        return _null_result()

    # ── 1. NaN interpolation (keep uniform sample rate) ───────────────────
    signal = centroid_z.copy().astype(float)
    nan_mask = np.isnan(signal)
    n_good = int((~nan_mask).sum())
    if n_good < _MIN_GOOD_SAMPLES:
        return _null_result()

    # Linear interpolation over NaN gaps; clamp edges
    indices = np.arange(len(signal))
    good_idx = indices[~nan_mask]
    signal[nan_mask] = np.interp(indices[nan_mask], good_idx, signal[~nan_mask])

    # ── 2. Detrend ────────────────────────────────────────────────────────
    signal = detrend(signal, type="linear")

    # ── 3. Window — use last N samples (or all if fewer) ─────────────────
    n_win = min(len(signal), config.fft_window_length)
    sig_win = signal[-n_win:]
    sig_windowed = sig_win * np.hanning(n_win)

    # ── 4. Zero-padded FFT ────────────────────────────────────────────────
    n_fft = config.fft_window_length * config.fft_zero_pad_factor
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))   # next power-of-2

    fft_complex = np.fft.rfft(sig_windowed, n=n_fft)
    power = np.abs(fft_complex) ** 2
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    # ── 5. Power-weighted centroid in physiological RR band ──────────────
    # 2026-05-16: switched from single-peak-pick to power-weighted centroid.
    # Peak-pick was fragile against narrow LF artifacts (e.g. a 6 BPM
    # postural/cluster-jitter peak winning over the broader respiratory
    # band's distributed power). Centroid integrates the WHOLE in-band
    # power distribution, so a broad respiratory peak (natural breath-rate
    # variability) wins over a narrow artifact peak of the same height.
    band_mask = (freq_axis >= config.rr_freq_min) & (freq_axis <= config.rr_freq_max)
    if not np.any(band_mask):
        return _null_result()

    power_band = power[band_mask]
    freq_band  = freq_axis[band_mask]
    total_p    = float(power_band.sum())
    if total_p <= 0:
        return _null_result()

    rr_hz   = float((freq_band * power_band).sum() / total_p)
    rr_bpm  = rr_hz * 60.0

    # ── 6. SNR — peak power in band vs mean in band ──────────────────────
    # We still report peak-based SNR (the strongest spectral feature) as
    # the signal-quality indicator. The frequency estimate is the centroid
    # but the strength of the dominant spectral feature is more interpretable.
    peak_idx   = int(np.argmax(power_band))
    peak_power = float(power_band[peak_idx])
    mean_power = float(power_band.mean())
    snr = peak_power / (mean_power + 1e-12)

    # ── 7. Confidence ────────────────────────────────────────────────────
    if snr >= config.snr_high_threshold:
        confidence = "HIGH"
    elif snr >= config.snr_medium_threshold:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "rr_bpm":     rr_bpm,
        "snr":        snr,
        "confidence": confidence,
        "signal":     signal,
        "freq_axis":  freq_axis,
        "power":      power,
    }


def classify_confidence(snr: float, config) -> str:
    """Apply the three-tier confidence label based on config thresholds."""
    if snr >= config.snr_high_threshold:
        return "HIGH"
    if snr >= config.snr_medium_threshold:
        return "MEDIUM"
    return "LOW"


def settled_median(
    rr_history: list,
    config,
    min_window: int = None,
    std_threshold_bpm: float = None,
    min_snr: float = None,
) -> Dict:
    """Compute the final RR estimate from a per-window history.

    PR M algorithm (2026-05-17): SNR-gated longest stable subsequence.

    1. **SNR gate**: drop windows whose snr < min_snr. Low-SNR windows are
       susceptible to centroid-degenerate-to-band-midpoint artifacts (e.g.
       a 20 BPM "estimate" when the in-band spectrum is essentially flat).
    2. **Longest stable subsequence**: among the remaining high-SNR
       windows, find the longest contiguous run (in original time order)
       whose BPMs have std < std_threshold_bpm. Median of those is the
       final estimate.

    Replaces PR A's "trailing-only" rule (deferred PR G) — trailing was
    fragile to late noise, and SNR-blindness let artifact runs win over
    short real-signal runs.

    Returns a dict with keys: ``rr_bpm``, ``snr``, ``confidence``,
    ``settled``, ``settle_note``, ``window_n``, ``windows_total``.
    """
    # Defaults from config (kwargs override for unit-test convenience)
    if min_window is None:
        min_window = getattr(config, "settled_median_min_window", 4)
    if std_threshold_bpm is None:
        std_threshold_bpm = getattr(
            config, "settled_median_std_threshold_bpm", 2.0)
    if min_snr is None:
        min_snr = getattr(config, "settled_median_min_snr", 3.0)

    n_total = len(rr_history)
    if n_total == 0:
        return {
            "rr_bpm": 0.0, "snr": 0.0, "confidence": "LOW",
            "settled": False, "settle_note": "no windows produced",
            "window_n": 0, "windows_total": 0,
        }

    bpms_all = np.array([h["rr_bpm"] for h in rr_history], dtype=float)
    snrs_all = np.array([h["snr"]    for h in rr_history], dtype=float)

    # ── SNR gate ─────────────────────────────────────────────────────────
    hi_mask = snrs_all >= min_snr
    n_hi    = int(hi_mask.sum())

    if n_hi < min_window:
        # Not enough trustworthy windows. Fall back to the last min_window
        # entries regardless of SNR — flagged as not-settled so the caller
        # knows the result is unreliable.
        sel_bpms = bpms_all[-min_window:] if n_total >= min_window else bpms_all
        sel_snrs = snrs_all[-min_window:] if n_total >= min_window else snrs_all
        snr = float(np.median(sel_snrs)) if len(sel_snrs) else 0.0
        return {
            "rr_bpm": float(np.median(sel_bpms)) if len(sel_bpms) else 0.0,
            "snr": snr,
            "confidence": classify_confidence(snr, config),
            "settled": False,
            "settle_note": (f"only {n_hi} of {n_total} windows above SNR "
                            f"{min_snr}; need ≥{min_window} — using last "
                            f"{min(min_window, n_total)} regardless"),
            "window_n": min(min_window, n_total),
            "windows_total": n_total,
        }

    # Indices (in original time order) of windows that passed the SNR gate
    hi_idx = np.where(hi_mask)[0]
    bpms_hi = bpms_all[hi_idx]

    # ── Longest stable subsequence within high-SNR windows ───────────────
    # We sweep over contiguous (by hi_idx position) subsets of size
    # >= min_window and pick the longest whose BPM std < threshold.
    best_start = best_end = -1
    best_len = 0
    for start in range(len(bpms_hi) - min_window + 1):
        # Extend rightwards as long as std stays low
        end = start + min_window
        while (end <= len(bpms_hi)
               and float(np.std(bpms_hi[start:end])) < std_threshold_bpm):
            end += 1
        run_len = end - 1 - start
        if run_len >= min_window and run_len > best_len:
            best_len = run_len
            best_start = start
            best_end = end - 1

    if best_len == 0:
        # High-SNR windows exist but never form a stable run — report
        # median of all high-SNR windows, flagged as not-settled.
        snr = float(np.median(snrs_all[hi_idx]))
        return {
            "rr_bpm": float(np.median(bpms_hi)),
            "snr": snr,
            "confidence": classify_confidence(snr, config),
            "settled": False,
            "settle_note": (f"{n_hi} high-SNR windows but no stable run of "
                            f"≥{min_window} (std<{std_threshold_bpm} BPM) "
                            f"— median of all high-SNR"),
            "window_n": n_hi,
            "windows_total": n_total,
        }

    sel_bpms = bpms_hi[best_start:best_end]
    sel_snrs = snrs_all[hi_idx[best_start:best_end]]
    snr = float(np.median(sel_snrs))
    # Time-range note uses the ORIGINAL indices for readability
    orig_lo = int(hi_idx[best_start])
    orig_hi = int(hi_idx[best_end - 1])
    return {
        "rr_bpm": float(np.median(sel_bpms)),
        "snr": snr,
        "confidence": classify_confidence(snr, config),
        "settled": True,
        "settle_note": (f"settled over {best_len} high-SNR windows "
                        f"(indices {orig_lo}-{orig_hi} of {n_total}, "
                        f"SNR≥{min_snr}, std<{std_threshold_bpm} BPM)"),
        "window_n": best_len,
        "windows_total": n_total,
    }


def _null_result() -> Dict[str, Union[float, str, np.ndarray]]:
    """Return a zeroed result dict (used when insufficient data)."""
    return {
        "rr_bpm":     0.0,
        "snr":        0.0,
        "confidence": "LOW",
        "signal":     np.array([]),
        "freq_axis":  np.array([]),
        "power":      np.array([]),
    }
