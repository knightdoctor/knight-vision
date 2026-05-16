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
    min_window: int = 4,
    std_threshold_bpm: float = 2.0,
) -> Dict:
    """Compute the final RR estimate from a per-window history.

    Finds the longest trailing window of length ≥ ``min_window`` where the
    per-window BPM std is below ``std_threshold_bpm`` (considered settled),
    then returns the median BPM over that window. Pre-settling windows are
    dropped — they typically reflect transient sensor/subject startup.

    Returns a dict with keys: ``rr_bpm``, ``snr`` (median over settled
    window), ``confidence``, ``settled``, ``settle_note``, ``window_n``,
    ``windows_total``.
    """
    n_total = len(rr_history)
    if n_total == 0:
        return {
            "rr_bpm": 0.0, "snr": 0.0, "confidence": "LOW",
            "settled": False, "settle_note": "no windows produced",
            "window_n": 0, "windows_total": 0,
        }

    bpms = np.array([h["rr_bpm"] for h in rr_history], dtype=float)
    snrs = np.array([h["snr"]    for h in rr_history], dtype=float)

    if n_total < min_window:
        # Not enough windows to assess convergence — fall back to last.
        snr = float(snrs[-1])
        return {
            "rr_bpm": float(bpms[-1]),
            "snr": snr,
            "confidence": classify_confidence(snr, config),
            "settled": False,
            "settle_note": f"insufficient windows for settled median "
                           f"(have {n_total}, need ≥{min_window}) — using last",
            "window_n": 1,
            "windows_total": n_total,
        }

    # Expand the trailing window backwards as long as its std stays low.
    best_n = 0
    for n in range(min_window, n_total + 1):
        if float(np.std(bpms[-n:])) < std_threshold_bpm:
            best_n = n
        else:
            break

    if best_n == 0:
        # Never settled — use the last min_window as a least-bad fallback.
        sel_bpms = bpms[-min_window:]
        sel_snrs = snrs[-min_window:]
        snr = float(np.median(sel_snrs))
        return {
            "rr_bpm": float(np.median(sel_bpms)),
            "snr": snr,
            "confidence": classify_confidence(snr, config),
            "settled": False,
            "settle_note": f"did not settle "
                           f"(min std over {min_window}+ windows ≥ "
                           f"{std_threshold_bpm} BPM) — median of last {min_window}",
            "window_n": min_window,
            "windows_total": n_total,
        }

    sel_bpms = bpms[-best_n:]
    sel_snrs = snrs[-best_n:]
    snr = float(np.median(sel_snrs))
    pre_drop = n_total - best_n
    return {
        "rr_bpm": float(np.median(sel_bpms)),
        "snr": snr,
        "confidence": classify_confidence(snr, config),
        "settled": True,
        "settle_note": f"settled over last {best_n} of {n_total} windows "
                       f"(std<{std_threshold_bpm} BPM); dropped {pre_drop} pre-settling",
        "window_n": best_n,
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
