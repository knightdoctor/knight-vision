"""
phase1/visualise.py
===================
Visualisation helpers for the Knight Vision Phase 1 pipeline.

All functions are optional — the pipeline runs without calling any of
them.  Matplotlib plots are saved to ``config.output_dir`` rather than
displayed interactively (safe for headless/server use).  Pass
``show=True`` to any function to also open an interactive window.

Open3D 3-D visualisations are available when open3d is installed.
If it is absent, those functions print a warning and return immediately.

Usage
-----
    from phase1.visualise import plot_rr_spectrum, plot_live_rr
    plot_rr_spectrum(result, config, save_path="phase1/output/spectrum.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# ── Matplotlib (required) ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless systems
import matplotlib.pyplot as plt

# ── Open3D (optional) ────────────────────────────────────────────────────
try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


# ── Open3D helpers ───────────────────────────────────────────────────────

def plot_background_model(bg_model, show: bool = False) -> None:
    """Visualise voxel occupancy as an Open3D point cloud.

    Each occupied voxel is represented by its mean XYZ, coloured by
    point count (blue → red = low → high).

    Parameters
    ----------
    bg_model : BackgroundModel
        A built BackgroundModel instance.
    show : bool
        If True, open an interactive Open3D viewer window.
    """
    if not _HAS_O3D:
        print("[visualise] open3d not available — skipping 3-D background plot.")
        return
    if not bg_model.is_built:
        print("[visualise] BackgroundModel not built yet.")
        return

    pts = bg_model._voxel_means
    counts = bg_model._voxel_counts.astype(float)
    counts_norm = (counts - counts.min()) / (counts.ptp() + 1e-9)

    # Colour: blue (low) → red (high)
    colours = np.column_stack([counts_norm, np.zeros_like(counts_norm),
                                1 - counts_norm])

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts)
    pcd.colors  = o3d.utility.Vector3dVector(colours)

    if show:
        o3d.visualization.draw_geometries([pcd], window_name="Background Model")


def plot_residuals(background, frame: np.ndarray, show: bool = False) -> None:
    """Show background voxels (grey) and residual points (red).

    Parameters
    ----------
    background : BackgroundModel
        A built BackgroundModel instance.
    frame : np.ndarray
        Shape (N, 3) live frame to compare against background.
    show : bool
        Open interactive Open3D viewer if True.
    """
    if not _HAS_O3D:
        print("[visualise] open3d not available — skipping residual plot.")
        return

    residuals = background.subtract(frame)

    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(background._voxel_means)
    bg_pcd.paint_uniform_color([0.6, 0.6, 0.6])   # grey

    res_pcd = o3d.geometry.PointCloud()
    res_pcd.points = o3d.utility.Vector3dVector(residuals)
    res_pcd.paint_uniform_color([0.9, 0.1, 0.1])   # red

    if show:
        o3d.visualization.draw_geometries(
            [bg_pcd, res_pcd], window_name="Background + Residuals"
        )


# ── Matplotlib helpers ────────────────────────────────────────────────────

def plot_rr_spectrum(
    result: Dict,
    config=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the time-domain centroid-z signal and the FFT power spectrum.

    The dominant RR peak is annotated with a vertical line and BPM label.

    Parameters
    ----------
    result : dict
        Output of :func:`phase1.respiratory.extract_rr`.
    config : KVConfig, optional
        Used to shade the physiological RR band on the spectrum plot.
    save_path : str or Path, optional
        Where to save the PNG.  If None, saves to the current directory.
    show : bool
        Display the figure interactively (blocks until window is closed).

    Returns
    -------
    Path or None
        Path to the saved PNG, or None if no signal data was available.
    """
    signal    = result.get("signal", np.array([]))
    freq_axis = result.get("freq_axis", np.array([]))
    power     = result.get("power", np.array([]))
    rr_bpm    = result.get("rr_bpm", 0.0)
    snr       = result.get("snr", 0.0)
    conf      = result.get("confidence", "—")

    if signal.size == 0:
        print("[visualise] No signal data to plot.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(
        f"Knight Vision — Respiratory Analysis\n"
        f"RR = {rr_bpm:.1f} BPM  |  SNR = {snr:.1f}  |  Confidence = {conf}",
        fontsize=13, fontweight="bold",
    )

    # ── Time-domain signal ────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(signal, color="#2563EB", linewidth=1.2, label="Centroid-Z displacement")
    ax0.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax0.set_xlabel("Frame index")
    ax0.set_ylabel("Displacement (m)")
    ax0.set_title("Centroid-Z Time Series (detrended)")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    # ── Power spectrum ────────────────────────────────────────────────────
    ax1 = axes[1]

    # Limit x-axis to 0–3 Hz for readability
    plot_mask = freq_axis <= 3.0
    ax1.semilogy(freq_axis[plot_mask], power[plot_mask] + 1e-20,
                 color="#64748B", linewidth=1.0, label="Power spectrum")

    # Shade physiological band
    if config is not None:
        ax1.axvspan(config.rr_freq_min, config.rr_freq_max,
                    alpha=0.08, color="green", label="RR band (0.1–2.0 Hz)")

    # Mark dominant peak
    rr_hz = rr_bpm / 60.0
    if rr_hz > 0:
        ax1.axvline(rr_hz, color="#DC2626", linewidth=1.5,
                    linestyle="--", label=f"RR peak ({rr_hz:.3f} Hz = {rr_bpm:.1f} BPM)")

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power (log scale)")
    ax1.set_title("FFT Power Spectrum")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = Path("phase1/output/rr_spectrum.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [visualise] Spectrum saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_live_rr(
    rr_history: List[Dict],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot a scrolling history of respiratory rate estimates.

    Parameters
    ----------
    rr_history : list of dict
        Sequence of result dicts from successive :func:`extract_rr` calls.
    save_path : str or Path, optional
        PNG save path.
    show : bool
        Display interactively if True.

    Returns
    -------
    Path or None
        Saved PNG path, or None if history is empty.
    """
    if not rr_history:
        print("[visualise] No RR history to plot.")
        return None

    rr_values = [r["rr_bpm"]    for r in rr_history]
    snr_values = [r["snr"]      for r in rr_history]
    confs      = [r["confidence"] for r in rr_history]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Knight Vision — Live Respiratory Rate History", fontsize=13,
                 fontweight="bold")

    x = list(range(1, len(rr_values) + 1))

    # Colour-code by confidence
    colour_map = {"HIGH": "#16A34A", "MEDIUM": "#CA8A04", "LOW": "#DC2626"}
    colours = [colour_map.get(c, "#64748B") for c in confs]

    ax0.scatter(x, rr_values, c=colours, s=60, zorder=3)
    ax0.plot(x, rr_values, color="#94A3B8", linewidth=0.8, zorder=2)
    ax0.axhline(12, color="grey", linewidth=0.6, linestyle=":", label="Normal RR bounds")
    ax0.axhline(20, color="grey", linewidth=0.6, linestyle=":")
    ax0.set_ylabel("RR (BPM)")
    ax0.set_title("Estimated Respiratory Rate over Time")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(bottom=0)

    ax1.bar(x, snr_values, color="#3B82F6", alpha=0.7)
    ax1.axhline(3.0, color="#DC2626", linewidth=1.0, linestyle="--",
                label="High confidence threshold (SNR=3)")
    ax1.set_xlabel("Estimate #")
    ax1.set_ylabel("SNR")
    ax1.set_title("Signal-to-Noise Ratio")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = Path("phase1/output/rr_history.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [visualise] RR history saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path
