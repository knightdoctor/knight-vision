"""
phase1/config.py
================
Central configuration for the Knight Vision Phase 1 respiratory monitoring pipeline.

All magic numbers live here.  Change this file rather than hunting through the codebase.
Import with:
    from phase1.config import KVConfig
    cfg = KVConfig()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class KVConfig:
    """All tunable parameters for the Knight Vision Phase 1 pipeline.

    Attributes:
        lidar_fps:               LiDAR capture rate (Hz).
        radar_fps:               mmWave radar capture rate (Hz).
        camera_fps:              RGB camera capture rate (Hz).

        voxel_size:              Edge length of each voxel cube (metres).  2 cm default.

        background_sigma:        Residual threshold — points more than this many σ from
                                 the per-voxel background mean are flagged as residuals.
        background_update_rate:  Exponential moving-average rate for slow background
                                 drift correction (call update() only when subject absent).
        background_min_std:      Floor on per-voxel std to avoid zero-division artefacts.

        dbscan_eps:              DBSCAN neighbourhood radius (metres).
        dbscan_min_samples:      DBSCAN core-point minimum neighbour count.
        cluster_min_points:      Clusters with fewer points than this are discarded.

        monitoring_volume:       (2, 3) array [[xmin, ymin, zmin], [xmax, ymax, zmax]]
                                 (metres).  Subject cluster centroid must fall inside.

        rr_freq_min:             Lower bound of physiological RR band (Hz) — 6 BPM.
        rr_freq_max:             Upper bound of physiological RR band (Hz) — 120 BPM.
        fft_window_length:       Number of frames fed into the FFT window.
        fft_zero_pad_factor:     Zero-pad multiplier for sub-bin frequency interpolation.

        snr_high_threshold:      SNR ≥ this → 'HIGH' confidence.
        snr_medium_threshold:    SNR ≥ this → 'MEDIUM' confidence; below → 'LOW'.

        background_save_path:    Where BackgroundModel.save() writes the .npz file.
        output_dir:              Directory for spectrum plots and other artefacts.
    """

    # ── Sensor frame rates (Hz) ──────────────────────────────────────────────
    lidar_fps: float = 10.0  # actual end-to-end pipeline rate; sensor itself is 30 Hz, drained
    radar_fps: float = 10.0
    camera_fps: float = 30.0

    # ── Voxel grid ───────────────────────────────────────────────────────────
    voxel_size: float = 0.02          # 2 cm cubes

    # ── Background model ─────────────────────────────────────────────────────
    background_sigma: float = 2.0
    background_update_rate: float = 0.01
    background_min_std: float = 0.005  # 5 mm floor on per-voxel σ

    # ── DBSCAN clustering ────────────────────────────────────────────────────
    dbscan_eps: float = 0.20          # 20 cm neighbourhood radius (person-sized clusters)
    dbscan_min_samples: int = 5
    cluster_min_points: int = 10

    # ── Monitoring volume (metres) ───────────────────────────────────────────
    monitoring_volume: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-1.5, -1.5, 0.0],
             [ 1.5,  1.5, 2.5]],
            dtype=float,
        )
    )

    # ── Respiratory analysis ─────────────────────────────────────────────────
    rr_freq_min: float = 0.17           # Hz -> 10 BPM (excludes sub-physiological postural/sway artifacts)
    rr_freq_max: float = 0.50           # Hz -> 30 BPM (covers adult + light tachypnea; excludes cardiac >40)
    fft_window_length: int = 256       # samples fed into FFT
    fft_zero_pad_factor: int = 16      # zero-pad to fft_window_length × this

    # ── Sliding-window FFT (PR J 2026-05-17) ─────────────────────────────────
    # Each RR estimate uses only the last rr_window_seconds of cz, advancing
    # by rr_window_overlap fraction of the window. 20s @ 75% overlap → new
    # estimate every 5s; bin spacing ~3 BPM; apnoea detection lag ~10s.
    # See phase1/notes/next_session.md for the decision reasoning.
    rr_window_seconds: float = 20.0
    rr_window_overlap: float = 0.75

    # ── SNR / confidence thresholds ──────────────────────────────────────────
    snr_high_threshold: float = 5.0    # tightened 2026-05-16 per M1 requirement
    snr_medium_threshold: float = 3.0  # LOW reports value with caveat, never suppress

    # ── File paths ───────────────────────────────────────────────────────────
    background_save_path: Path = Path("phase1/data/background_model.npz")
    output_dir: Path = Path("phase1/output")
