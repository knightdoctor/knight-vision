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

    # ── Subject tracking (task #58 + PR Y, 2026-05-18) ───────────────────────
    # Once a subject cluster is found, prefer the cluster nearest the
    # previous frame's centroid over the largest-in-volume rule. Without
    # this, two similarly-sized clusters (subject vs. sofa edge) cause
    # the ROI to flicker frame-to-frame and produce step-jumps in chest-Z
    # — the "664mm spike artifact" seen in M1 paired captures.
    subject_lock_radius_m: float = 0.30      # max per-frame centroid jump

    # PR Y tightening: original 10 frames @ ~6-14 fps = 0.7-1.7 s of grace.
    # Apnoea detection wants faster re-acquire after a real occlusion, and
    # the bigger risk is a momentary blip locking the tracker onto noise
    # for a full second. 2 frames @ ~10 fps = ~200 ms, still rides through
    # a single-frame DBSCAN noise event without dropping a real subject.
    subject_lock_timeout_frames: int = 2

    # Stable-acquisition gate (PR Y): require N consecutive frames where
    # the largest-in-volume cluster centroid stays within
    # ``subject_acquire_consistency_radius_m`` before formally declaring
    # a lock. Defeats single-frame phantoms (residual flicker on a wall
    # corner that pops up for 1 frame then vanishes) from being adopted
    # as the subject. Trade: 3 frames @ ~10 fps = 300 ms of latency on
    # re-acquire, which is acceptable given the apnoea response window.
    subject_acquire_consistency_frames: int = 3
    subject_acquire_consistency_radius_m: float = 0.20

    # Shape gate (PR Y): drop clusters whose bounding-box dimensions are
    # incompatible with a human silhouette. Cheap belt-and-braces filter
    # for wall-slice phantoms that share DBSCAN cluster mass but no
    # human-like vertical extent. Bounds are deliberately generous:
    # infant in a cot vs. adult standing.
    #   Y span:  20 cm  (newborn lying)  →  1.80 m  (adult standing)
    #   X+Z span: 1.0 m  diagonal envelope (covers arms-out adult)
    # Set ``subject_shape_y_min_m = 0.0`` to disable.
    subject_shape_y_min_m: float = 0.20
    subject_shape_y_max_m: float = 1.80
    subject_shape_xz_max_m: float = 1.00

    # ── Monitoring volume (metres) ───────────────────────────────────────────
    # 2026-05-17: tightened Z from (0.0, 2.5) → (0.5, 2.0) after sofa at 2-3m
    # was contaminating subject selection — cluster picker bounced between
    # Phil at ~1m and sofa at 2-3m frame-to-frame. New range covers a seated
    # subject at 1-1.8m with slack; tune up if subject sits further back.
    monitoring_volume: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-1.5, -1.5, 0.5],
             [ 1.5,  1.5, 2.0]],
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

    # ── Chest sub-region (PR L 2026-05-17) ───────────────────────────────────
    # Y-band fraction of subject cluster Y span used as the "chest" region.
    # Tightened from (0.50, 0.85) → (0.55, 0.75) based on offline replay of
    # run 20260517_064421: narrowed band excludes head/neck (top 25%) and
    # lower abdomen (bottom 55%), closing LiDAR-vs-Polar Δ by ~1 BPM and
    # raising SNR. Wider re-introduces head/arm noise; tighter (e.g. 0.60-
    # 0.70 sternum-only) is too sparse and noisier in this fps regime.
    # PR X 2026-05-18: widened from (0.55, 0.75) to (0.30, 0.75) so the
    # analysis window covers chest + abdomen, not just upper chest. Phase
    # 1's clinical target (infants in cots) breathes predominantly with
    # the diaphragm — abdominal motion is the load-bearing signal there,
    # and a sternum-only band would miss it entirely. Adult validation
    # may give back ~0.5-1 BPM of accuracy vs the narrower band, but the
    # deployment use-case requires this trade.
    chest_y_band_min: float = 0.30
    chest_y_band_max: float = 0.75

    # ── Chest X/Z lateral crop (PR T 2026-05-18) ─────────────────────────────
    # After picking the Y band, also drop points more than this many metres
    # from the band's median X / Z. Prevents the chest band from spilling onto
    # furniture or wall residuals when DBSCAN chains the subject's cluster
    # through an arm-on-desk or chair-back to nearby static structure.
    # 0.30 m = ~60 cm box around the torso, ample for breathing-driven
    # X-spread but tight enough to exclude desks ~50 cm to either side.
    chest_xz_radius_m: float = 0.30

    # ── SNR / confidence thresholds ──────────────────────────────────────────
    snr_high_threshold: float = 5.0    # tightened 2026-05-16 per M1 requirement
    snr_medium_threshold: float = 3.0  # LOW reports value with caveat, never suppress

    # ── Settled-median window selection (PR M 2026-05-17) ────────────────────
    # settled_median picks per-window estimates that BOTH have SNR >= this
    # threshold AND form the longest stable subsequence anywhere in the
    # history (std < settled_median_std_threshold_bpm BPM). Defaults to
    # snr_medium_threshold so we only trust MEDIUM/HIGH windows.
    settled_median_min_snr: float = 3.0
    settled_median_std_threshold_bpm: float = 2.0
    settled_median_min_window: int = 4

    # ── File paths ───────────────────────────────────────────────────────────
    background_save_path: Path = Path("phase1/data/background_model.npz")
    output_dir: Path = Path("phase1/output")
