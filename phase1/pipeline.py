"""
phase1/pipeline.py
==================
Top-level orchestration for the Knight Vision Phase 1 pipeline.

``Phase1Pipeline`` wires together the sensor drivers, background model,
clustering, and respiratory analyser into a single callable interface.

Architecture (hot path)
-----------------------
    capture_frame()
        │
        ▼
    background.subtract()   ← geometry-first; no ML in hot path
        │
        ▼
    cluster_residuals()      ← DBSCAN on residuals only
        │
        ▼
    select_subject_cluster() ← largest cluster inside monitoring volume
        │
        ▼
    frame_buffer.append()
        │ (every compute_rr_every frames)
        ▼
    extract_rr()             ← FFT on centroid-z time series

Usage
-----
    from phase1.config import KVConfig
    from phase1.pipeline import Phase1Pipeline

    cfg = KVConfig()
    pipe = Phase1Pipeline(cfg, use_lidar=True)

    # Step 1 — empty room
    pipe.run_background_capture(seconds=60)

    # Step 2 — subject in frame
    result = pipe.run_live(duration_seconds=120)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np

from phase1.background import BackgroundModel
from phase1.clustering import (
    cluster_residuals,
    select_subject_cluster,
    select_chest_subset,
)
from phase1.config import KVConfig
from phase1.drivers.camera_driver import CameraDriver
from phase1.drivers.lidar_driver import LidarDriver
from phase1.drivers.radar_driver import RadarDriver
from phase1.respiratory import extract_rr, extract_rr_from_signal


class Phase1Pipeline:
    """End-to-end respiratory monitoring pipeline.

    Parameters
    ----------
    config : KVConfig
        Shared pipeline configuration.
    use_lidar : bool
        Enable LiDAR sensor (default True).
    use_radar : bool
        Enable mmWave radar sensor (default False).
    use_camera : bool
        Enable RGB camera (default False).

    Notes
    -----
    When both LiDAR and radar are enabled, the LiDAR is used as the
    primary sensor for background subtraction and clustering.  Radar
    fusion is reserved for Phase 2.

    When no sensor flag is True, the pipeline raises ValueError.
    """

    _COMPUTE_RR_EVERY_DEFAULT = 50   # frames between RR estimates

    def __init__(
        self,
        config: KVConfig,
        use_lidar: bool = True,
        use_radar: bool = False,
        use_camera: bool = False,
    ) -> None:
        if not (use_lidar or use_radar):
            raise ValueError("At least one of use_lidar or use_radar must be True.")

        self.config = config
        self.lidar  = LidarDriver(config)  if use_lidar  else None
        self.radar  = RadarDriver(config)  if use_radar  else None
        self.camera = CameraDriver(config) if use_camera else None

        # LiDAR takes priority as primary depth sensor
        self.primary_driver = self.lidar if self.lidar else self.radar

        self.background = BackgroundModel(config)

        # Runtime state
        self._frame_buffer:  List[np.ndarray] = []   # subject cluster frames (kept for run_from_recording)
        self._centroid_z:    List[float]       = []   # uniform centroid-z series; NaN when subject absent
        self._rr_history:    List[Dict]        = []   # all RR results

    # ── Background capture ────────────────────────────────────────────────

    def run_background_capture(self, seconds: float = 60) -> None:
        """Capture the empty-room background and persist it.

        The primary driver is placed into background mode (no subject),
        ``seconds`` of frames are captured, and the resulting model is
        saved to ``config.background_save_path``.

        Parameters
        ----------
        seconds : float
            Duration of background capture in seconds.
        """
        print(f"\n── Background Capture ({seconds:.0f}s) ──")
        self.primary_driver.set_live_mode(False)
        self.background.capture(self.primary_driver, seconds=seconds)
        self.config.background_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.background.save(self.config.background_save_path)
        print("── Background capture complete ──\n")

    # ── Live monitoring ───────────────────────────────────────────────────

    def run_live(
        self,
        duration_seconds: float = 60,
        compute_rr_every: int | None = None,
        verbose: bool = True,
        frame_callback=None,
        should_stop=None,
        force_fps: Optional[float] = None,
    ) -> Optional[Dict]:
        """Run the live monitoring loop.

        Captures frames from the primary driver, subtracts background,
        clusters residuals, selects the subject, and computes RR at a
        configurable cadence.

        Parameters
        ----------
        duration_seconds : float
            Total duration of the live run.
        compute_rr_every : int, optional
            Compute a new RR estimate every N frames.  Defaults to
            ``_COMPUTE_RR_EVERY_DEFAULT`` (50 frames).
        verbose : bool
            Print live status line each time RR is computed.

        Returns
        -------
        dict or None
            The most recent RR result dict from :func:`extract_rr`,
            or ``None`` if no estimate was produced.
        """
        if not self.background.is_built:
            # Try loading from disk
            try:
                self.background.load(self.config.background_save_path)
            except FileNotFoundError:
                print("ERROR: No background model found.  Run --mode background first.",
                      file=sys.stderr)
                return None

        # Configured fps is used to size the loop budget; the FFT uses the
        # MEASURED loop rate (computed below) because if config and reality
        # disagree, FFT bin→Hz mapping is wrong and reported BPM is biased
        # by (config_fps / actual_fps). Bug class found 2026-05-16.
        cfg_fps  = self.config.lidar_fps if self.lidar else self.config.radar_fps
        n_frames = max(1, int(duration_seconds * cfg_fps))

        # PR J 2026-05-17: derive compute cadence from sliding window config
        # if caller didn't override. step = window * (1 - overlap) seconds.
        if compute_rr_every is None:
            step_s = (self.config.rr_window_seconds
                      * (1.0 - self.config.rr_window_overlap))
            compute_rr_every = max(1, int(round(step_s * cfg_fps)))

        self.primary_driver.set_live_mode(True)

        self._frame_buffer = []
        self._centroid_z   = []
        self._rr_history   = []
        self._measured_fps: Optional[float] = None
        last_result: Optional[Dict] = None

        # Task #58 / PR Y — sticky subject tracking with stable-acquisition
        # gate. State machine:
        #
        #   prev_subject_centroid is None   → unlocked. Largest in-volume
        #       cluster (after shape gate) is treated as a CANDIDATE.
        #       We require ``subject_acquire_consistency_frames`` in a
        #       row where the candidate stays within
        #       ``subject_acquire_consistency_radius_m`` before promoting
        #       it to a real lock. Defeats single-frame noise blobs.
        #
        #   prev_subject_centroid is not None → locked. select_subject_
        #       cluster picks the candidate nearest prev within
        #       ``subject_lock_radius_m``, with a size-dominance override
        #       for the case where a much-larger cluster has appeared
        #       (the wall-phantom case). Lock is cleared after
        #       ``subject_lock_timeout_frames`` consecutive blank frames.
        prev_subject_centroid: Optional[np.ndarray] = None
        frames_without_subject = 0
        candidate_centroid: Optional[np.ndarray] = None
        candidate_streak = 0

        print(f"\n── Live Monitoring ({duration_seconds:.0f}s, "
              f"{n_frames} frames @ {cfg_fps:.0f} fps configured) ──")

        t_first_frame: Optional[float] = None

        n_clusters  = 0
        subject_pts = 0

        for i in range(n_frames):
            frame = self.primary_driver.capture_frame()
            if t_first_frame is None:
                t_first_frame = time.time()

            # ── Background subtraction ────────────────────────────────────
            residuals = self.background.subtract(frame)

            subject = None
            chest = None
            n_clusters = 0
            subject_pts = 0

            if residuals.shape[0] >= self.config.cluster_min_points:
                # ── Clustering ────────────────────────────────────────────
                clusters = cluster_residuals(residuals, self.config)
                n_clusters = len(clusters)

                # ── Subject selection ─────────────────────────────────────
                pick = select_subject_cluster(
                    clusters,
                    self.config.monitoring_volume,
                    prev_centroid=prev_subject_centroid,
                    lock_radius_m=self.config.subject_lock_radius_m,
                    shape_y_min_m=self.config.subject_shape_y_min_m,
                    shape_y_max_m=self.config.subject_shape_y_max_m,
                    shape_xz_max_m=self.config.subject_shape_xz_max_m,
                )

                if prev_subject_centroid is not None:
                    # Locked: trust whatever select returned.
                    subject = pick
                elif pick is not None:
                    # Unlocked: run stable-acquisition gate.
                    c_centroid = pick.mean(axis=0)
                    if (candidate_centroid is None or
                            np.linalg.norm(c_centroid - candidate_centroid)
                            > self.config.subject_acquire_consistency_radius_m):
                        candidate_centroid = c_centroid
                        candidate_streak = 1
                    else:
                        candidate_streak += 1
                    if (candidate_streak >=
                            self.config.subject_acquire_consistency_frames):
                        # Promote candidate to lock.
                        subject = pick
                        prev_subject_centroid = c_centroid
                        candidate_centroid = None
                        candidate_streak = 0
                    else:
                        subject = None   # candidate not trusted yet

            # Append to uniform centroid-z series (NaN when no subject).
            # PREFER the chest sub-region centroid — the whole-subject
            # centroid is dominated by sway (~20-25 BPM) and cardiac
            # ballistics (60+ BPM), per harmonic diagnostic 2026-05-16.
            # Falls back to whole subject when chest band can't be isolated
            # (cluster too short, too few chest points).
            if subject is not None:
                chest = select_chest_subset(
                    subject,
                    y_band_frac=(self.config.chest_y_band_min,
                                 self.config.chest_y_band_max),
                    xz_radius_m=self.config.chest_xz_radius_m,
                )
                analysis_pts = chest if chest is not None else subject
                cz = float(analysis_pts[:, 2].mean())
                subject_pts = subject.shape[0]   # report full subject count
                self._frame_buffer.append(subject)
                # Update the lock with whatever cluster we just settled on.
                prev_subject_centroid = subject.mean(axis=0)
                frames_without_subject = 0
            else:
                cz = float("nan")
                frames_without_subject += 1
                if frames_without_subject >= self.config.subject_lock_timeout_frames:
                    prev_subject_centroid = None
                    candidate_centroid = None
                    candidate_streak = 0
            self._centroid_z.append(cz)

            # Per-frame hook for orchestrator (logging / recording / viewer).
            # `chest` is the chest-band subset used as the FFT input — viewer
            # highlights it so the operator can verify the analysis window
            # is on torso, not head/legs.
            if frame_callback is not None:
                frame_callback(i, frame, residuals, subject, n_clusters, cz,
                               chest=chest)

            # External stop request (viewer Stop Session button, 's' key).
            if should_stop is not None and should_stop():
                if verbose:
                    print(f"[pipeline] stop requested at frame {i+1}/{n_frames}")
                break

            # ── Periodic RR estimation (sliding window, PR J) ────────────
            if (i + 1) % compute_rr_every == 0:
                # Measure loop fps regardless — it's recorded to meta.json.
                if t_first_frame is not None:
                    elapsed = time.time() - t_first_frame
                    actual_fps = (i + 1) / max(elapsed, 1e-3)
                else:
                    actual_fps = cfg_fps
                self._measured_fps = actual_fps
                # FFT fps: use measured for real sensors, but force_fps
                # for synthetic/demo runs where the synthetic driver bakes
                # the signal frequency against a known nominal rate.
                fps_for_fft = force_fps if force_fps is not None else actual_fps

                # Sliding window: FFT only the last rr_window_seconds of cz.
                # Earlier we FFTed the cumulative series, which smoothed real
                # transitions (apnoea onset, rate changes) by ~20s. With a
                # bounded window, the FFT reflects only the recent state.
                # Minimum 64 samples to keep FFT meaningful before window fills.
                fps_for_window = actual_fps if actual_fps > 0 else cfg_fps
                window_n = max(64, int(self.config.rr_window_seconds
                                       * fps_for_window))
                cz_arr  = np.array(self._centroid_z, dtype=float)
                sig     = cz_arr[-window_n:] if cz_arr.size > window_n else cz_arr

                result = extract_rr_from_signal(sig, fps_for_fft, self.config)
                result["fps_used"]        = fps_for_fft
                result["window_samples"]  = int(len(sig))
                result["window_seconds"]  = (len(sig) / fps_for_window
                                             if fps_for_window > 0 else 0.0)
                # t_centre: wall-clock time of the centre of this analysis
                # window. Apnoea / phase alignment uses this to map BPM
                # estimates onto manual phase markers in gt.csv.
                result["t_centre"] = (time.time()
                                      - result["window_seconds"] / 2.0)
                self._rr_history.append(result)
                last_result = result

                if verbose:
                    ts   = datetime.now().strftime("%H:%M:%S")
                    conf = result["confidence"]
                    rr   = result["rr_bpm"]
                    snr  = result["snr"]
                    print(
                        f"[{ts}] RR: {rr:5.1f} BPM | "
                        f"SNR: {snr:5.1f} | "
                        f"Conf: {conf:<6s} | "
                        f"Clusters: {n_clusters} | "
                        f"Subject pts: {subject_pts}"
                    )

        print("── Live run complete ──\n")
        return last_result

    # ── Replay from recording ─────────────────────────────────────────────

    def run_from_recording(
        self,
        frames: List[np.ndarray],
        fps: float | None = None,
        compute_rr_every: int | None = None,
        verbose: bool = True,
    ) -> Optional[Dict]:
        """Run the pipeline on pre-captured frames (offline replay).

        Useful for Phase 2 POC development and algorithm validation
        without needing live hardware.

        Parameters
        ----------
        frames : list of np.ndarray
            Pre-captured point-cloud frames in chronological order.
        fps : float, optional
            Frame rate of the recording.  Defaults to ``config.lidar_fps``.
        compute_rr_every : int, optional
            RR estimation cadence in frames.

        Returns
        -------
        dict or None
            Final RR result, or ``None`` if estimation failed.
        """
        if not self.background.is_built:
            try:
                self.background.load(self.config.background_save_path)
            except FileNotFoundError:
                print("ERROR: No background model.  Call run_background_capture() first.",
                      file=sys.stderr)
                return None

        if fps is None:
            fps = self.config.lidar_fps
        if compute_rr_every is None:
            compute_rr_every = self._COMPUTE_RR_EVERY_DEFAULT

        self._frame_buffer = []
        self._rr_history   = []
        last_result: Optional[Dict] = None

        # Mirror run_live's sticky-tracking state (task #58 + PR Y).
        prev_subject_centroid: Optional[np.ndarray] = None
        frames_without_subject = 0
        candidate_centroid: Optional[np.ndarray] = None
        candidate_streak = 0

        print(f"\n── Replay ({len(frames)} frames @ {fps:.0f} fps) ──")

        for i, frame in enumerate(frames):
            residuals = self.background.subtract(frame)

            n_clusters = 0
            subject_pts = 0
            subject = None

            if residuals.shape[0] >= self.config.cluster_min_points:
                clusters = cluster_residuals(residuals, self.config)
                n_clusters = len(clusters)
                pick = select_subject_cluster(
                    clusters,
                    self.config.monitoring_volume,
                    prev_centroid=prev_subject_centroid,
                    lock_radius_m=self.config.subject_lock_radius_m,
                    shape_y_min_m=self.config.subject_shape_y_min_m,
                    shape_y_max_m=self.config.subject_shape_y_max_m,
                    shape_xz_max_m=self.config.subject_shape_xz_max_m,
                )
                if prev_subject_centroid is not None:
                    subject = pick
                elif pick is not None:
                    c_centroid = pick.mean(axis=0)
                    if (candidate_centroid is None or
                            np.linalg.norm(c_centroid - candidate_centroid)
                            > self.config.subject_acquire_consistency_radius_m):
                        candidate_centroid = c_centroid
                        candidate_streak = 1
                    else:
                        candidate_streak += 1
                    if (candidate_streak >=
                            self.config.subject_acquire_consistency_frames):
                        subject = pick
                        prev_subject_centroid = c_centroid
                        candidate_centroid = None
                        candidate_streak = 0
                if subject is not None:
                    self._frame_buffer.append(subject)
                    subject_pts = subject.shape[0]

            if subject is not None:
                prev_subject_centroid = subject.mean(axis=0)
                frames_without_subject = 0
            else:
                frames_without_subject += 1
                if frames_without_subject >= self.config.subject_lock_timeout_frames:
                    prev_subject_centroid = None
                    candidate_centroid = None
                    candidate_streak = 0

            if (i + 1) % compute_rr_every == 0 and len(self._frame_buffer) >= 20:
                result = extract_rr(self._frame_buffer, fps, self.config)
                self._rr_history.append(result)
                last_result = result
                if verbose:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{ts}] RR: {result['rr_bpm']:5.1f} BPM | "
                        f"SNR: {result['snr']:5.1f} | "
                        f"Conf: {result['confidence']:<6s} | "
                        f"Clusters: {n_clusters} | "
                        f"Subject pts: {subject_pts}"
                    )

        print("── Replay complete ──\n")
        return last_result
