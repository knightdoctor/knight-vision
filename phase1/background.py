"""
phase1/background.py
====================
Voxel-grid background model for the Knight Vision Phase 1 pipeline.

The model divides 3-D space into equal-sized voxel cubes.  During a
pre-scan capture phase (room empty, subject absent), it accumulates
per-voxel statistics (mean XYZ, per-axis σ) from many frames.  At
runtime, any incoming point that lands in a voxel that is either
(a) absent from the background model, or (b) deviates by more than
``config.background_sigma`` σ from the learned mean, is returned as a
residual — candidate belonging to the subject.

Usage
-----
    from phase1.config import KVConfig
    from phase1.background import BackgroundModel
    from phase1.drivers import LidarDriver

    cfg = KVConfig()
    driver = LidarDriver(cfg)
    driver.set_live_mode(False)          # background-only mode

    bg = BackgroundModel(cfg)
    bg.capture(driver, seconds=60)       # build the model
    bg.save(cfg.background_save_path)    # persist

    # Later, in the live loop:
    bg2 = BackgroundModel(cfg)
    bg2.load(cfg.background_save_path)
    residuals = bg2.subtract(live_frame) # (M, 3) residual points
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Type alias for voxel key
_VKey = Tuple[int, int, int]


class BackgroundModel:
    """Per-voxel statistical background model.

    Parameters
    ----------
    config : KVConfig
        Shared pipeline configuration.  Reads ``voxel_size``,
        ``background_sigma``, ``background_update_rate``,
        ``background_min_std``.

    Attributes
    ----------
    _voxel_means : np.ndarray or None
        (M, 3) float64 — per-voxel mean XYZ.
    _voxel_stds : np.ndarray or None
        (M, 3) float64 — per-voxel σ (floored at ``background_min_std``).
    _voxel_counts : np.ndarray or None
        (M,) int64 — number of points that contributed to each voxel.
    _key_to_idx : dict
        Maps (i, j, k) voxel index → row index in the numpy arrays above.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._voxel_means:  np.ndarray | None = None
        self._voxel_stds:   np.ndarray | None = None
        self._voxel_counts: np.ndarray | None = None
        self._key_to_idx:   Dict[_VKey, int] = {}
        # Cached for fast subtract — built lazily, invalidated on load/_fit.
        self._bg_packed_sorted:  np.ndarray | None = None
        self._bg_row_for_sorted: np.ndarray | None = None

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        """Return True if the model has been built or loaded."""
        return self._voxel_means is not None

    def capture(self, driver, seconds: float = 60) -> None:
        """Build the background model from a capture period.

        The room must be empty (subject absent) during this call.
        Calls ``driver.capture_seconds(seconds)`` and accumulates
        per-voxel statistics across all returned frames.

        Parameters
        ----------
        driver : LidarDriver | RadarDriver
            Any driver with a ``capture_seconds(float)`` method.
        seconds : float
            How long to capture the background.  60 s is a comfortable
            default; 5 s is sufficient for stub/demo mode.
        """
        print(f"  [BackgroundModel] Capturing {seconds:.0f}s of background frames …")
        frames = driver.capture_seconds(seconds)
        self._fit(frames)
        n_voxels = len(self._key_to_idx)
        print(f"  [BackgroundModel] Model built: {n_voxels} occupied voxels "
              f"from {len(frames)} frames.")

    def save(self, path: Path | str) -> None:
        """Persist the model to a numpy .npz file.

        Parameters
        ----------
        path : Path or str
            Destination file path (will be created with parents if needed).
        """
        if not self.is_built:
            raise RuntimeError("BackgroundModel has not been built yet.  "
                               "Call capture() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise the key→index mapping as parallel arrays
        keys_arr = np.array(list(self._key_to_idx.keys()), dtype=np.int32)
        np.savez_compressed(
            path,
            keys=keys_arr,
            means=self._voxel_means,
            stds=self._voxel_stds,
            counts=self._voxel_counts,
        )
        print(f"  [BackgroundModel] Saved to {path}")

    def load(self, path: Path | str) -> None:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : Path or str
            Path to the .npz file written by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Background model not found at {path}. "
                                    "Run --mode background first.")
        data = np.load(path)
        self._voxel_means  = data["means"]
        self._voxel_stds   = data["stds"]
        self._voxel_counts = data["counts"]
        self._bg_packed_sorted = None       # invalidate cache
        self._bg_row_for_sorted = None
        self._key_to_idx   = {
            tuple(row): i for i, row in enumerate(data["keys"])
        }
        print(f"  [BackgroundModel] Loaded {len(self._key_to_idx)} voxels from {path}")

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """Return the subset of *frame* that deviates from the background.

        A point is classified as a **residual** if:
        - Its voxel has never been seen in the background model (new voxel), OR
        - Its per-axis deviation from the voxel mean exceeds
          ``config.background_sigma`` × voxel σ on at least one axis.

        Parameters
        ----------
        frame : np.ndarray
            Shape (N, 3) point cloud to test.

        Returns
        -------
        np.ndarray
            Shape (M, 3) residual points, M ≤ N.  Returns empty array if
            the model has not been built.
        """
        if frame.shape[0] == 0:
            return np.empty((0, 3), dtype=float)
        if not self.is_built:
            return frame   # no model yet → everything is a residual

        vs = self.config.voxel_size
        sigma_thresh = self.config.background_sigma

        # Frame voxel keys, packed into 1-D int64 for fast searchsorted lookup.
        # Same bit-packing trick as lidar_driver._voxel_downsample.
        voxel_indices = (frame / vs).astype(np.int64)
        fpacked = self._pack_keys(voxel_indices)

        # Build/cache sorted bg packed keys + parallel row indices.
        if self._bg_packed_sorted is None:
            self._build_packed_index()

        # Vectorised lookup: pos[i] is where fpacked[i] would slot in.
        pos = np.searchsorted(self._bg_packed_sorted, fpacked)
        pos = np.clip(pos, 0, len(self._bg_packed_sorted) - 1)
        found = self._bg_packed_sorted[pos] == fpacked              # (N,)
        bg_rows = self._bg_row_for_sorted[pos]                       # (N,)

        # Default: every point is a residual. For points whose voxel exists
        # in the background, mark "not residual" iff deviation ≤ threshold.
        residual_mask = np.ones(frame.shape[0], dtype=bool)
        if found.any():
            f_idx = np.where(found)[0]
            rows = bg_rows[f_idx]
            means = self._voxel_means[rows]
            stds = self._voxel_stds[rows]
            deviation = np.max(np.abs(frame[f_idx] - means) / stds, axis=1)
            within = deviation <= sigma_thresh
            residual_mask[f_idx[within]] = False

        return frame[residual_mask]

    # ── Vectorised lookup helpers ─────────────────────────────────────────
    _PACK_OFFSET = 1 << 20   # shift to non-negative; ±10⁶ voxels headroom

    @staticmethod
    def _pack_keys(keys_3d: np.ndarray) -> np.ndarray:
        """Pack int64 (N, 3) voxel indices to (N,) int64 by bit-shifting."""
        shifted = keys_3d + BackgroundModel._PACK_OFFSET
        return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]

    def _build_packed_index(self) -> None:
        """Cache packed bg voxel keys + matching row indices, sorted ascending."""
        n = len(self._key_to_idx)
        keys = np.empty((n, 3), dtype=np.int64)
        rows = np.empty(n, dtype=np.int64)
        for i, (k, r) in enumerate(self._key_to_idx.items()):
            keys[i] = k
            rows[i] = r
        packed = self._pack_keys(keys)
        order = np.argsort(packed)
        self._bg_packed_sorted = packed[order]
        self._bg_row_for_sorted = rows[order]

    def update(self, frame: np.ndarray, rate: float | None = None) -> None:
        """Gradually update the background model for slow scene drift.

        Uses an exponential moving average on the per-voxel mean.
        **Only call this when you are confident the subject is absent.**

        Parameters
        ----------
        frame : np.ndarray
            Shape (N, 3) background-only frame.
        rate : float, optional
            EMA learning rate (0–1).  Defaults to
            ``config.background_update_rate``.
        """
        if not self.is_built:
            return
        if rate is None:
            rate = self.config.background_update_rate

        vs = self.config.voxel_size
        voxel_indices = (frame / vs).astype(int)

        for vkey_row, point in zip(voxel_indices, frame):
            vkey = (int(vkey_row[0]), int(vkey_row[1]), int(vkey_row[2]))
            idx = self._key_to_idx.get(vkey)
            if idx is not None:
                self._voxel_means[idx] = (
                    (1 - rate) * self._voxel_means[idx] + rate * point
                )

    # ── Internal ──────────────────────────────────────────────────────────

    def _fit(self, frames: List[np.ndarray]) -> None:
        """Fit per-voxel statistics from a list of point-cloud frames."""
        vs = self.config.voxel_size
        min_std = self.config.background_min_std

        # Accumulate points per voxel
        voxel_acc: Dict[_VKey, List[np.ndarray]] = {}
        for frame in frames:
            if frame.shape[0] == 0:
                continue
            indices = (frame / vs).astype(int)
            for row, point in zip(indices, frame):
                vkey = (int(row[0]), int(row[1]), int(row[2]))
                if vkey not in voxel_acc:
                    voxel_acc[vkey] = []
                voxel_acc[vkey].append(point)

        # Compute statistics
        n = len(voxel_acc)
        means  = np.zeros((n, 3), dtype=float)
        stds   = np.zeros((n, 3), dtype=float)
        counts = np.zeros(n, dtype=np.int64)
        key_to_idx: Dict[_VKey, int] = {}

        for i, (vkey, pts) in enumerate(voxel_acc.items()):
            arr = np.array(pts)
            means[i]  = arr.mean(axis=0)
            stds[i]   = np.maximum(arr.std(axis=0), min_std)
            counts[i] = len(pts)
            key_to_idx[vkey] = i

        self._voxel_means  = means
        self._voxel_stds   = stds
        self._voxel_counts = counts
        self._key_to_idx   = key_to_idx
        self._bg_packed_sorted = None       # invalidate cache
        self._bg_row_for_sorted = None
