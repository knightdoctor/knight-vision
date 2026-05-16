"""
phase1/drivers/lidar_driver.py
================================
LiDAR driver — Femto Bolt depth (iToF) implementation.

Hardware target: Orbbec Femto Bolt via pyorbbecsdk. The Femto Bolt's
iToF depth output is treated as our "LiDAR" channel for Phase 1.

Coordinate frame
----------------
All drivers return XYZ in the shared Knight Vision frame:
**X right, Y up, Z forward, metres**.

Femto native is X right, Y down, Z forward. We negate Y on the way out
of ``_real_capture()``.

Shared session
--------------
Depth + color come from the same pyorbbecsdk Pipeline, owned by
``_femto_session.FemtoBoltSession``. The lidar driver is the
**puller** — every call to ``capture_frame()`` does
``session.wait_and_cache()``. The camera driver is the **ride-along**
— it reads the cached frameset.

To revert to stub
-----------------
Flip ``STUB_MODE = True`` below.

Stub behaviour (when STUB_MODE = True)
--------------------------------------
Background mode (``set_live_mode(False)``):
    Returns a deterministic grid of floor + wall points with tiny
    Gaussian jitter — ensures stable voxel occupancy across frames so
    the background model builds cleanly.

Live mode (``set_live_mode(True)``):
    Adds a person-shaped cluster (torso + head) whose centroid z
    oscillates sinusoidally at BREATHING_FREQ_HZ (default 0.25 Hz,
    ≈ 15 BPM). Lets the full pipeline produce a valid FFT peak with no
    hardware.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

# ── Toggle here for stub vs real Femto Bolt ─────────────────────────────────
STUB_MODE = False
# ────────────────────────────────────────────────────────────────────────────

# Voxel grid for downsampling the ~368k pts/frame from the Femto Bolt
# down to ~5–10k pts before DBSCAN. 3 cm balances detail vs cluster perf.
_VOXEL_SIZE_M = 0.03

# Synthetic breathing parameters (stub only)
_BREATHING_FREQ_HZ = 0.25        # 15 BPM
_BREATHING_AMP_M   = 0.025       # ±2.5 cm chest-rise amplitude
_PERSON_CENTRE     = np.array([0.0, 0.3, 1.0])   # metres from sensor origin


# Shift voxel keys to non-negative for 1-D packing. 21 bits/axis = ±10⁶ voxels
# (≈ ±30 km at 3 cm) — vastly more headroom than needed for indoor scenes.
_VOXEL_OFFSET_BITS = 20
_VOXEL_OFFSET = 1 << _VOXEL_OFFSET_BITS


def _voxel_downsample(pts: np.ndarray, voxel: float = _VOXEL_SIZE_M) -> np.ndarray:
    """Bin points to a voxel grid; keep one representative per occupied voxel.

    The naive ``np.unique(keys, axis=0)`` cost ~370 ms/frame on 280 k pts —
    the dominant bottleneck. Bit-packing (i, j, k) into one int64 then doing
    a 1-D unique drops that to ~10 ms.
    """
    if pts.shape[0] == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64) + _VOXEL_OFFSET
    packed = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]
    _, idx = np.unique(packed, return_index=True)
    return pts[idx]


class LidarDriver:
    """Femto Bolt depth driver (or deterministic stub).

    Parameters
    ----------
    config : KVConfig
        Shared pipeline configuration (uses ``lidar_fps``).

    Usage
    -----
    >>> driver = LidarDriver(config)
    >>> frame = driver.capture_frame()          # (N, 3) np.ndarray, metres
    >>> frames = driver.capture_seconds(5.0)    # list of (N, 3) arrays
    """

    def __init__(self, config) -> None:
        self.config = config
        self._live_mode: bool = False
        self._frame_counter: int = 0
        self._bg_base = self._build_background_base()

    # ── Public API ────────────────────────────────────────────────────────

    def set_live_mode(self, live: bool) -> None:
        """Switch between background-only and subject-present modes.

        For real hardware this is just a flag — the sensor streams
        continuously regardless. For the stub it controls whether a
        synthetic person cluster is included and resets the breathing
        phase counter.
        """
        self._live_mode = live
        if live and STUB_MODE:
            self._frame_counter = 0

    def capture_frame(self) -> np.ndarray:
        """Capture one depth frame as an XYZ point cloud.

        Returns
        -------
        np.ndarray
            Shape (N, 3), dtype float64. XYZ in metres, shared frame
            (X right, Y up, Z forward). Empty (0, 3) on dropped frames.
        """
        if STUB_MODE:
            return self._synthetic_frame()
        return self._real_capture()

    def capture_seconds(self, seconds: float) -> List[np.ndarray]:
        """Capture *seconds* worth of frames at the configured LiDAR FPS.

        In stub mode there is no sleep between frames (instant playback).
        In real mode the pyorbbecsdk Pipeline paces itself; we just call
        ``capture_frame()`` in a tight loop and let ``wait_for_frames``
        block on the next sensor frame.
        """
        n_frames = max(1, int(seconds * self.config.lidar_fps))
        frames: List[np.ndarray] = []
        for _ in range(n_frames):
            frames.append(self.capture_frame())
            if STUB_MODE:
                pass  # instant
        return frames

    # ── Real hardware ─────────────────────────────────────────────────────

    def _real_capture(self) -> np.ndarray:
        """Pull one depth frame from the shared Femto Bolt session."""
        from ._femto_session import get_session

        session = get_session(
            expected_lidar_fps=getattr(self.config, "lidar_fps", None)
        )
        frames = session.wait_and_cache(timeout_ms=200)
        if frames is None:
            return np.empty((0, 3), dtype=np.float64)

        depth = frames.get_depth_frame()
        if depth is None:
            return np.empty((0, 3), dtype=np.float64)

        h, w = depth.get_height(), depth.get_width()
        raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w)

        valid = raw > 0
        if not valid.any():
            return np.empty((0, 3), dtype=np.float64)

        ys, xs = np.where(valid)
        z_m = raw[ys, xs].astype(np.float64) / 1000.0
        x_m = (xs.astype(np.float64) - session.cx) * z_m / session.fx
        y_m = (ys.astype(np.float64) - session.cy) * z_m / session.fy

        # Femto native: X right, Y down, Z forward.
        # Shared KV frame:  X right, Y up,   Z forward → negate Y.
        pts = np.column_stack([x_m, -y_m, z_m])

        return _voxel_downsample(pts)

    # ── Stub synthesis ────────────────────────────────────────────────────

    @staticmethod
    def _build_background_base() -> np.ndarray:
        """Deterministic 4 m × 4 m floor grid + back wall (stub only)."""
        xs = np.arange(-2.0, 2.05, 0.3)
        ys = np.arange(-2.0, 2.05, 0.3)
        xx, yy = np.meshgrid(xs, ys)
        floor = np.column_stack([xx.ravel(), yy.ravel(),
                                  np.zeros(xx.size)])

        xw = np.arange(-2.0, 2.05, 0.4)
        zw = np.arange(0.0, 2.05, 0.4)
        xww, zww = np.meshgrid(xw, zw)
        wall = np.column_stack([xww.ravel(),
                                 np.full(xww.size, -2.0),
                                 zww.ravel()])
        return np.vstack([floor, wall]).astype(float)

    def _synthetic_frame(self) -> np.ndarray:
        """Return one synthetic LiDAR frame (stub only)."""
        rng = np.random.default_rng()
        bg = self._bg_base + rng.normal(0.0, 0.003, self._bg_base.shape)

        if not self._live_mode:
            return bg

        t = self._frame_counter / self.config.lidar_fps
        self._frame_counter += 1
        dz = _BREATHING_AMP_M * np.sin(2.0 * np.pi * _BREATHING_FREQ_HZ * t)
        centre = _PERSON_CENTRE + np.array([0.0, 0.0, dz])

        torso = rng.normal(centre,           [0.15, 0.08, 0.20], (100, 3))
        head  = rng.normal(centre + [0, 0, 0.38], 0.06,          ( 20, 3))
        return np.vstack([bg, torso, head])
