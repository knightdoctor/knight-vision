"""
phase1/drivers/camera_driver.py
================================
RGB camera driver — Femto Bolt color channel implementation.

The Femto Bolt provides depth and color from one physical sensor; both
streams come from the same pyorbbecsdk Pipeline owned by
``_femto_session.FemtoBoltSession``. This driver is the **ride-along**
consumer: it never calls ``wait_for_frames`` itself — the lidar driver
does that and caches the result. The camera driver reads color from
the cached frameset.

The RGB camera is NOT in the Phase 1 hot processing path
(geometry-first). It is used for:
  • Optional subject-presence confirmation
  • Audit / ground-truth recording alongside LiDAR / radar
  • Future colour-aware fusion

Coordinate frame
----------------
Returns BGR images, not point clouds — so no spatial frame applies.
For the depth/radar XYZ convention see lidar_driver.py.

To revert to stub
-----------------
Flip ``STUB_MODE = True`` below.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

# ── Toggle here for stub vs real Femto Bolt ─────────────────────────────────
STUB_MODE = False
# ────────────────────────────────────────────────────────────────────────────

_STUB_RESOLUTION: Tuple[int, int] = (480, 640)   # (height, width) pixels


class CameraDriver:
    """Femto Bolt color channel (or blank stub)."""

    def __init__(self, config, device_index: int = 0) -> None:
        self.config = config
        self.device_index = device_index   # unused for Femto Bolt path

    # ── Public API ────────────────────────────────────────────────────────

    def capture_frame(self) -> np.ndarray:
        """Capture one color frame.

        Returns
        -------
        np.ndarray
            Shape (H, W, 3), dtype uint8, BGR. Stub returns a blank
            grey frame; real path returns the decoded color image from
            the latest cached Femto Bolt frameset.
        """
        if STUB_MODE:
            return self._blank_frame()
        return self._real_capture()

    def capture_seconds(self, seconds: float) -> List[np.ndarray]:
        """Capture *seconds* worth of frames at the configured camera FPS."""
        n_frames = max(1, int(seconds * self.config.camera_fps))
        frames: List[np.ndarray] = []
        for _ in range(n_frames):
            frames.append(self.capture_frame())
            if STUB_MODE:
                pass
            else:
                time.sleep(1.0 / self.config.camera_fps)
        return frames

    def release(self) -> None:
        """No-op — the underlying Pipeline is owned by the shared session."""
        return

    # ── Stub ─────────────────────────────────────────────────────────────

    @staticmethod
    def _blank_frame() -> np.ndarray:
        h, w = _STUB_RESOLUTION
        return np.full((h, w, 3), 128, dtype=np.uint8)

    # ── Real hardware ─────────────────────────────────────────────────────

    def _real_capture(self) -> np.ndarray:
        """Read color from the cached Femto Bolt frameset."""
        from ._femto_session import get_session
        import cv2

        session = get_session(
            expected_lidar_fps=getattr(self.config, "lidar_fps", None)
        )
        frames = session.get_cached(timeout_ms=200)
        if frames is None:
            return self._blank_frame()

        color = frames.get_color_frame()
        if color is None:
            return self._blank_frame()

        decoded = _decode_color(color)
        return decoded if decoded is not None else self._blank_frame()


def _decode_color(frame) -> Optional[np.ndarray]:
    """Decode a Femto Bolt color frame to BGR.

    Mirrors knightvision-hub/scripts/femto_multiview.py:decode_color —
    tries MJPG (cv2.imdecode) first, falls back to YUYV unpack.
    """
    import cv2

    try:
        w, h = frame.get_width(), frame.get_height()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        yuyv = data.reshape(h, w, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    except Exception:
        return None
