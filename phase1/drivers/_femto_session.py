"""
phase1/drivers/_femto_session.py
=================================
Singleton owner of the one pyorbbecsdk Pipeline shared by the Femto Bolt
depth (lidar) and color (camera) drivers.

Why a singleton
---------------
The Femto Bolt is a single physical sensor that exposes both depth and
color streams. pyorbbecsdk requires both streams to be enabled on a
single Config attached to a single Pipeline; opening two Pipelines for
the same device fails. We also can't have two consumers calling
``wait_for_frames`` independently — frames would race between them.

Convention
----------
- Lidar driver pulls every loop iteration: ``session.wait_and_cache()``.
- Camera driver reads from the cache: ``session.get_cached()`` (only
  blocks if no frame has been pulled yet).

Coordinate frame
----------------
Femto native is X right, Y down, Z forward. The shared Knight Vision
frame is X right, Y up, Z forward. Y-negation is applied by the lidar
driver, not here — the session returns the raw frameset.
"""
from __future__ import annotations

import threading
import warnings

# Approximate Femto Bolt depth intrinsics for 640x576 NFOV unbinned.
# Used as a fallback if pipeline.get_camera_param().depth_intrinsic
# attribute names differ on the installed SDK build.
_FALLBACK_INTRINSICS = (504.0, 504.0, 320.0, 288.0)   # fx, fy, cx, cy

_SESSION: "FemtoBoltSession | None" = None
_LOCK = threading.Lock()


class FemtoBoltSession:
    """Owns one pyorbbecsdk Pipeline with depth + color streams enabled."""

    def __init__(self, expected_lidar_fps: float | None = None) -> None:
        from pyorbbecsdk import Pipeline, Config, OBSensorType

        self.pipeline = Pipeline()
        cfg = Config()

        depth_profile = (
            self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            .get_default_video_stream_profile()
        )
        cfg.enable_stream(depth_profile)

        color_profile = (
            self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            .get_default_video_stream_profile()
        )
        cfg.enable_stream(color_profile)

        self.pipeline.start(cfg)

        depth_w, depth_h = depth_profile.get_width(), depth_profile.get_height()
        depth_fps = depth_profile.get_fps()
        color_w, color_h = color_profile.get_width(), color_profile.get_height()
        color_fps = color_profile.get_fps()

        print(f"[FemtoBolt] depth: {depth_w}x{depth_h} @ {depth_fps} fps")
        print(f"[FemtoBolt] color: {color_w}x{color_h} @ {color_fps} fps")

        self.depth_fps = float(depth_fps)
        self.color_fps = float(color_fps)
        self.depth_size = (depth_w, depth_h)

        if expected_lidar_fps is not None and abs(expected_lidar_fps - depth_fps) > 0.5:
            print(
                f"[FemtoBolt] WARNING: config.lidar_fps={expected_lidar_fps} "
                f"differs from effective depth fps={depth_fps}. RR FFT will "
                f"misinterpret frame spacing — update KVConfig.lidar_fps."
            )

        # Intrinsics: try the standard pyorbbecsdk path, fall back if attr
        # names differ on this SDK build.
        try:
            cam = self.pipeline.get_camera_param()
            intr = cam.depth_intrinsic
            self.fx = float(intr.fx)
            self.fy = float(intr.fy)
            self.cx = float(intr.cx)
            self.cy = float(intr.cy)
            print(f"[FemtoBolt] intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f}")
        except Exception as e:
            self.fx, self.fy, self.cx, self.cy = _FALLBACK_INTRINSICS
            warnings.warn(
                f"[FemtoBolt] could not read depth intrinsics ({type(e).__name__}: {e}); "
                f"using fallback fx=fy=504, cx=320, cy=288 for 640x576 NFOV unbinned. "
                f"Verify pipeline.get_camera_param() API on this SDK build.",
                stacklevel=2,
            )

        self._latest = None  # cached FrameSet from the most recent wait_for_frames

        # Warm-up: the sensor takes ~100–500 ms after pipeline.start() before
        # it produces frames. Drain a few with a generous timeout so the
        # first user-visible capture_frame() returns real data, not (0, 3).
        for _ in range(10):
            f = self.pipeline.wait_for_frames(500)
            if f is not None and f.get_depth_frame() is not None:
                self._latest = f
                break

    def wait_and_cache(self, timeout_ms: int = 200):
        """Block up to ``timeout_ms`` for the next frameset, drain any older
        queued framesets so the caller always gets the *latest* one, cache
        it, and return it.

        Why drain: the pyorbbecsdk Pipeline buffers frames internally. If
        downstream processing runs slower than 30 fps, the queue grows and
        ``wait_for_frames`` returns increasingly stale data. Draining keeps
        us real-time at the cost of dropping intermediate frames.
        """
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        # Non-blocking drain — keep only the freshest frameset.
        for _ in range(60):  # safety cap
            more = self.pipeline.wait_for_frames(1)
            if more is None:
                break
            frames = more
        with _LOCK:
            self._latest = frames
        return frames

    def get_cached(self, timeout_ms: int = 200):
        """Return cached frameset; if none yet, pull one with a short block."""
        with _LOCK:
            cached = self._latest
        if cached is not None:
            return cached
        return self.wait_and_cache(timeout_ms)

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def get_session(expected_lidar_fps: float | None = None) -> FemtoBoltSession:
    """Lazy module-level accessor. ``expected_lidar_fps`` only used on first call."""
    global _SESSION
    if _SESSION is None:
        _SESSION = FemtoBoltSession(expected_lidar_fps=expected_lidar_fps)
    return _SESSION
