"""
phase1/cubeeye_adapter.py
=========================
Converts CubeEye I200D depth frames into the shared Knight Vision
point-cloud format expected by ``phase1/drivers/lidar_driver.py``.

CubeEye I200D
    640×480 depth image, ~15 fps, depth in millimetres (uint16 or float32).
    Mounted overhead looking down at a cot — Z axis is approximately
    vertical (camera → bed), X/Y are lateral.

Shared KV frame
    X right, Y up, Z forward, metres.

Mapping (camera looking straight down at a horizontal subject):
    cam_x  →  KV X  (right)
    cam_y  →  KV Z  (forward = away from lens)
    cam_z  →  KV -Y (depth from camera becomes -Y in KV; subject is below)

We treat the CubeEye as a pinhole camera with estimated intrinsics
derived from the I200D's 60° FoV at 640×480.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ── CubeEye I200D estimated intrinsics ───────────────────────────────────────
# The I200D has ~60° HFoV.  fx = (W/2) / tan(HFoV/2).
# 60° HFoV → fx = 320 / tan(30°) ≈ 554.  Assume square pixels → fy ≈ fx.
_CUBEEYE_FX = 554.0
_CUBEEYE_FY = 554.0
_CUBEEYE_CX = 320.0   # principal point at image centre
_CUBEEYE_CY = 240.0
_CUBEEYE_W  = 640
_CUBEEYE_H  = 480


def depth_image_to_pointcloud(
    depth_mm: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    fx: float = _CUBEEYE_FX,
    fy: float = _CUBEEYE_FY,
    cx: float = _CUBEEYE_CX,
    cy: float = _CUBEEYE_CY,
    min_depth_mm: float = 100.0,
    max_depth_mm: float = 2000.0,
) -> np.ndarray:
    """Convert a CubeEye depth image to a KV-frame point cloud.

    Parameters
    ----------
    depth_mm : np.ndarray
        (H, W) depth image in millimetres. Zero or NaN = invalid.
    roi : tuple (y1, x1, y2, x2), optional
        If given, only deproject pixels within this rectangle.
        Coordinates are in pixel space (row, col).
    fx, fy, cx, cy : float
        Camera intrinsics (pixels).
    min_depth_mm, max_depth_mm : float
        Depth validity range.

    Returns
    -------
    np.ndarray
        (N, 3) float64, XYZ in metres, shared KV frame
        (X right, Y up, Z forward).
    """
    h, w = depth_mm.shape[:2]

    if roi is not None:
        y1, x1, y2, x2 = roi
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(h, y2), min(w, x2)
        patch = depth_mm[y1:y2, x1:x2].astype(np.float64)
        ys_base, xs_base = np.mgrid[y1:y2, x1:x2]
    else:
        patch = depth_mm.astype(np.float64)
        ys_base, xs_base = np.mgrid[0:h, 0:w]

    # Validity mask
    valid = (patch > min_depth_mm) & (patch < max_depth_mm) & ~np.isnan(patch)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    z_cam_mm = patch[valid]
    xs = xs_base[valid].astype(np.float64)
    ys = ys_base[valid].astype(np.float64)

    # Camera-frame 3D (mm)
    x_cam = (xs - cx) * z_cam_mm / fx
    y_cam = (ys - cy) * z_cam_mm / fy
    z_cam = z_cam_mm

    # Convert mm → metres
    x_cam_m = x_cam / 1000.0
    y_cam_m = y_cam / 1000.0
    z_cam_m = z_cam / 1000.0

    # Map to KV frame (camera looking down at subject):
    #   KV X  =  cam_x   (right)
    #   KV Y  = -cam_z   (up; camera Z is down toward bed)
    #   KV Z  =  cam_y   (forward; cam_y is vertical on the image plane)
    kv_x = x_cam_m
    kv_y = -z_cam_m
    kv_z = y_cam_m

    return np.column_stack([kv_x, kv_y, kv_z])


def depth_roi_to_centroid_z(
    depth_mm: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> float:
    """Extract the mean depth (metres) within an ROI — fast path for breathing signal.

    This bypasses the full point-cloud pipeline and directly computes the
    KV-frame centroid-Z from the ROI depth, which is what the respiratory
    analyser ultimately uses.  Useful for high-throughput replay where
    clustering is unnecessary (CubeEye already has an ROI from the original
    breathing_rate_monitor).

    Parameters
    ----------
    depth_mm : np.ndarray
        (H, W) depth image in millimetres.
    roi : tuple (y1, x1, y2, x2)
        Region of interest in pixel space.

    Returns
    -------
    float
        Mean depth in metres within the ROI.  NaN if no valid pixels.
    """
    y1, x1, y2, x2 = roi
    patch = depth_mm[y1:y2, x1:x2].astype(np.float64)
    valid = (patch > 100.0) & (patch < 2000.0) & ~np.isnan(patch)
    if not valid.any():
        return float("nan")
    return float(patch[valid].mean()) / 1000.0
