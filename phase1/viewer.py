"""
phase1/viewer.py
================
Live web viewer for the Phase 1 pipeline.

Renders the current residual point cloud (top-down X/Z), highlights the
detected subject cluster, and plots the centroid-z time series. Runs on
port 5005 by default.

Render is done with cv2 (not matplotlib) because matplotlib's render
takes ~350 ms/frame on the Jetson — that blew the per-frame budget AND
held the GIL long enough to halve the pipeline rate. cv2 renders the
same scene in ~10 ms.

Usage
-----
    from phase1.viewer import set_frame, start
    start(port=5005)
    # ... in pipeline frame_callback:
    set_frame(i, residuals, subject, n_clusters, cz)
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# ── Tunables ────────────────────────────────────────────────────────────
RENDER_FPS = 10                     # render loop target rate
CZ_HISTORY_MAX = 1800               # 60 s × 30 fps headroom
TOPDOWN_X_LIMITS = (-1.5, 1.5)      # metres
TOPDOWN_Z_LIMITS = (0.5, 3.5)       # metres — typical subject range, kills empty zones
PANEL_W = 600                       # top-down panel width  (px)
PANEL_H = 360                       # top-down panel height (px) — tightened
WAVEFORM_W = 520                    # respiratory waveform render width
WAVEFORM_H = 220                    # respiratory waveform render height
SPARKLINE_SAMPLES = 60              # downsampled cz history sent to client

# Colours (BGR for cv2)
BG     = (10, 10, 10)
PANEL  = (26, 26, 26)
GRID   = (40, 40, 40)
TEXT   = (220, 220, 220)
DIM    = (140, 140, 140)
RESID  = (220, 120, 30)             # blue-ish — residual cloud
SUBJ   = (40, 170, 250)             # orange — subject cluster
CENTRD = (60, 80, 240)              # red — centroid ring
TRACE  = (120, 220, 60)             # green — centroid-z trace

# ── Shared state ────────────────────────────────────────────────────────
_LOCK = threading.Lock()
_state = {
    "frame_no":   0,
    "residuals":  np.empty((0, 3), dtype=float),
    "subject":    None,
    "n_clusters": 0,
    "cz_history": [],
    "started_at": None,
    "rr_bpm":     None,
    "rr_snr":     None,
    "rr_conf":    None,
    "rr_n":       0,
    "frame_ts":   [],   # recent set_frame wall times for live fps calc
    # Radar sidecar state (PR Q 2026-05-18 viewer integration)
    "radar_pts":  np.empty((0, 3), dtype=np.float32),
    "radar_n":    0,
    "radar_active": False,
    "radar_rr_bpm":  None,
    "radar_rr_snr":  None,
    "radar_rr_conf": None,
    "radar_rr_n": 0,
    # Trail buffer for visualisation: list of (recv_ts, points). Radar
    # is too sparse (5-15 pts/frame) to read as a single-frame panel;
    # accumulating the last few seconds of detections builds up enough
    # density to see the subject. Newer drawn brighter, older faded.
    "radar_trail": [],
}

# Static metadata set once at startup (band limits, sensor info, etc).
_meta = {
    "band_min_bpm": None,
    "band_max_bpm": None,
}
_topdown_lock = threading.Lock()
_latest_topdown_jpg: Optional[bytes] = None
_waveform_lock = threading.Lock()
_latest_waveform_jpg: Optional[bytes] = None
_rgb_jpg_lock = threading.Lock()
_latest_rgb_jpg: Optional[bytes] = None
_rgb_active = False
_depth_jpg_lock = threading.Lock()
_latest_depth_jpg: Optional[bytes] = None
_depth_active = False
_radar_topdown_lock = threading.Lock()
_latest_radar_topdown_jpg: Optional[bytes] = None
_radar_side_lock = threading.Lock()
_latest_radar_side_jpg: Optional[bytes] = None
_started = False

# Radar geometry — IWR6843AOPEVM antenna-on-package.
RADAR_FOV_AZ_DEG = 60.0   # ±60° azimuth (X-Z plane half-angle)
RADAR_FOV_EL_DEG = 60.0   # ±60° elevation (Y-Z plane half-angle)
RADAR_SIDE_PANEL_H = 220  # px — shorter panel for side elevation view
RADAR_SIDE_Y_LIMITS = (-0.8, 1.0)   # metres (down → up)

# Control state (browser → pipeline). Pipeline polls these each frame.
_control = {
    "recording":    False,   # toggle: when True, orchestrator persists frames
    "stop_session": False,   # one-shot: pipeline breaks out of run_live
    "record_seq":   0,       # increments on each False→True transition
}

# Ground-truth state (manual entry now, Polar/Garmin daemon later).
_gt_lock = threading.Lock()
_gt_state = {
    "current": None,    # most-recent entry dict
    "history": [],      # append-only list of all entries this session
}

app = Flask(__name__)


def set_frame(frame_no: int, residuals: np.ndarray, subject,
              n_clusters: int, cz: float) -> None:
    """Push the latest pipeline frame data into shared viewer state."""
    with _LOCK:
        if _state["started_at"] is None:
            _state["started_at"] = time.time()
        _state["frame_no"]   = frame_no
        _state["residuals"]  = residuals if residuals is not None else np.empty((0, 3))
        _state["subject"]    = subject
        _state["n_clusters"] = n_clusters
        hist = _state["cz_history"]
        hist.append(cz)
        if len(hist) > CZ_HISTORY_MAX:
            del hist[: len(hist) - CZ_HISTORY_MAX]
        # Track last ~3s of frame timestamps for a rolling fps estimate.
        ts = _state["frame_ts"]
        now = time.time()
        ts.append(now)
        cutoff = now - 3.0
        while ts and ts[0] < cutoff:
            ts.pop(0)


def set_rr(rr_bpm: float, snr: float, confidence: str) -> None:
    """Push the latest RR estimate (called once per FFT window)."""
    with _LOCK:
        _state["rr_bpm"]  = float(rr_bpm)
        _state["rr_snr"]  = float(snr)
        _state["rr_conf"] = str(confidence)
        _state["rr_n"]   += 1


def reset() -> None:
    with _LOCK:
        _state["frame_no"]   = 0
        _state["residuals"]  = np.empty((0, 3), dtype=float)
        _state["subject"]    = None
        _state["n_clusters"] = 0
        _state["cz_history"] = []
        _state["started_at"] = None
        _state["rr_bpm"]     = None
        _state["rr_snr"]     = None
        _state["rr_conf"]    = None
        _state["rr_n"]       = 0
        _state["frame_ts"]   = []
        _state["radar_pts"]      = np.empty((0, 3), dtype=np.float32)
        _state["radar_n"]        = 0
        _state["radar_active"]   = False
        _state["radar_rr_bpm"]   = None
        _state["radar_rr_snr"]   = None
        _state["radar_rr_conf"]  = None
        _state["radar_rr_n"]     = 0
        _state["radar_trail"]    = []
        _control["recording"]    = False
        _control["stop_session"] = False
        _control["record_seq"]   = 0
    with _gt_lock:
        _gt_state["current"] = None
        _gt_state["history"] = []


def post_gt(hr=None, rr=None, source: str = "manual") -> dict:
    """Append a ground-truth reading. Either field may be None.
    Returns the entry dict (with ts_iso filled in)."""
    from datetime import datetime as _dt
    entry = {
        "ts_iso": _dt.now().isoformat(timespec="milliseconds"),
        "source": str(source),
        "hr":     None if hr is None else float(hr),
        "rr":     None if rr is None else float(rr),
    }
    with _gt_lock:
        _gt_state["current"] = entry
        _gt_state["history"].append(entry)
    return entry


def get_gt_history() -> list:
    with _gt_lock:
        return list(_gt_state["history"])


def get_gt_current() -> Optional[dict]:
    with _gt_lock:
        return dict(_gt_state["current"]) if _gt_state["current"] else None


def set_meta(band_min_bpm: float, band_max_bpm: float) -> None:
    with _LOCK:
        _meta["band_min_bpm"] = float(band_min_bpm)
        _meta["band_max_bpm"] = float(band_max_bpm)


def set_radar_frame(points: np.ndarray) -> None:
    """Sidecar pushes the latest radar frame (N, 3) in shared XYZ metres.
    Stored as latest-frame + appended to trail for accumulation in the
    top-down render."""
    pts = (points.astype(np.float32)
           if points is not None and len(points) else
           np.empty((0, 3), dtype=np.float32))
    with _LOCK:
        _state["radar_pts"] = pts
        _state["radar_n"]   = int(len(pts))
        _state["radar_active"] = True
        _state["radar_trail"].append((time.time(), pts))
        # Cap trail age at 5s; sidecar pushes ~10 Hz so ~50 entries.
        cutoff = time.time() - 5.0
        _state["radar_trail"] = [(t, p) for (t, p) in _state["radar_trail"]
                                 if t >= cutoff]


def set_radar_rr(rr_bpm: float, snr: float, confidence: str) -> None:
    """Sidecar pushes the latest radar RR estimate."""
    with _LOCK:
        _state["radar_rr_bpm"]  = float(rr_bpm) if rr_bpm is not None else None
        _state["radar_rr_snr"]  = float(snr) if snr is not None else None
        _state["radar_rr_conf"] = str(confidence) if confidence else None
        _state["radar_rr_n"]   += 1


def set_rgb(jpeg_bytes: bytes) -> None:
    """Orchestrator pushes a pre-encoded JPEG of the current colour frame."""
    global _latest_rgb_jpg, _rgb_active
    with _rgb_jpg_lock:
        _latest_rgb_jpg = jpeg_bytes
        _rgb_active = True


def set_depth(jpeg_bytes: bytes) -> None:
    """Orchestrator pushes a pre-encoded JPEG of the depth heatmap."""
    global _latest_depth_jpg, _depth_active
    with _depth_jpg_lock:
        _latest_depth_jpg = jpeg_bytes
        _depth_active = True


def set_recording(on: bool) -> None:
    """Orchestrator-side setter — keeps viewer state in sync when the
    pipeline starts/stops recording for reasons other than a button press
    (e.g. --preview flag default, duration auto-stop)."""
    with _LOCK:
        prev = _control["recording"]
        _control["recording"] = bool(on)
        if (not prev) and on:
            _control["record_seq"] += 1


def get_control() -> dict:
    with _LOCK:
        return dict(_control)


# ── Rendering ───────────────────────────────────────────────────────────

def _world_to_topdown_px(x: np.ndarray, z: np.ndarray) -> tuple:
    """Map world (X right, Z forward) metres → top-down panel pixels."""
    xmin, xmax = TOPDOWN_X_LIMITS
    zmin, zmax = TOPDOWN_Z_LIMITS
    px = ((x - xmin) / (xmax - xmin) * PANEL_W).astype(np.int32)
    # Z forward → up the screen; flip y axis.
    py = (PANEL_H - (z - zmin) / (zmax - zmin) * PANEL_H).astype(np.int32)
    return px, py


def _draw_topdown(canvas: np.ndarray, residuals: np.ndarray,
                  subject, n_clusters: int, frame_no: int) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, h), PANEL, -1)
    # Grid: every 1 m
    for m in range(int(TOPDOWN_X_LIMITS[0]), int(TOPDOWN_X_LIMITS[1]) + 1):
        x_px = int((m - TOPDOWN_X_LIMITS[0]) / (TOPDOWN_X_LIMITS[1] - TOPDOWN_X_LIMITS[0]) * PANEL_W)
        cv2.line(canvas, (x_px, 0), (x_px, PANEL_H), GRID, 1)
        cv2.putText(canvas, f"{m:+d}", (x_px + 2, PANEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
    for m in range(int(TOPDOWN_Z_LIMITS[0]), int(TOPDOWN_Z_LIMITS[1]) + 1):
        y_px = int(PANEL_H - (m - TOPDOWN_Z_LIMITS[0]) / (TOPDOWN_Z_LIMITS[1] - TOPDOWN_Z_LIMITS[0]) * PANEL_H)
        cv2.line(canvas, (0, y_px), (PANEL_W, y_px), GRID, 1)
        cv2.putText(canvas, f"{m}m", (4, y_px - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)

    # Residual cloud
    if len(residuals):
        x_px, y_px = _world_to_topdown_px(residuals[:, 0], residuals[:, 2])
        in_view = (x_px >= 0) & (x_px < PANEL_W) & (y_px >= 0) & (y_px < PANEL_H)
        x_px, y_px = x_px[in_view], y_px[in_view]
        # Vectorised dot fill via numpy index — single pixel each, fast.
        canvas[y_px, x_px] = RESID

    # Subject cluster (slightly bigger so they pop)
    if subject is not None and len(subject):
        x_px, y_px = _world_to_topdown_px(subject[:, 0], subject[:, 2])
        in_view = (x_px >= 0) & (x_px < PANEL_W) & (y_px >= 0) & (y_px < PANEL_H)
        for u, v in zip(x_px[in_view], y_px[in_view]):
            cv2.circle(canvas, (int(u), int(v)), 2, SUBJ, -1)
        cx = float(subject[:, 0].mean())
        cz = float(subject[:, 2].mean())
        cu, cv_ = _world_to_topdown_px(np.array([cx]), np.array([cz]))
        cv2.circle(canvas, (int(cu[0]), int(cv_[0])), 14, CENTRD, 2)

    # HUD strip
    cv2.rectangle(canvas, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    msg = f"frame {frame_no}  |  residuals {len(residuals)}  |  clusters {n_clusters}"
    cv2.putText(canvas, msg, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                TEXT, 1, cv2.LINE_AA)


def _draw_waveform(canvas: np.ndarray, cz_hist: list) -> None:
    """Render respiratory waveform onto a standalone (WAVEFORM_W x WAVEFORM_H)
    canvas. Detrends cz_hist (median-subtract), shows last ~10s."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, h), PANEL, -1)
    x0 = 0
    chart_w = w
    panel_h = h

    cv2.putText(canvas, "Respiratory waveform · Z displacement (mm) · last 10s",
                (x0 + 8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT, 1, cv2.LINE_AA)

    arr_full = np.asarray(cz_hist, dtype=float)
    if arr_full.size == 0:
        return

    # Show the last ~10s window. Pipeline pushes one sample per frame at
    # ~30 Hz so 300 samples ≈ 10s.
    WINDOW = 300
    arr = arr_full[-WINDOW:] if arr_full.size > WINDOW else arr_full

    valid = ~np.isnan(arr)
    if not valid.any():
        cv2.putText(canvas, "no subject yet", (x0 + 12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)
        return

    median = float(np.median(arr[valid]))
    detrended_mm = np.where(valid, (arr - median) * 1000.0, np.nan)

    valid_vals_mm = detrended_mm[valid]
    amp = max(10.0, float(np.max(np.abs(valid_vals_mm))) * 1.15)
    ymin, ymax = -amp, +amp

    # Centre zero line + 4 grid lines.
    y_zero = int(panel_h * 0.5)
    cv2.line(canvas, (x0, y_zero), (x0 + chart_w, y_zero), (60, 60, 60), 1)
    for frac in (0.15, 0.35, 0.65, 0.85):
        y = int(panel_h * frac)
        cv2.line(canvas, (x0, y), (x0 + chart_w, y), GRID, 1)
        v = ymax - frac * (ymax - ymin)
        cv2.putText(canvas, f"{v:+.0f}", (x0 + 4, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1, cv2.LINE_AA)

    n = len(detrended_mm)
    xs = (np.arange(n) / max(n - 1, 1) * (chart_w - 1)).astype(np.int32) + x0
    ys = np.where(np.isnan(detrended_mm),
                  -1,
                  (panel_h - (detrended_mm - ymin) / (ymax - ymin) * panel_h).astype(np.int32))
    pts = []
    for x, y in zip(xs, ys):
        if y < 0:
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                              TRACE, 2, cv2.LINE_AA)
            pts = []
        else:
            pts.append([x, y])
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                      TRACE, 2, cv2.LINE_AA)


def _render_topdown_jpeg() -> Optional[bytes]:
    with _LOCK:
        residuals  = _state["residuals"]
        subject    = _state["subject"]
        n_clusters = _state["n_clusters"]
        frame_no   = _state["frame_no"]
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    _draw_topdown(canvas, residuals, subject, n_clusters, frame_no)
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return jpg.tobytes() if ok else None


def _render_waveform_jpeg() -> Optional[bytes]:
    with _LOCK:
        cz_hist = list(_state["cz_history"])
    canvas = np.full((WAVEFORM_H, WAVEFORM_W, 3), BG, dtype=np.uint8)
    _draw_waveform(canvas, cz_hist)
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return jpg.tobytes() if ok else None


def _render_radar_topdown_jpeg() -> Optional[bytes]:
    """Radar top-down panel (PR Q). Same X/Z bounds as LiDAR top-down for
    direct spatial comparison. Renders a 3-second TRAIL of detections —
    radar is too sparse (5-15 pts/frame) for single-frame to be readable;
    accumulating recent frames builds up enough density to see the subject.
    Newer frames drawn brighter, older faded by age."""
    with _LOCK:
        trail = list(_state["radar_trail"])
        n_now = _state["radar_n"]
        active = _state["radar_active"]
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_W, PANEL_H), PANEL, -1)
    # Same grid as LiDAR top-down for spatial alignment.
    for m in range(int(TOPDOWN_X_LIMITS[0]), int(TOPDOWN_X_LIMITS[1]) + 1):
        x_px = int((m - TOPDOWN_X_LIMITS[0]) /
                   (TOPDOWN_X_LIMITS[1] - TOPDOWN_X_LIMITS[0]) * PANEL_W)
        cv2.line(canvas, (x_px, 0), (x_px, PANEL_H), GRID, 1)
        cv2.putText(canvas, f"{m:+d}", (x_px + 2, PANEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
    for m in range(int(TOPDOWN_Z_LIMITS[0]), int(TOPDOWN_Z_LIMITS[1]) + 1):
        y_px = int(PANEL_H - (m - TOPDOWN_Z_LIMITS[0]) /
                   (TOPDOWN_Z_LIMITS[1] - TOPDOWN_Z_LIMITS[0]) * PANEL_H)
        cv2.line(canvas, (0, y_px), (PANEL_W, y_px), GRID, 1)
        cv2.putText(canvas, f"{m}m", (4, y_px - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
    # ── FOV cone overlay: ±60° azimuth from sensor at (X=0, Z=0).
    # Sensor in this panel sits at x_sensor_px (X=0) at the BOTTOM of
    # the canvas (Z=Z_min line). Cone fans upward and outward.
    import math
    x_sensor_px, _ = _world_to_topdown_px(np.array([0.0]),
                                          np.array([TOPDOWN_Z_LIMITS[0]]))
    x_sensor_px = int(x_sensor_px[0])
    sensor_y_px = PANEL_H
    # Build cone polygon (sensor + two edge points at max range).
    rng = TOPDOWN_Z_LIMITS[1] - TOPDOWN_Z_LIMITS[0]
    tan_az = math.tan(math.radians(RADAR_FOV_AZ_DEG))
    edge_x_left  =  -tan_az * rng
    edge_x_right = +tan_az * rng
    xl_px, yl_px = _world_to_topdown_px(np.array([edge_x_left]),
                                         np.array([TOPDOWN_Z_LIMITS[1]]))
    xr_px, yr_px = _world_to_topdown_px(np.array([edge_x_right]),
                                         np.array([TOPDOWN_Z_LIMITS[1]]))
    cone = np.array([[x_sensor_px, sensor_y_px],
                     [int(xl_px[0]), int(yl_px[0])],
                     [int(xr_px[0]), int(yr_px[0])]], dtype=np.int32)
    # Faint fill inside cone (visible "radar can see here" region).
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [cone], (50, 30, 60))
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
    # Cone edges.
    cv2.line(canvas, (x_sensor_px, sensor_y_px),
             (int(xl_px[0]), int(yl_px[0])), (100, 60, 130), 1, cv2.LINE_AA)
    cv2.line(canvas, (x_sensor_px, sensor_y_px),
             (int(xr_px[0]), int(yr_px[0])), (100, 60, 130), 1, cv2.LINE_AA)
    # Range arcs at 1, 2, 3 m.
    for r in (1.0, 2.0, 3.0):
        if r < TOPDOWN_Z_LIMITS[0] or r > TOPDOWN_Z_LIMITS[1]:
            continue
        # Sample arc points at azimuth -60° .. +60°.
        arc_pts = []
        for az_deg in range(-int(RADAR_FOV_AZ_DEG), int(RADAR_FOV_AZ_DEG) + 1, 5):
            az = math.radians(az_deg)
            x_w = r * math.sin(az)
            z_w = r * math.cos(az)
            if z_w < TOPDOWN_Z_LIMITS[0] or z_w > TOPDOWN_Z_LIMITS[1]:
                continue
            xp, yp = _world_to_topdown_px(np.array([x_w]), np.array([z_w]))
            arc_pts.append([int(xp[0]), int(yp[0])])
        if len(arc_pts) >= 2:
            cv2.polylines(canvas, [np.array(arc_pts, dtype=np.int32)],
                          False, (80, 50, 100), 1, cv2.LINE_AA)
            # Range label at azimuth=0 (centre).
            xp, yp = _world_to_topdown_px(np.array([0.0]), np.array([r]))
            cv2.putText(canvas, f"{r:.0f}m", (int(xp[0]) + 4, int(yp[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 80, 140),
                        1, cv2.LINE_AA)

    # Render trail oldest -> newest so newer pixels overwrite older ones.
    now = time.time()
    total_pts = 0
    for ts, pts in trail:
        if len(pts) == 0:
            continue
        age = max(0.0, now - ts)
        # Fade: 1.0 at age 0 → 0.15 at age 5s.
        alpha = max(0.15, 1.0 - (age / 5.0) * 0.85)
        col = tuple(int(c * alpha) for c in (255, 60, 200))
        x_px, y_px = _world_to_topdown_px(pts[:, 0], pts[:, 2])
        in_view = ((x_px >= 0) & (x_px < PANEL_W)
                   & (y_px >= 0) & (y_px < PANEL_H))
        for u, v in zip(x_px[in_view], y_px[in_view]):
            cv2.circle(canvas, (int(u), int(v)), 4, col, -1)
        total_pts += int(in_view.sum())
    # HUD strip.
    cv2.rectangle(canvas, (0, 0), (PANEL_W, 22), (0, 0, 0), -1)
    if not active:
        msg = "radar OFF — launch with --record-radar"
    else:
        msg = (f"radar plan  ±{RADAR_FOV_AZ_DEG:.0f}° az  ·  "
               f"current {n_now} pts  ·  5s trail {total_pts} "
               f"({len(trail)} frames)")
    cv2.putText(canvas, msg, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                TEXT, 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return jpg.tobytes() if ok else None


def _world_to_side_px(z: np.ndarray, y: np.ndarray) -> tuple:
    """Map world (Z forward, Y up) metres → side-elevation panel pixels.
    Z is horizontal (left → right = closer → farther); Y is vertical
    (top of panel = up). Sensor at left edge (Z=Z_min) bottom-ish (Y=0)."""
    zmin, zmax = TOPDOWN_Z_LIMITS
    ymin, ymax = RADAR_SIDE_Y_LIMITS
    px = ((z - zmin) / (zmax - zmin) * PANEL_W).astype(np.int32)
    py = (RADAR_SIDE_PANEL_H - (y - ymin) / (ymax - ymin)
          * RADAR_SIDE_PANEL_H).astype(np.int32)
    return px, py


def _render_radar_side_jpeg() -> Optional[bytes]:
    """Radar side-elevation panel (Y vs Z). Shows what the radar sees in
    the vertical plane — useful for confirming the chest cluster is at
    expected height (not picking up legs/head/ceiling)."""
    import math
    with _LOCK:
        trail = list(_state["radar_trail"])
        active = _state["radar_active"]
    H = RADAR_SIDE_PANEL_H
    canvas = np.full((H, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_W, H), PANEL, -1)
    # Grid: Z gridlines every 1m, Y gridlines every 0.5m.
    for m in range(int(TOPDOWN_Z_LIMITS[0]), int(TOPDOWN_Z_LIMITS[1]) + 1):
        x_px = int((m - TOPDOWN_Z_LIMITS[0]) /
                   (TOPDOWN_Z_LIMITS[1] - TOPDOWN_Z_LIMITS[0]) * PANEL_W)
        cv2.line(canvas, (x_px, 0), (x_px, H), GRID, 1)
        cv2.putText(canvas, f"{m}m", (x_px + 2, H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
    for y_m_int in range(-1, 2):
        y_m = float(y_m_int)
        if not (RADAR_SIDE_Y_LIMITS[0] <= y_m <= RADAR_SIDE_Y_LIMITS[1]):
            continue
        y_px = int(H - (y_m - RADAR_SIDE_Y_LIMITS[0]) /
                   (RADAR_SIDE_Y_LIMITS[1] - RADAR_SIDE_Y_LIMITS[0]) * H)
        cv2.line(canvas, (0, y_px), (PANEL_W, y_px), GRID, 1)
        cv2.putText(canvas, f"{y_m:+.1f}m", (4, y_px - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)
    # FOV ±60° elevation cone, from sensor at Z=Z_min, Y=0.
    sx_px, sy_px = _world_to_side_px(np.array([TOPDOWN_Z_LIMITS[0]]),
                                      np.array([0.0]))
    sx_px = int(sx_px[0]); sy_px = int(sy_px[0])
    rng = TOPDOWN_Z_LIMITS[1] - TOPDOWN_Z_LIMITS[0]
    tan_el = math.tan(math.radians(RADAR_FOV_EL_DEG))
    y_top    = +tan_el * rng
    y_bottom = -tan_el * rng
    xt_px, yt_px = _world_to_side_px(np.array([TOPDOWN_Z_LIMITS[1]]),
                                      np.array([y_top]))
    xb_px, yb_px = _world_to_side_px(np.array([TOPDOWN_Z_LIMITS[1]]),
                                      np.array([y_bottom]))
    cone = np.array([[sx_px, sy_px],
                     [int(xt_px[0]), int(yt_px[0])],
                     [int(xb_px[0]), int(yb_px[0])]], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [cone], (50, 30, 60))
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
    cv2.line(canvas, (sx_px, sy_px),
             (int(xt_px[0]), int(yt_px[0])), (100, 60, 130), 1, cv2.LINE_AA)
    cv2.line(canvas, (sx_px, sy_px),
             (int(xb_px[0]), int(yb_px[0])), (100, 60, 130), 1, cv2.LINE_AA)
    # Range arcs at 1, 2, 3 m (semicircles from sensor in Z-Y plane).
    for r in (1.0, 2.0, 3.0):
        arc_pts = []
        for el_deg in range(-int(RADAR_FOV_EL_DEG),
                            int(RADAR_FOV_EL_DEG) + 1, 5):
            el = math.radians(el_deg)
            y_w = r * math.sin(el)
            z_w = r * math.cos(el)
            if z_w < TOPDOWN_Z_LIMITS[0] or z_w > TOPDOWN_Z_LIMITS[1]:
                continue
            xp, yp = _world_to_side_px(np.array([z_w]), np.array([y_w]))
            arc_pts.append([int(xp[0]), int(yp[0])])
        if len(arc_pts) >= 2:
            cv2.polylines(canvas, [np.array(arc_pts, dtype=np.int32)],
                          False, (80, 50, 100), 1, cv2.LINE_AA)
    # Radar points: trail with fading.
    now = time.time()
    total_pts = 0
    for ts, pts in trail:
        if len(pts) == 0:
            continue
        age = max(0.0, now - ts)
        alpha = max(0.15, 1.0 - (age / 3.0) * 0.85)
        col = tuple(int(c * alpha) for c in (255, 60, 200))
        x_px, y_px = _world_to_side_px(pts[:, 2], pts[:, 1])
        in_view = ((x_px >= 0) & (x_px < PANEL_W)
                   & (y_px >= 0) & (y_px < H))
        for u, v in zip(x_px[in_view], y_px[in_view]):
            cv2.circle(canvas, (int(u), int(v)), 4, col, -1)
        total_pts += int(in_view.sum())
    # HUD.
    cv2.rectangle(canvas, (0, 0), (PANEL_W, 20), (0, 0, 0), -1)
    msg = (f"radar side  Y vs Z  ±{RADAR_FOV_EL_DEG:.0f}° el  ·  "
           f"trail {total_pts}" if active else
           "radar OFF — launch with --record-radar")
    cv2.putText(canvas, msg, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                TEXT, 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return jpg.tobytes() if ok else None


def _render_loop() -> None:
    global _latest_topdown_jpg, _latest_waveform_jpg
    global _latest_radar_topdown_jpg, _latest_radar_side_jpg
    period = 1.0 / RENDER_FPS
    while True:
        t0 = time.time()
        try:
            jpg = _render_topdown_jpeg()
            if jpg is not None:
                with _topdown_lock:
                    _latest_topdown_jpg = jpg
            jpg = _render_waveform_jpeg()
            if jpg is not None:
                with _waveform_lock:
                    _latest_waveform_jpg = jpg
            jpg = _render_radar_topdown_jpeg()
            if jpg is not None:
                with _radar_topdown_lock:
                    _latest_radar_topdown_jpg = jpg
            jpg = _render_radar_side_jpeg()
            if jpg is not None:
                with _radar_side_lock:
                    _latest_radar_side_jpg = jpg
        except Exception as e:
            print(f"[viewer] render error: {type(e).__name__}: {e}")
        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


def _make_stream_gen(lock, getter):
    def gen():
        while True:
            with lock:
                jpg = getter()
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1.0 / RENDER_FPS)
    return gen


def _rgb_stream_gen():
    last_yielded = None
    while True:
        with _rgb_jpg_lock:
            jpg = _latest_rgb_jpg
        if jpg is None or jpg is last_yielded:
            time.sleep(0.1)
            continue
        last_yielded = jpg
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.1)   # cap RGB stream at ~10 Hz upper bound


def _depth_stream_gen():
    last_yielded = None
    while True:
        with _depth_jpg_lock:
            jpg = _latest_depth_jpg
        if jpg is None or jpg is last_yielded:
            time.sleep(0.1)
            continue
        last_yielded = jpg
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.1)


PAGE = """<!doctype html>
<html><head><title>Knight Vision · Phase 1 viewer</title>
<style>
  *{box-sizing:border-box}
  body{background:#0a0a0a;color:#eee;font-family:system-ui;margin:0;padding:0;font-size:13px}
  img{display:block;background:#0a0a0a}

  /* ── Header (single row) ──────────────────────────────────────── */
  .header{display:flex;align-items:center;gap:18px;padding:8px 18px;
          border-bottom:1px solid #1d1d1d;background:#0d0d0d;flex-wrap:wrap}
  .brand{font-size:12px;letter-spacing:2px;color:#888;text-transform:uppercase;font-weight:500}
  .bpm{display:flex;align-items:baseline;gap:4px}
  .bpm .big{font-size:38px;font-weight:200;line-height:1;letter-spacing:-1px;color:#777;
            min-width:78px;text-align:right;transition:color .2s}
  .bpm .big.conf-HIGH{color:#2d7}
  .bpm .big.conf-MEDIUM{color:#fa3}
  .bpm .big.conf-LOW,.bpm .big.conf-NONE{color:#777}
  .bpm .unit{font-size:13px;color:#888}
  .conf-line{font-family:monospace;font-size:11px;color:#888;letter-spacing:1px}
  .conf-line .conf-HIGH{color:#2d7;font-weight:600}
  .conf-line .conf-MEDIUM{color:#fa3;font-weight:600}
  .conf-line .conf-LOW,.conf-line .conf-NONE{color:#777}
  .ctrls{display:flex;align-items:center;gap:6px;margin-left:auto}
  .ctrls button{background:#222;color:#eee;border:1px solid #333;border-radius:4px;
                padding:5px 12px;font-family:inherit;font-size:11px;cursor:pointer;
                letter-spacing:1px;text-transform:uppercase}
  .ctrls button:hover{background:#2a2a2a;border-color:#555}
  .ctrls button.rec.on{background:#a22;border-color:#c33;color:#fff}
  .ctrls .hint{font-size:9px;color:#555;letter-spacing:1px;margin-left:4px}
  .recbadge{display:inline-block;padding:2px 6px;background:#a22;color:#fff;
            font-size:9px;letter-spacing:2px;border-radius:3px}
  .recbadge.off{display:none}
  .meta-pill{font-family:monospace;font-size:10px;color:#888;letter-spacing:1px;
             padding:3px 8px;background:#161616;border-radius:3px}
  .meta-pill b{color:#bbb}
  .helpico{display:inline-block;width:14px;height:14px;line-height:14px;text-align:center;
           background:#222;color:#888;border-radius:50%;font-size:10px;cursor:help;
           position:relative}
  .helpico .tip{display:none;position:absolute;top:100%;right:0;margin-top:6px;
                width:240px;background:#161616;border:1px solid #2a2a2a;padding:8px 10px;
                border-radius:5px;text-align:left;font-family:system-ui;font-size:11px;
                color:#aaa;line-height:1.5;z-index:20;text-transform:none;letter-spacing:0}
  .helpico:hover .tip{display:block}

  /* ── Main grid: top-down LEFT, dense stack RIGHT ──────────────── */
  .main{display:grid;grid-template-columns:55fr 45fr;gap:10px;
        padding:10px 14px 14px;align-items:start;max-height:calc(100vh - 60px)}
  .topdown-panel{background:#161616;border-radius:6px;padding:6px;
                 align-self:stretch;display:flex;flex-direction:column}
  .topdown-panel img{width:100%;height:auto;border-radius:4px;max-height:100%}
  .right-stack{display:flex;flex-direction:column;gap:8px}
  .wave-panel{background:#161616;border-radius:6px;padding:6px}
  .wave-panel img{width:100%;height:auto;border-radius:4px}
  .metrics{background:#161616;border-radius:6px;padding:6px 12px;font-family:monospace}
  .metrics .row{display:flex;justify-content:space-between;align-items:center;
                font-size:11px;margin:2px 0;gap:8px;line-height:1.2}
  .metrics .row .k{color:#888}
  .metrics .row .v{color:#2d7;font-weight:600}
  .sparkline{height:18px;width:120px;display:block}

  /* Sub-strip inside right column: depth | rgb | gt all in one row */
  .substrip{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .thumb{background:#161616;border-radius:6px;padding:5px;display:flex;flex-direction:column}
  .thumb .lbl{font-family:monospace;font-size:9px;color:#888;letter-spacing:1px;
              text-transform:uppercase;margin-bottom:3px}
  .thumb img{width:100%;height:auto;border-radius:3px;background:#000;max-height:160px;object-fit:contain}
  .thumb.disabled .lbl{color:#444}
  .thumb.disabled .placeholder{font-family:monospace;font-size:9px;color:#444;
                                padding:18px 0;text-align:center;border:1px dashed #222;
                                border-radius:3px}
  .gt-thumb{background:#161616;border-radius:6px;padding:8px 10px;font-family:monospace}
  .gt-thumb .lbl{font-size:9px;color:#888;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
  .gt-thumb .vals{display:flex;gap:10px;margin-bottom:4px}
  .gt-thumb .vals .pair{flex:1}
  .gt-thumb .vals .pair .k{font-size:9px;color:#888;text-transform:uppercase;letter-spacing:1px}
  .gt-thumb .vals .pair .v{font-size:20px;font-weight:300;color:#9cf;line-height:1}
  .gt-thumb .vals .pair .src{font-size:8px;color:#666;margin-top:1px}
  .gt-thumb .delta{font-size:10px;color:#aaa;margin-bottom:4px}
  .gt-thumb .delta b{color:#fff}
  .gt-thumb input{background:#0a0a0a;border:1px solid #333;border-radius:3px;color:#eee;
                 padding:3px 5px;font-family:inherit;font-size:11px;width:54px}
  .gt-thumb input:focus{border-color:#9cf;outline:none}
  .gt-thumb .gt-form{display:flex;gap:4px;align-items:center;font-size:9px;color:#888;flex-wrap:wrap}
  .gt-thumb .gt-form button{background:#234;border:1px solid #356;border-radius:3px;color:#eee;
                            padding:3px 8px;font-family:inherit;font-size:10px;cursor:pointer}
  .gt-thumb .gt-form button:hover{background:#345}

  /* No-signal state in header */
  .nosignal{display:inline-block;padding:4px 10px;background:#3a1818;color:#f88;
            font-size:11px;letter-spacing:2px;border-radius:4px;font-weight:600;
            text-transform:uppercase;border:1px solid #642}
  .nosignal.hidden{display:none}
</style></head>
<body>
  <header class="header">
    <span class="brand">Knight Vision · Phase 1</span>
    <div class="bpm">
      <span class="big" id="bpm">—</span><span class="unit">BPM</span>
    </div>
    <span id="nosignal" class="nosignal hidden">⚠ No breathing detected</span>
    <span class="conf-line">
      <span id="conf" class="conf-NONE">awaiting</span>
      &nbsp;·&nbsp; SNR <span id="snr">—</span>
      &nbsp;·&nbsp; n=<span id="rrn">0</span>
    </span>
    <span id="recbadge" class="recbadge off">● REC</span>
    <div class="ctrls">
      <button id="btn_rec" class="rec" onclick="toggleRec()">Rec</button>
      <button id="btn_stop" onclick="stopSession()">Stop</button>
      <span class="hint">R / S</span>
    </div>
    <span class="meta-pill">BAND <b id="band">—</b></span>
    <span class="meta-pill"><b id="fps">—</b> fps</span>
  </header>

  <main class="main">
    <section class="topdown-panel">
      <img id="topdown_feed" src="/topdown_stream">
    </section>
    <aside class="right-stack">
      <div class="wave-panel">
        <img id="waveform_feed" src="/waveform_stream">
      </div>
      <div class="metrics">
        <div class="row"><span class="k">Frame</span><span class="v" id="frame">—</span></div>
        <div class="row"><span class="k">Residual pts</span><span class="v" id="resid">—</span></div>
        <div class="row"><span class="k">Clusters</span><span class="v" id="clusters">—</span></div>
        <div class="row"><span class="k">Centroid Z (m)</span><span class="v" id="cz">—</span></div>
        <div class="row">
          <span class="k">CZ trend</span>
          <svg class="sparkline" id="cz_spark" viewBox="0 0 120 18" preserveAspectRatio="none">
            <polyline fill="none" stroke="#2d7" stroke-width="1.2" points=""></polyline>
          </svg>
        </div>
        <div class="row" style="border-top:1px solid #222;padding-top:4px;margin-top:4px">
          <span class="k">Legend</span>
          <span class="helpico">?
            <span class="tip">
              Top-down view in shared frame: X right, Z forward.
              Subject points orange, residual cloud blue, centroid red ring.
              Waveform: detrended Z displacement (mm), last 10s.
            </span>
          </span>
        </div>
      </div>
      <div id="radar_card" class="thumb" style="padding:6px">
        <div class="lbl">Radar · top-down (X/Z, ±60° az cone)</div>
        <img id="radar_feed" src="/radar_stream"
             style="width:100%;height:auto;border-radius:3px;background:#000">
      </div>
      <div id="radar_side_card" class="thumb" style="padding:6px">
        <div class="lbl">Radar · side elevation (Y/Z, ±60° el cone)</div>
        <img id="radar_side_feed" src="/radar_side_stream"
             style="width:100%;height:auto;border-radius:3px;background:#000">
      </div>
      <div class="substrip">
        <div id="depth_card" class="thumb disabled">
          <div class="lbl">Depth · JET</div>
          <img id="depth_feed" style="display:none">
          <div class="placeholder">disabled · --depth</div>
        </div>
        <div id="rgb_card" class="thumb disabled">
          <div class="lbl">RGB</div>
          <img id="rgb_feed" style="display:none">
          <div class="placeholder">disabled · --rgb</div>
        </div>
      </div>
      <div class="gt-thumb">
        <div class="lbl">Ground truth (Polar / manual) · radar cross-check</div>
        <div class="vals">
          <div class="pair">
            <div class="k">HR (Polar)</div>
            <div class="v" id="gt_hr">—</div>
            <div class="src" id="gt_hr_src">&nbsp;</div>
          </div>
          <div class="pair">
            <div class="k">RR (Polar)</div>
            <div class="v" id="gt_rr">—</div>
            <div class="src" id="gt_rr_src">&nbsp;</div>
          </div>
          <div class="pair">
            <div class="k">RR (Radar)</div>
            <div class="v" id="radar_rr" style="color:#f6b">—</div>
            <div class="src" id="radar_rr_src">&nbsp;</div>
          </div>
        </div>
        <div class="delta">Δ LiDAR−Polar: <b id="gt_delta_v">—</b> &nbsp;·&nbsp; Δ LiDAR−Radar: <b id="radar_delta_v">—</b></div>
        <form class="gt-form" onsubmit="submitGT(event)">
          HR <input type="number" id="gt_hr_in" min="20" max="220" step="1">
          RR <input type="number" id="gt_rr_in" min="0" max="60" step="0.1">
          <button type="submit">Log</button>
        </form>
      </div>
    </aside>
  </main>
<script>
function drawSparkline(values){
  const poly = document.querySelector('#cz_spark polyline');
  if (!poly) return;
  const W = 140, H = 24;
  const valid = values.filter(v => v !== null);
  if (valid.length < 2) { poly.setAttribute('points',''); return; }
  const mn = Math.min(...valid), mx = Math.max(...valid);
  const span = Math.max(0.001, mx - mn);
  const n = values.length;
  let pts = [];
  for (let i = 0; i < n; i++) {
    if (values[i] === null) continue;
    const x = (i / (n - 1)) * (W - 2) + 1;
    const y = H - ((values[i] - mn) / span) * (H - 2) - 1;
    pts.push(x.toFixed(1) + ',' + y.toFixed(1));
  }
  poly.setAttribute('points', pts.join(' '));
}
async function refresh(){
  try{
    const j = await (await fetch('/state')).json();
    document.getElementById('frame').textContent = j.frame_no;
    document.getElementById('resid').textContent = j.n_residuals;
    document.getElementById('clusters').textContent = j.n_clusters;
    document.getElementById('cz').textContent = (j.cz === null) ? '—' : j.cz.toFixed(3);
    document.getElementById('rrn').textContent = j.rr_n;
    if (j.cz_spark && j.cz_spark.length) drawSparkline(j.cz_spark);

    // No-signal detection: cz_spark is mostly null (no subject) OR no RR yet.
    const spark = j.cz_spark || [];
    const validFrac = spark.length ? spark.filter(v => v !== null).length / spark.length : 0;
    const noSignal = (j.rr_bpm === null && j.frame_no > 30)
                     || (spark.length >= 20 && validFrac < 0.2);
    document.getElementById('nosignal').classList.toggle('hidden', !noSignal);

    if (j.rr_bpm !== null) {
      const bpm = document.getElementById('bpm');
      bpm.textContent = j.rr_bpm.toFixed(1);
      bpm.className = 'big conf-' + j.rr_conf;
      document.getElementById('snr').textContent = j.rr_snr.toFixed(1);
      const conf = document.getElementById('conf');
      conf.textContent = j.rr_conf;
      conf.className = 'conf-' + j.rr_conf;
    } else if (noSignal) {
      document.getElementById('bpm').textContent = '—';
      document.getElementById('bpm').className = 'big conf-NONE';
    }
    const rec = j.recording;
    document.getElementById('btn_rec').textContent = rec ? 'Stop Rec' : 'Rec';
    document.getElementById('btn_rec').classList.toggle('on', rec);
    document.getElementById('recbadge').classList.toggle('off', !rec);
    if (j.band_min_bpm !== null && j.band_max_bpm !== null) {
      document.getElementById('band').textContent =
        Math.round(j.band_min_bpm) + '–' + Math.round(j.band_max_bpm) + ' BPM';
    }
    document.getElementById('fps').textContent =
      (j.fps === null) ? '—' : j.fps.toFixed(1);
    const rgbCard = document.getElementById('rgb_card');
    if (j.rgb_active && rgbCard.classList.contains('disabled')) {
      rgbCard.classList.remove('disabled');
      const img = document.getElementById('rgb_feed');
      img.style.display = 'block';
      img.src = '/rgb_stream';
      const ph = rgbCard.querySelector('.placeholder');
      if (ph) ph.remove();
    }
    const depthCard = document.getElementById('depth_card');
    if (j.depth_active && depthCard.classList.contains('disabled')) {
      depthCard.classList.remove('disabled');
      const img = document.getElementById('depth_feed');
      img.style.display = 'block';
      img.src = '/depth_stream';
      const ph = depthCard.querySelector('.placeholder');
      if (ph) ph.remove();
    }
    const gt = j.gt;
    if (gt) {
      document.getElementById('gt_hr').textContent = (gt.hr === null) ? '—' : gt.hr.toFixed(0);
      document.getElementById('gt_rr').textContent = (gt.rr === null) ? '—' : gt.rr.toFixed(1);
      document.getElementById('gt_hr_src').textContent = (gt.hr === null) ? '' : gt.source;
      document.getElementById('gt_rr_src').textContent = (gt.rr === null) ? '' : gt.source;
      if (gt.rr !== null && j.rr_bpm !== null) {
        const d = j.rr_bpm - gt.rr;
        const sign = d >= 0 ? '+' : '';
        document.getElementById('gt_delta_v').textContent = sign + d.toFixed(1) + ' BPM';
      } else {
        document.getElementById('gt_delta_v').textContent = '—';
      }
    }
    // Radar BPM + Δ vs LiDAR
    const rrr = j.radar_rr_bpm;
    if (rrr !== null && rrr !== undefined) {
      document.getElementById('radar_rr').textContent = rrr.toFixed(1);
      document.getElementById('radar_rr_src').textContent =
        (j.radar_active ? `mean_z · n=${j.radar_rr_n}` : 'inactive');
      if (j.rr_bpm !== null) {
        const d = j.rr_bpm - rrr;
        const sign = d >= 0 ? '+' : '';
        document.getElementById('radar_delta_v').textContent = sign + d.toFixed(1) + ' BPM';
      }
    } else if (j.radar_active) {
      document.getElementById('radar_rr_src').textContent =
        `warming up · n=${j.radar_n||0}`;
    }
  }catch(e){}
}
async function submitGT(ev){
  ev.preventDefault();
  const hr = document.getElementById('gt_hr_in').value;
  const rr = document.getElementById('gt_rr_in').value;
  const body = {source: 'manual'};
  if (hr !== '') body.hr = parseFloat(hr);
  if (rr !== '') body.rr = parseFloat(rr);
  if (body.hr === undefined && body.rr === undefined) return;
  try{
    await fetch('/gt', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify(body)});
    document.getElementById('gt_hr_in').value = '';
    document.getElementById('gt_rr_in').value = '';
    refresh();
  }catch(e){}
}
async function toggleRec(){
  try{ await fetch('/toggle_record', {method:'POST'}); refresh(); }catch(e){}
}
async function stopSession(){
  if(!confirm('Stop the session? This will exit the pipeline.')) return;
  try{ await fetch('/stop_session', {method:'POST'}); }catch(e){}
}
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'r' || e.key === 'R') { toggleRec(); }
  else if (e.key === 's' || e.key === 'S') { stopSession(); }
});
setInterval(refresh, 250); refresh();
</script>
</body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/state")
def state():
    with _LOCK:
        cz_hist = _state["cz_history"]
        last_cz = cz_hist[-1] if cz_hist else None
        if last_cz is not None and (last_cz != last_cz):
            last_cz = None
        # Downsample cz_history to ~SPARKLINE_SAMPLES points for the
        # client-side mini-sparkline. NaN entries pass through as null.
        spark = []
        if cz_hist:
            arr = np.asarray(cz_hist, dtype=float)
            # Take the trailing window we care about (last 30 s ≈ 900 samples
            # at 30 Hz), then downsample.
            tail = arr[-900:] if arr.size > 900 else arr
            if tail.size <= SPARKLINE_SAMPLES:
                ds = tail
            else:
                idx = np.linspace(0, tail.size - 1, SPARKLINE_SAMPLES).astype(int)
                ds = tail[idx]
            spark = [None if (x != x) else float(x) for x in ds]
        return jsonify({
            "frame_no": _state["frame_no"],
            "n_residuals": int(len(_state["residuals"])),
            "n_clusters": _state["n_clusters"],
            "cz": last_cz,
            "cz_history_len": len(cz_hist),
            "rr_bpm": _state["rr_bpm"],
            "rr_snr": _state["rr_snr"],
            "rr_conf": _state["rr_conf"],
            "rr_n": _state["rr_n"],
            "recording": _control["recording"],
            "stop_requested": _control["stop_session"],
            "record_seq": _control["record_seq"],
            "rgb_active": _rgb_active,
            "depth_active": _depth_active,
            "radar_active":   _state["radar_active"],
            "radar_n":        _state["radar_n"],
            "radar_rr_bpm":   _state["radar_rr_bpm"],
            "radar_rr_snr":   _state["radar_rr_snr"],
            "radar_rr_conf":  _state["radar_rr_conf"],
            "radar_rr_n":     _state["radar_rr_n"],
            "gt": get_gt_current(),
            "band_min_bpm": _meta["band_min_bpm"],
            "band_max_bpm": _meta["band_max_bpm"],
            "fps": _compute_fps(),
            "cz_spark": spark,
        })


def _compute_fps() -> Optional[float]:
    """Caller must hold _LOCK."""
    ts = _state["frame_ts"]
    if len(ts) < 2:
        return None
    span = ts[-1] - ts[0]
    if span <= 0:
        return None
    return (len(ts) - 1) / span


@app.route("/gt", methods=["POST"])
def post_gt_endpoint():
    data = request.get_json(silent=True) or {}
    hr = data.get("hr")
    rr = data.get("rr")
    src = data.get("source", "manual")
    if hr is None and rr is None:
        return jsonify({"error": "need at least one of hr / rr"}), 400
    entry = post_gt(hr=hr, rr=rr, source=src)
    return jsonify(entry)


@app.route("/toggle_record", methods=["POST"])
def toggle_record():
    with _LOCK:
        prev = _control["recording"]
        _control["recording"] = not prev
        if not prev:
            _control["record_seq"] += 1
        return jsonify({"recording": _control["recording"],
                        "record_seq": _control["record_seq"]})


@app.route("/stop_session", methods=["POST"])
def stop_session():
    with _LOCK:
        _control["stop_session"] = True
        # Auto-stop recording so finalize() runs.
        _control["recording"] = False
        return jsonify({"stop_session": True})


@app.route("/topdown_stream")
def topdown_stream():
    gen = _make_stream_gen(_topdown_lock, lambda: _latest_topdown_jpg)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/waveform_stream")
def waveform_stream():
    gen = _make_stream_gen(_waveform_lock, lambda: _latest_waveform_jpg)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/rgb_stream")
def rgb_stream():
    return Response(_rgb_stream_gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/radar_stream")
def radar_stream():
    gen = _make_stream_gen(_radar_topdown_lock,
                           lambda: _latest_radar_topdown_jpg)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/radar_side_stream")
def radar_side_stream():
    gen = _make_stream_gen(_radar_side_lock,
                           lambda: _latest_radar_side_jpg)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/depth_stream")
def depth_stream():
    return Response(_depth_stream_gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def start(port: int = 5005) -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_render_loop, daemon=True).start()
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                                threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    print(f"[viewer] http://0.0.0.0:{port}/")
