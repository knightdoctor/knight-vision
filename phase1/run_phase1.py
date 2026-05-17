#!/usr/bin/env python3
"""
phase1/run_phase1.py
====================
Command-line entry point for the Knight Vision Phase 1 pipeline.

Modes
-----
  --mode background   Capture empty-room background and save model.
                      (Saved to phase1/data/background_model.npz, reusable
                      across live runs.)
  --mode live         Load background, run live respiratory monitoring.
  --mode demo         Full synthetic demo — no hardware required.

Examples
--------
    # 30-s background capture (room empty)
    phase1/run.sh phase1/run_phase1.py --mode background --duration 30 --lidar

    # 60-s live run with web viewer on port 5005
    phase1/run.sh phase1/run_phase1.py --mode live --duration 60 --lidar --viewer

    # Same, also save raw point clouds (~150-300 MB)
    phase1/run.sh phase1/run_phase1.py --mode live --duration 60 --lidar --viewer --save-raw

    # Synthetic demo with viewer
    phase1/run.sh phase1/run_phase1.py --mode demo --viewer
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so the package imports resolve.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from phase1.config import KVConfig
from phase1.pipeline import Phase1Pipeline
from phase1.respiratory import settled_median
from phase1.visualise import plot_rr_spectrum, plot_live_rr


# ── CLI ───────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_phase1",
        description="Knight Vision Phase 1 — spatial-prior respiratory monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["background", "live", "demo"], required=True)
    p.add_argument("--duration", type=float, default=None,
                   help="Duration in seconds (background or live).")
    p.add_argument("--lidar",  action="store_true")
    p.add_argument("--radar",  action="store_true")
    p.add_argument("--camera", action="store_true")

    # New orchestration flags (live + demo only)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Output dir for live/demo artifacts. "
                        "Default: phase1/runs/<YYYYmmdd_HHMMSS>/")
    p.add_argument("--save-raw", action="store_true",
                   help="Also save raw per-frame point clouds (~150-300 MB/min).")
    p.add_argument("--viewer", action="store_true",
                   help="Launch the web viewer on port 5005.")
    p.add_argument("--viewer-port", type=int, default=5005)
    p.add_argument("--preview", action="store_true",
                   help="Start in preview mode (no recording). Press R in viewer "
                        "to start/stop recording, S to stop session. Implies --viewer.")
    p.add_argument("--rgb", action="store_true",
                   help="Stream the colour camera to a viewer panel "
                        "(~2 Hz, downscaled). Implies --camera + --viewer.")
    p.add_argument("--depth", action="store_true",
                   help="Stream a JET-coloured depth heatmap (0-3 m) to a "
                        "viewer panel (~2 Hz). Implies --lidar + --viewer.")
    return p


# ── Orchestration shared by run_demo / run_live ───────────────────────────

class _Orchestrator:
    """Session-level state.

    A *session* runs from launch until the duration elapses or the user
    hits Stop Session. Within a session, recording can be toggled on/off;
    each recording window mints its own ``runs/<ts>/`` directory.

    --preview starts with recording OFF; user toggles via viewer.
    Without --preview, recording starts immediately for legacy behaviour.
    """

    def __init__(self, args, config, mode: str, fps_estimate: float) -> None:
        self.args = args
        self.config = config
        self.mode = mode
        self.fps_estimate = fps_estimate

        # Preview implies viewer (no other way to toggle recording).
        if args.preview and not args.viewer:
            args.viewer = True
            print("[run] --preview implies --viewer; viewer enabled")
        # RGB implies viewer + camera capture.
        if args.rgb:
            if not args.viewer:
                args.viewer = True
                print("[run] --rgb implies --viewer; viewer enabled")
            if not args.camera:
                args.camera = True

        self.preview_mode = bool(args.preview)
        self.rgb_enabled = bool(args.rgb)
        self.depth_enabled = bool(args.depth)
        self.save_raw = bool(args.save_raw)
        self._explicit_run_dir = (
            Path(args.run_dir).expanduser() if args.run_dir is not None else None
        )

        # Visual stream tuning — push to viewer roughly every Nth pipeline
        # frame. At lidar 30 fps actual that's ~5 Hz on the panels; cheap.
        self._rgb_every_n = 5
        self._rgb_max_w = 480
        self._depth_every_n = 5
        self._depth_max_w = 480

        # Viewer (must be up before recording starts so its state is reflected)
        if args.viewer:
            from phase1 import viewer
            viewer.reset()
            viewer.start(port=args.viewer_port)
            viewer.set_meta(
                band_min_bpm=config.rr_freq_min * 60.0,
                band_max_bpm=config.rr_freq_max * 60.0,
            )
            self._viewer = viewer
        else:
            self._viewer = None

        # Current recording state (None when not recording).
        self.run_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.log_fh = None
        self.raw_frames: list = []
        self.raw_residuals: list = []
        self.recording = False
        self._recordings_done: list = []   # paths of finalized runs in this session

        # Ground-truth bookkeeping.
        self._gt_csv_fh = None
        self._gt_csv_writer = None
        self._gt_idx = 0       # how many entries in viewer history we've consumed

        self.t_start = time.time()
        self.n_frames_seen = 0
        self._last_rr_pushed = 0   # how many entries of pipe._rr_history we've forwarded
        self.pipe = None           # set by run_live/run_demo right after construction

        # Legacy mode: start recording immediately.
        if not self.preview_mode:
            self._start_recording()
        else:
            print("[run] preview mode — press R in viewer to start recording")

    # ── Recording lifecycle ───────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self.recording:
            return
        # Reset per-recording buffers + counters.
        self.raw_frames = []
        self.raw_residuals = []
        self._last_rr_pushed = 0

        if self._explicit_run_dir is not None and not self._recordings_done:
            self.run_dir = self._explicit_run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = _HERE / "runs" / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "per_frame.log"
        self.log_fh = self.log_path.open("w", buffering=1)
        self.log_fh.write("# frame  pts  resid  clusters  cz_m\n")

        # Ground-truth log — header even if empty.
        gt_path = self.run_dir / "gt.csv"
        self._gt_csv_fh = gt_path.open("w", buffering=1, newline="")
        self._gt_csv_writer = csv.writer(self._gt_csv_fh)
        self._gt_csv_writer.writerow(["ts_iso", "source", "hr", "rr"])

        # Snap GT idx forward: anything posted before record start isn't
        # part of this recording — start fresh from "now".
        if self._viewer is not None:
            self._gt_idx = len(self._viewer.get_gt_history())

        self.recording = True
        if self._viewer is not None:
            self._viewer.set_recording(True)
        print(f"[run] ● recording → {self.run_dir}")

    def _stop_recording(self) -> None:
        if not self.recording:
            return
        # Snapshot pipe RR history end index so finalize plots only this window.
        result = self.pipe._rr_history[-1] if (self.pipe and self.pipe._rr_history) else None
        self._finalize_current(result)
        self.recording = False
        if self._viewer is not None:
            self._viewer.set_recording(False)
        print(f"[run] ■ recording stopped")

    def _sync_with_viewer(self) -> None:
        """Apply any viewer-side recording toggles (button or R key)."""
        if self._viewer is None:
            return
        ctrl = self._viewer.get_control()
        want_record = bool(ctrl["recording"])
        if want_record and not self.recording:
            self._start_recording()
        elif (not want_record) and self.recording:
            self._stop_recording()

    def should_stop(self) -> bool:
        if self._viewer is None:
            return False
        return bool(self._viewer.get_control()["stop_session"])

    # ── Forwarding helpers ────────────────────────────────────────────────

    def push_rr_to_viewer(self, pipe) -> None:
        """Forward any new RR estimates from pipe._rr_history into the viewer."""
        if self._viewer is None:
            return
        hist = pipe._rr_history
        while self._last_rr_pushed < len(hist):
            r = hist[self._last_rr_pushed]
            self._viewer.set_rr(r["rr_bpm"], r["snr"], r["confidence"])
            self._last_rr_pushed += 1

    def frame_callback(self, i: int, frame: np.ndarray, residuals: np.ndarray,
                       subject, n_clusters: int, cz: float) -> None:
        self.n_frames_seen = i + 1

        # Apply any pending viewer-side record toggle before persisting.
        self._sync_with_viewer()

        n_pts   = int(len(frame)) if frame is not None else 0
        n_resid = int(len(residuals)) if residuals is not None else 0
        cz_str  = f"{cz:.4f}" if not (cz != cz) else "nan"  # NaN check

        # Stdout always throttled to every 10th frame (preview + recording).
        if (i + 1) % 10 == 0 or i == 0:
            rec_tag = "●" if self.recording else " "
            print(f"{rec_tag} f={i+1:5d}  pts={n_pts:6d}  resid={n_resid:5d}  "
                  f"clusters={n_clusters:3d}  cz={cz_str}")

        # Persistence is gated on recording.
        if self.recording:
            line = f"{i+1:5d}  {n_pts:6d}  {n_resid:5d}  {n_clusters:3d}  {cz_str}"
            self.log_fh.write(line + "\n")
            if self.save_raw and frame is not None:
                self.raw_frames.append(frame.astype(np.float32))
            if residuals is not None:
                self.raw_residuals.append(residuals.astype(np.float32))
            self._drain_gt_to_csv()

        if self._viewer is not None:
            self._viewer.set_frame(i + 1, residuals, subject, n_clusters, cz)
            if self.pipe is not None:
                self.push_rr_to_viewer(self.pipe)
            if self.rgb_enabled and (i % self._rgb_every_n == 0):
                self._push_rgb_frame()
            if self.depth_enabled and (i % self._depth_every_n == 0):
                self._push_depth_frame()

    def _push_rgb_frame(self) -> None:
        """Grab a colour frame from the camera driver, downscale, JPEG-encode,
        and push to the viewer. Silently no-ops on any error to keep the
        main pipeline immune to camera glitches."""
        if self.pipe is None or self.pipe.camera is None or self._viewer is None:
            return
        try:
            import cv2
            frame = self.pipe.camera.capture_frame()
            if frame is None or frame.size == 0:
                return
            h, w = frame.shape[:2]
            if w > self._rgb_max_w:
                scale = self._rgb_max_w / w
                frame = cv2.resize(frame, (self._rgb_max_w, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self._viewer.set_rgb(jpg.tobytes())
        except Exception as e:
            if not getattr(self, "_rgb_warned", False):
                print(f"[rgb] capture failed: {type(e).__name__}: {e}")
                self._rgb_warned = True

    def _push_depth_frame(self) -> None:
        """Render the latest cached depth frame as a JET-coloured heatmap
        (0-3 m clipped) and push to the viewer."""
        if self.pipe is None or self._viewer is None:
            return
        try:
            import cv2
            from phase1.drivers._femto_session import get_session
            session = get_session(
                expected_lidar_fps=getattr(self.config, "lidar_fps", None)
            )
            frames = session.get_cached(timeout_ms=200)
            if frames is None:
                return
            depth = frames.get_depth_frame()
            if depth is None:
                return
            w, h = depth.get_width(), depth.get_height()
            arr = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w)
            # Clip 0-3000 mm → 0-255 for colormap, then mark invalid (=0) black.
            clipped = np.clip(arr, 0, 3000)
            norm = (clipped.astype(np.float32) * (255.0 / 3000.0)).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            heatmap[arr == 0] = (0, 0, 0)
            # Downscale for transport.
            if w > self._depth_max_w:
                scale = self._depth_max_w / w
                heatmap = cv2.resize(heatmap, (self._depth_max_w, int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self._viewer.set_depth(jpg.tobytes())
        except Exception as e:
            if not getattr(self, "_depth_warned", False):
                print(f"[depth] render failed: {type(e).__name__}: {e}")
                self._depth_warned = True

    def _drain_gt_to_csv(self) -> None:
        """Append any new viewer GT entries to gt.csv. Only writes when recording."""
        if not self.recording or self._gt_csv_writer is None or self._viewer is None:
            return
        hist = self._viewer.get_gt_history()
        while self._gt_idx < len(hist):
            e = hist[self._gt_idx]
            self._gt_csv_writer.writerow([
                e["ts_iso"], e["source"],
                "" if e["hr"] is None else f"{e['hr']:.2f}",
                "" if e["rr"] is None else f"{e['rr']:.2f}",
            ])
            self._gt_idx += 1

    def _finalize_current(self, result) -> None:
        """Save the in-flight recording window to disk."""
        if not self.recording or self.run_dir is None:
            return
        # Flush any final GT entries before closing.
        self._drain_gt_to_csv()
        try:
            if self.log_fh is not None:
                self.log_fh.close()
        except Exception:
            pass
        try:
            if self._gt_csv_fh is not None:
                self._gt_csv_fh.close()
        except Exception:
            pass
        self._gt_csv_fh = None
        self._gt_csv_writer = None

        pipe = self.pipe

        # Centroid-z series for this recording window.
        if pipe is not None:
            cz_arr = np.array(pipe._centroid_z, dtype=float)
            np.save(self.run_dir / "centroid_z.npy", cz_arr)

        if self.raw_residuals:
            res_dict = {f"f{idx:05d}": r for idx, r in enumerate(self.raw_residuals)}
            np.savez_compressed(self.run_dir / "residuals.npz", **res_dict)
        if self.save_raw and self.raw_frames:
            raw_dict = {f"f{idx:05d}": f for idx, f in enumerate(self.raw_frames)}
            np.savez_compressed(self.run_dir / "frames.npz", **raw_dict)

        if result is not None:
            try:
                plot_rr_spectrum(result, self.config,
                                 save_path=self.run_dir / "rr_spectrum.png")
            except Exception as e:
                print(f"[run] spectrum plot failed: {e}")
        if pipe is not None and pipe._rr_history:
            try:
                plot_live_rr(pipe._rr_history,
                             save_path=self.run_dir / "rr_history.png")
            except Exception as e:
                print(f"[run] history plot failed: {e}")

        # Measured loop fps (the value actually used by the FFT).
        measured_fps = getattr(pipe, "_measured_fps", None) if pipe is not None else None

        meta = {
            "mode":          self.mode,
            "cmd":           " ".join(sys.argv),
            "started_at":    datetime.fromtimestamp(self.t_start).isoformat(),
            "ended_at":      datetime.now().isoformat(),
            "wall_seconds":  round(time.time() - self.t_start, 2),
            "frames_seen":   self.n_frames_seen,
            "fps_configured": self.fps_estimate,
            "fps_measured":   round(measured_fps, 3) if measured_fps else None,
            "config":        _config_as_dict(self.config),
            "sensors": {
                "lidar":  self.args.lidar  or (self.mode == "demo"),
                "radar":  self.args.radar,
                "camera": self.args.camera,
            },
            "save_raw":      self.save_raw,
            "viewer":        bool(self.args.viewer),
            "preview_mode":  self.preview_mode,
        }
        # Final estimate = settled-window median (NOT last-window). Pre-
        # settling transient windows are dropped per M1 requirement.
        if pipe is not None and pipe._rr_history:
            settled = settled_median(pipe._rr_history, self.config)
            meta["result_final"] = {
                "rr_bpm":       settled["rr_bpm"],
                "snr":          settled["snr"],
                "confidence":   settled["confidence"],
                "settled":      settled["settled"],
                "settle_note":  settled["settle_note"],
                "window_n":     settled["window_n"],
                "windows_total":settled["windows_total"],
                "fps_used":     float(result.get("fps_used", measured_fps or 0.0))
                                if result is not None else None,
            }
            # Per-window history kept for re-scoring / audit. Sliding-window
            # fields (window_samples, window_seconds, t_centre) are present
            # from PR J onwards — older runs won't have them.
            def _w_dict(w: dict) -> dict:
                out = {
                    "rr_bpm":     float(w["rr_bpm"]),
                    "snr":        float(w["snr"]),
                    "confidence": str(w["confidence"]),
                }
                for k in ("window_samples", "window_seconds", "t_centre",
                          "fps_used"):
                    if k in w:
                        out[k] = float(w[k]) if k != "window_samples" else int(w[k])
                return out
            meta["rr_windows"] = [_w_dict(w) for w in pipe._rr_history]
        else:
            meta["result_final"] = None
            meta["rr_windows"] = []

        with (self.run_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        print(f"[run] artifacts saved to {self.run_dir}")
        print(f"[run]   per_frame.log  ({self.log_path.stat().st_size} B)")
        if (self.run_dir / "centroid_z.npy").exists():
            print(f"[run]   centroid_z.npy ({(self.run_dir / 'centroid_z.npy').stat().st_size} B)")
        if self.raw_residuals:
            sz = (self.run_dir / "residuals.npz").stat().st_size
            print(f"[run]   residuals.npz  ({sz} B)")
        if self.save_raw and self.raw_frames:
            sz = (self.run_dir / "frames.npz").stat().st_size
            print(f"[run]   frames.npz     ({sz} B)")

        self._recordings_done.append(self.run_dir)
        self.log_fh = None
        self.run_dir = None
        self.log_path = None

    def finalize(self, pipe: Phase1Pipeline, result) -> None:
        """Session-end finalize. If a recording is still active, save it."""
        if self.recording:
            self._stop_recording()
        if self.preview_mode and not self._recordings_done:
            print("[run] session ended with no recordings saved")


def _config_as_dict(cfg: KVConfig) -> dict:
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, np.ndarray):
            out[f.name] = v.tolist()
        elif isinstance(v, Path):
            out[f.name] = str(v)
        else:
            out[f.name] = v
    return out


# ── Modes ─────────────────────────────────────────────────────────────────

def run_demo(config: KVConfig, args) -> None:
    print("=" * 60)
    print("  KNIGHT VISION — Phase 1 Demo (synthetic data)")
    print("=" * 60)

    # Demo means *no hardware*. Drivers default to STUB_MODE=False for
    # production; flip back on for this run only.
    from phase1.drivers import lidar_driver as _ld
    _ld.STUB_MODE = True

    # Demo writes its 5s synthetic bg to a separate file — must not
    # clobber the real captured background model that live mode uses.
    config.background_save_path = _HERE / "data" / "background_model_demo.npz"
    pipe = Phase1Pipeline(config, use_lidar=True)

    print("\n[1/2] Background capture (5 s synthetic) …")
    pipe.run_background_capture(seconds=5)

    orch = _Orchestrator(args, config, mode="demo", fps_estimate=config.lidar_fps)
    orch.pipe = pipe

    print("[2/2] Live monitoring (30 s synthetic, 0.25 Hz / 15 BPM target) …")
    result = pipe.run_live(
        duration_seconds=30, compute_rr_every=25,
        frame_callback=orch.frame_callback,
        should_stop=orch.should_stop,
        force_fps=config.lidar_fps,   # synthetic driver bakes against config rate
    )
    orch.finalize(pipe, result)

    if result is None or result["rr_bpm"] == 0.0:
        print("\nERROR: Pipeline did not produce an RR estimate.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  DEMO RESULT")
    print("=" * 60)
    print(f"  RR estimate :  {result['rr_bpm']:.2f} BPM   (target 15.00)")
    print(f"  SNR         :  {result['snr']:.1f}")
    print(f"  Confidence  :  {result['confidence']}")
    rr_error = abs(result["rr_bpm"] - 15.0)
    if rr_error > 3.0:
        print(f"  WARNING: deviates {rr_error:.1f} BPM from target.")
    else:
        print(f"  OK — within {rr_error:.1f} BPM of target.")
    print("=" * 60)


def run_background(config: KVConfig, args) -> None:
    """Background capture — saves to fixed path, no run dir."""
    use_lidar  = args.lidar  or not args.radar
    use_radar  = args.radar
    use_camera = args.camera
    duration   = args.duration if args.duration is not None else 30.0

    config.background_save_path = _HERE / "data" / "background_model.npz"
    pipe = Phase1Pipeline(config, use_lidar=use_lidar,
                          use_radar=use_radar, use_camera=use_camera)
    pipe.run_background_capture(seconds=duration)
    print(f"Background model saved to {config.background_save_path}.")
    print("Reusable across live runs — no need to recapture unless the room changes.")


def run_live(config: KVConfig, args) -> None:
    use_lidar  = args.lidar  or not args.radar
    use_radar  = args.radar
    use_camera = args.camera
    duration   = args.duration if args.duration is not None else 60.0

    config.background_save_path = _HERE / "data" / "background_model.npz"

    pipe = Phase1Pipeline(config, use_lidar=use_lidar,
                          use_radar=use_radar, use_camera=use_camera)

    fps = config.lidar_fps if use_lidar else config.radar_fps
    orch = _Orchestrator(args, config, mode="live", fps_estimate=fps)
    orch.pipe = pipe

    result = pipe.run_live(
        duration_seconds=duration,
        frame_callback=orch.frame_callback,
        should_stop=orch.should_stop,
    )
    orch.finalize(pipe, result)

    if result and orch._recordings_done:
        print(f"\nFinal estimate: RR = {result['rr_bpm']:.1f} BPM | "
              f"SNR = {result['snr']:.1f} | Confidence = {result['confidence']}")


# ── Entry point ───────────────────────────────────────────────────────────

def _resolve_flag_implications(args) -> None:
    """Apply flag implications BEFORE pipe construction so use_camera /
    use_viewer reflect --rgb / --preview correctly."""
    if args.rgb:
        args.camera = True
        args.viewer = True
    if args.depth:
        args.lidar = True
        args.viewer = True
    if args.preview:
        args.viewer = True


def main() -> None:
    args = _make_parser().parse_args()
    _resolve_flag_implications(args)
    config = KVConfig()
    if args.mode == "demo":
        run_demo(config, args)
    elif args.mode == "background":
        run_background(config, args)
    elif args.mode == "live":
        run_live(config, args)


if __name__ == "__main__":
    main()
