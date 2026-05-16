"""
phase1/drivers/radar_driver.py
================================
mmWave radar driver — TI IWR6843AOPEVM implementation.

Hardware target: TI IWR6843AOPEVM. Configured over the CLI UART
(``/dev/ttyUSB0`` @ 115200 baud) and streams TLV-framed point clouds
over the data UART (``/dev/ttyUSB1`` @ 921600 baud).

Reuses the parser and config-sender from the ``knightvision-hub`` repo:
  - ``scripts/radar_parser.py::FrameStream``    — TLV stream iterator
  - ``scripts/radar_send_config.py::send_command`` — line-by-line CLI

Coordinate frame — AXIS SWAP HAPPENS HERE
------------------------------------------
All Knight Vision drivers must return XYZ in the shared frame:
**X = right, Y = up, Z = forward (= subject distance), metres**.

IWR6843 native order is X = right, **Y = forward**, **Z = up**.
``_real_capture()`` therefore swaps the last two axes before returning::

    native (X, Yfwd, Zup) → shared (X, Zup, Yfwd)
    np indexing:           pts[:, [0, 2, 1]]

This matches what ``lidar_driver`` returns and what ``pipeline.py`` reads
as the centroid-z time series for RR FFT (Z = subject distance from sensor).
**Do not** remove or reorder the ``[0, 2, 1]`` reindex without updating
both other drivers and the FFT analyser to match.

Clutter-removal override
------------------------
The hub ships ``configs/iwr6843aop_default.cfg`` with
``clutterRemoval -1 1`` because the live viewers in the hub depend on
the in-sensor static-clutter filter. Phase 1's spatial-priors pipeline
does its own background subtraction and needs the **raw** point cloud,
so this driver reads the hub cfg, overrides that one line to
``clutterRemoval -1 0`` in memory, and sends the modified version. The
on-disk cfg is left untouched.

To revert to stub
-----------------
Flip ``STUB_MODE = True`` below.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Toggle here for stub vs real IWR6843AOP ─────────────────────────────────
STUB_MODE = False
# ────────────────────────────────────────────────────────────────────────────

_HUB_SCRIPTS = Path("~/knightvision-hub/scripts").expanduser()
_HUB_CFG     = Path("~/knightvision-hub/configs/iwr6843aop_default.cfg").expanduser()

_DEFAULT_CLI_PORT  = "/dev/ttyUSB0"
_DEFAULT_CLI_BAUD  = 115200
_DEFAULT_DATA_PORT = "/dev/ttyUSB1"
_DEFAULT_DATA_BAUD = 921600

# Synthetic breathing parameters (stub only)
_BREATHING_FREQ_HZ = 0.25
_BREATHING_AMP_M   = 0.020
_PERSON_CENTRE     = np.array([0.0, 0.3, 1.0])


def _import_hub_modules():
    """Lazy-import FrameStream and send_command from the hub scripts dir."""
    if str(_HUB_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_HUB_SCRIPTS))
    from radar_parser import FrameStream            # type: ignore
    from radar_send_config import send_command      # type: ignore
    return FrameStream, send_command


def _is_benign_idle_failure(cmd: str, resp: str) -> bool:
    """True if a failed sensorStop/flushCfg response is just 'already idle'."""
    head = cmd.split()[0] if cmd.split() else ""
    if head not in ("sensorStop", "flushCfg"):
        return False
    text = resp.lower() if resp else ""
    return "ignored" in text or "already stopped" in text or "not running" in text


def _load_cfg_lines_with_clutter_off(cfg_path: Path) -> List[str]:
    """Read hub cfg, drop comments/blanks, override clutterRemoval line."""
    out: List[str] = []
    for raw in cfg_path.read_text().splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("%") or ln.startswith("#"):
            continue
        tokens = ln.split()
        if tokens and tokens[0] == "clutterRemoval":
            out.append("clutterRemoval -1 0")
        else:
            out.append(ln)
    return out


class RadarDriver:
    """IWR6843AOP point-cloud driver (or deterministic stub)."""

    def __init__(self, config) -> None:
        self.config = config
        self._live_mode: bool = False
        self._frame_counter: int = 0
        self._bg_base = self._build_background_base()
        self._stream = None  # FrameStream, opened on first real capture

    # ── Public API ────────────────────────────────────────────────────────

    def set_live_mode(self, live: bool) -> None:
        """No-op for real hardware (sensor streams continuously)."""
        self._live_mode = live
        if live and STUB_MODE:
            self._frame_counter = 0

    def capture_frame(self) -> np.ndarray:
        """Capture one radar frame as an XYZ point cloud.

        Returns
        -------
        np.ndarray
            Shape (N, 3), dtype float64. XYZ in metres, shared frame
            (X right, Y up, Z forward). Empty (0, 3) if the frame
            contained no detected points.
        """
        if STUB_MODE:
            return self._synthetic_frame()
        return self._real_capture()

    def capture_seconds(self, seconds: float) -> List[np.ndarray]:
        n_frames = max(1, int(seconds * self.config.radar_fps))
        frames: List[np.ndarray] = []
        for _ in range(n_frames):
            frames.append(self.capture_frame())
            if STUB_MODE:
                pass
            else:
                # FrameStream blocks on the next sensor frame; no sleep needed.
                pass
        return frames

    def close(self) -> None:
        """Stop the radar (sensorStop over CLI) and close the data stream."""
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

            # Best-effort sensorStop so we leave the radar idle.
            try:
                _, send_command = _import_hub_modules()
                import serial
                cli_port = getattr(self.config, "radar_cli_port", _DEFAULT_CLI_PORT)
                cli_baud = getattr(self.config, "radar_cli_baud", _DEFAULT_CLI_BAUD)
                cli = serial.Serial(cli_port, cli_baud, timeout=2)
                time.sleep(0.2)
                cli.reset_input_buffer()
                send_command(cli, "sensorStop", timeout=2.0)
                cli.close()
                print("[Radar] sensorStop sent; sensor idle.")
            except Exception as e:
                print(f"[Radar] WARNING: sensorStop failed ({type(e).__name__}: {e})")

    # ── Real hardware ─────────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        """Send the modified cfg over CLI and open the data stream."""
        if self._stream is not None:
            return

        FrameStream, send_command = _import_hub_modules()

        cli_port  = getattr(self.config, "radar_cli_port",  _DEFAULT_CLI_PORT)
        cli_baud  = getattr(self.config, "radar_cli_baud",  _DEFAULT_CLI_BAUD)
        data_port = getattr(self.config, "radar_data_port", _DEFAULT_DATA_PORT)
        data_baud = getattr(self.config, "radar_data_baud", _DEFAULT_DATA_BAUD)

        import serial
        cmds = _load_cfg_lines_with_clutter_off(_HUB_CFG)
        print(f"[Radar] sending {len(cmds)} cfg lines (clutterRemoval forced off) "
              f"to {cli_port} @ {cli_baud}")

        cli = serial.Serial(cli_port, cli_baud, timeout=2)
        time.sleep(0.5)
        cli.reset_input_buffer()
        try:
            failures = 0
            for cmd in cmds:
                timeout = 5.0 if cmd.startswith("sensorStart") else 2.0
                ok, resp = send_command(cli, cmd, timeout=timeout)
                if not ok and _is_benign_idle_failure(cmd, resp):
                    # sensorStop / flushCfg return "Ignored" when the sensor
                    # is already in the desired state — harmless prelude noise.
                    continue
                if not ok:
                    failures += 1
                    print(f"[Radar] cfg line failed: {cmd}")
                time.sleep(0.05)
            if failures:
                print(f"[Radar] WARNING: {failures} cfg lines failed; sensor may "
                      f"not be streaming.")
        finally:
            cli.close()

        self._stream = FrameStream(port=data_port, baud=data_baud)
        print(f"[Radar] streaming on {data_port} @ {data_baud}")

    def _real_capture(self) -> np.ndarray:
        self._ensure_started()
        try:
            frame = next(self._stream)  # type: ignore[arg-type]
        except StopIteration:
            return np.empty((0, 3), dtype=np.float64)

        pts = frame.get("points")
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)

        # AXIS SWAP — see module docstring. Native (X, Yfwd, Zup) → shared
        # (X, Zup, Yfwd). The [0, 2, 1] reindex puts "up" in column 1 and
        # "forward" (subject distance) in column 2, matching lidar_driver
        # and what pipeline.py treats as the centroid-z series for RR FFT.
        xyz = pts[:, :3].astype(np.float64)
        return xyz[:, [0, 2, 1]]

    # ── Stub synthesis ────────────────────────────────────────────────────

    @staticmethod
    def _build_background_base() -> np.ndarray:
        xs = np.arange(-2.0, 2.05, 0.6)
        ys = np.arange(-2.0, 2.05, 0.6)
        xx, yy = np.meshgrid(xs, ys)
        floor = np.column_stack([xx.ravel(), yy.ravel(),
                                  np.zeros(xx.size)])
        return floor.astype(float)

    def _synthetic_frame(self) -> np.ndarray:
        rng = np.random.default_rng()
        bg = self._bg_base + rng.normal(0.0, 0.005, self._bg_base.shape)
        if not self._live_mode:
            return bg
        t = self._frame_counter / self.config.radar_fps
        self._frame_counter += 1
        dz = _BREATHING_AMP_M * np.sin(2.0 * np.pi * _BREATHING_FREQ_HZ * t)
        centre = _PERSON_CENTRE + np.array([0.0, 0.0, dz])
        person = rng.normal(centre, [0.12, 0.06, 0.18], (40, 3))
        return np.vstack([bg, person])
