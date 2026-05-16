"""
phase1/drivers/garmin_driver.py
================================
Garmin Forerunner 965 reference sensor driver for Knight Vision Phase 1.

Connects to the watch via Bluetooth Low Energy (BLE), receives R-R interval
notifications from the Heart Rate Service, and derives respiratory rate via
respiratory sinus arrhythmia (RSA) — the same method Garmin uses internally.

Requirements
------------
    pip install bleak

The Jetson Orin Nano has built-in Bluetooth — no extra hardware needed.

Usage
-----
    # The watch must be in an active recorded activity (any type)
    driver = GarminDriver(config)
    driver.start()                          # connects in background thread
    ...
    rr_bpm, confidence = driver.get_rr()   # latest estimate
    driver.stop()

Or run standalone to verify connection:
    python -m phase1.drivers.garmin_driver

Watch setup
-----------
1. Start any activity on the Forerunner 965 (Indoor Run works well)
2. Run this script — it will scan for and connect to the watch automatically
3. Keep the watch within ~5 m of the Jetson during testing

Notes
-----
- R-R intervals are streamed as part of the standard BLE Heart Rate Service
  (UUID 0x180D), characteristic Heart Rate Measurement (UUID 0x2A37).
- The watch only broadcasts HRS data during an active recorded activity.
- RSA-derived RR is less accurate than chest-belt reference (Zephyr BioHarness)
  but sufficient for Phase 1 self-testing and algorithm development.
- For the formal adult volunteer study, use a clinically-validated reference
  device (e.g. Zephyr BioHarness 3.0) alongside or instead of this driver.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import struct
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── BLE UUIDs ─────────────────────────────────────────────────────────────────
HR_SERVICE_UUID      = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID  = "00002a37-0000-1000-8000-00805f9b34fb"
GARMIN_DEVICE_PREFIX = ("Forerunner", "Fenix", "Venu", "Vivoactive", "Garmin")

# ── RSA respiratory rate extraction ───────────────────────────────────────────
_RR_FREQ_MIN_HZ = 0.1    # 6 BPM
_RR_FREQ_MAX_HZ = 0.6    # 36 BPM  (wider than LiDAR band — adults at rest)
_MIN_RRI_COUNT  = 60     # need ~60 s of data for a reliable estimate


class GarminDriver:
    """Reference respiratory rate sensor using Garmin Forerunner 965 via BLE.

    Parameters
    ----------
    config : KVConfig
        Shared pipeline configuration.
    log_csv : bool
        If True, write timestamped R-R intervals to a CSV file in
        ``config.output_dir / garmin_rri_<timestamp>.csv``.
    device_name : str, optional
        Partial name match for the target device.  Defaults to "Polar"
        (matches "Polar H10 XXXXXXXX"). Pass "Forerunner" for the Garmin
        watch as fallback.
    """

    def __init__(self, config, log_csv: bool = True,
                 device_name: str = "Polar") -> None:
        self.config      = config
        self.log_csv     = log_csv
        self.device_name = device_name

        # R-R interval buffer (milliseconds) with timestamps (seconds)
        self._rri_buf:  Deque[float] = deque(maxlen=300)   # ~5 min at 60 BPM
        self._rri_ts:   Deque[float] = deque(maxlen=300)

        self._latest_rr:   Optional[float] = None
        self._latest_conf: str             = "NONE"
        self._hr_bpm:      Optional[float] = None

        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._csv_path: Optional[Path] = None
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start BLE connection in a background thread.

        Returns immediately.  Use ``get_rr()`` to poll for estimates.
        The watch must be in an active recorded activity before calling this.
        """
        if self._running:
            log.warning("GarminDriver already running.")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop,
                                         daemon=True, name="garmin-ble")
        self._thread.start()
        log.info("GarminDriver started — scanning for %s…", self.device_name)

    def stop(self) -> None:
        """Stop the BLE connection and background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("GarminDriver stopped.")

    def get_rr(self) -> Tuple[Optional[float], str]:
        """Return the latest respiratory rate estimate.

        Returns
        -------
        (rr_bpm, confidence) : (float | None, str)
            ``rr_bpm`` is None until enough R-R intervals have been collected.
            ``confidence`` is one of: "NONE", "LOW", "MEDIUM", "HIGH".
        """
        with self._lock:
            return self._latest_rr, self._latest_conf

    def get_hr(self) -> Optional[float]:
        """Return the latest heart rate in BPM."""
        with self._lock:
            return self._hr_bpm

    def get_rri_buffer(self) -> Tuple[List[float], List[float]]:
        """Return a copy of the R-R interval buffer for offline analysis.

        Returns
        -------
        (rri_ms, timestamps) : (list[float], list[float])
        """
        with self._lock:
            return list(self._rri_buf), list(self._rri_ts)

    def status_line(self) -> str:
        """Return a single-line status string for live display."""
        rr, conf = self.get_rr()
        hr = self.get_hr()
        n  = len(self._rri_buf)
        if rr is None:
            return f"[Garmin] Waiting… ({n}/{_MIN_RRI_COUNT} RRI collected)"
        return (f"[Garmin] RR: {rr:5.1f} BPM | Conf: {conf:<6s} | "
                f"HR: {hr:.0f} BPM | RRI buf: {n}")

    # ── BLE loop ───────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the background thread — runs the asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ble_main())
        except Exception as exc:
            log.error("GarminDriver BLE error: %s", exc)
        finally:
            loop.close()

    async def _ble_main(self) -> None:
        """Scan for the watch, connect, and stream HR measurements."""
        from bleak import BleakClient, BleakScanner

        # ── Discover device ───────────────────────────────────────────────
        log.info("Scanning for Garmin device (name contains '%s')…",
                 self.device_name)
        device = None
        while self._running and device is None:
            found = await BleakScanner.discover(timeout=5.0)
            for d in found:
                if d.name and self.device_name in d.name:
                    device = d
                    log.info("Found: %s  (%s)", d.name, d.address)
                    break
            if device is None:
                log.debug("Not found — retrying in 3 s "
                          "(is an activity recording on the watch?)")
                await asyncio.sleep(3)

        if device is None:
            log.error("Scan stopped before device found.")
            return

        # ── Setup CSV logger ──────────────────────────────────────────────
        csv_file = None
        csv_writer = None
        if self.log_csv:
            out_dir = getattr(self.config, "output_dir",
                              Path("phase1/output"))
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_path = Path(out_dir) / f"garmin_rri_{ts}.csv"
            # buffering=1 → line-buffered; flushes after each writerow.
            csv_file   = open(self._csv_path, "w", buffering=1, newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["wall_time_s", "rri_ms", "hr_bpm"])
            log.info("Logging R-R intervals to %s", self._csv_path)

        # ── Connect and subscribe ─────────────────────────────────────────
        first_packet_logged = [False]

        def on_hr_notify(_sender, data: bytearray) -> None:
            """Parse Heart Rate Measurement characteristic notification."""
            # Byte 0 = flags
            flags = data[0]

            # ── First-packet diagnostic: dump bytes + flag breakdown ──────
            if not first_packet_logged[0]:
                first_packet_logged[0] = True
                hex_dump = " ".join(f"{b:02x}" for b in data)
                rri_present = bool(flags & 0x10)
                logging.info(
                    "first HR packet  raw=[%s]  flags=0x%02x  "
                    "(HR_uint16=%d  EE=%d  RRI=%d  SC=%d)",
                    hex_dump, flags,
                    int(bool(flags & 0x01)),
                    int(bool(flags & 0x08)),
                    int(rri_present),
                    (flags >> 1) & 0x03,
                )
                if rri_present:
                    logging.info("R-R intervals present — RSA-based RR estimation is viable.")
                else:
                    logging.warning(
                        "R-R intervals NOT present in HR notifications. "
                        "Watch firmware/profile does not broadcast RRI; "
                        "RSA-based RR cannot be derived. Use a chest strap "
                        "(Polar H10, Wahoo Tickr) for ground-truth RR."
                    )
            # Bit 0: 0 = HR format uint8, 1 = uint16
            if flags & 0x01:
                hr_val = struct.unpack_from("<H", data, 1)[0]
                offset = 3
            else:
                hr_val = data[1]
                offset = 2

            # Bits 4-5: RR interval present if bit 4 set
            rri_values: List[float] = []
            if flags & 0x10:
                while offset + 1 < len(data):
                    rri_raw = struct.unpack_from("<H", data, offset)[0]
                    rri_ms  = rri_raw * 1000.0 / 1024.0   # 1/1024 s units → ms
                    rri_values.append(rri_ms)
                    offset += 2

            now = time.time()
            with self._lock:
                self._hr_bpm = float(hr_val)
                for rri in rri_values:
                    # Reject physiologically implausible values
                    if 300 < rri < 2000:
                        self._rri_buf.append(rri)
                        self._rri_ts.append(now)
                        if csv_writer:
                            csv_writer.writerow([f"{now:.3f}", f"{rri:.1f}",
                                                 f"{hr_val}"])
                # Recompute RR estimate when buffer has enough data
                if len(self._rri_buf) >= _MIN_RRI_COUNT:
                    self._latest_rr, self._latest_conf = \
                        _compute_rr_from_rri(list(self._rri_buf))

        # Outer retry loop — if notifications stall (Polar BLE bug we saw
        # 2026-05-16 where RRIs froze after ~15s), reconnect.
        STALL_TIMEOUT_S = 6.0
        attempt = 0
        try:
            while self._running:
                attempt += 1
                try:
                    async with BleakClient(device.address) as client:
                        log.info("Connected to %s%s", device.name,
                                 f" (attempt {attempt})" if attempt > 1 else "")
                        await client.start_notify(HR_MEASUREMENT_UUID, on_hr_notify)
                        loop_start = time.time()
                        while self._running:
                            await asyncio.sleep(1.0)
                            with self._lock:
                                last_rri_t = (self._rri_ts[-1]
                                              if self._rri_ts else loop_start)
                            silent_for = time.time() - last_rri_t
                            if silent_for > STALL_TIMEOUT_S:
                                log.warning("BLE stall (%.1fs since last RRI) "
                                            "— reconnecting", silent_for)
                                break
                        try:
                            await client.stop_notify(HR_MEASUREMENT_UUID)
                        except Exception:
                            pass
                except Exception as exc:
                    log.error("BLE connection error: %s — retrying in 2s", exc)
                    await asyncio.sleep(2.0)
        finally:
            if csv_file:
                csv_file.close()


# ── RSA respiratory rate computation ─────────────────────────────────────────

def _compute_rr_from_rri(rri_ms: List[float]) -> Tuple[float, str]:
    """Derive respiratory rate from R-R intervals using FFT (RSA method).

    Parameters
    ----------
    rri_ms : list of float
        R-R intervals in milliseconds, most recent last.

    Returns
    -------
    (rr_bpm, confidence) : (float, str)
    """
    rri = np.array(rri_ms, dtype=float)

    # Interpolate to uniform 4 Hz grid (standard for HRV)
    cumtime = np.cumsum(rri) / 1000.0   # cumulative time in seconds
    fs = 4.0
    t_uniform = np.arange(cumtime[0], cumtime[-1], 1.0 / fs)
    rri_interp = np.interp(t_uniform, cumtime, rri)

    # Detrend + Hanning window
    rri_detrended = rri_interp - np.mean(rri_interp)
    window = np.hanning(len(rri_detrended))
    rri_windowed = rri_detrended * window

    # Zero-padded FFT
    n_fft  = max(len(rri_windowed), 4096)
    freqs  = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    power  = np.abs(np.fft.rfft(rri_windowed, n=n_fft)) ** 2

    # Find peak in respiratory band
    mask = (freqs >= _RR_FREQ_MIN_HZ) & (freqs <= _RR_FREQ_MAX_HZ)
    if not mask.any():
        return 0.0, "NONE"

    band_power = power[mask]
    band_freqs = freqs[mask]
    peak_idx   = np.argmax(band_power)
    peak_freq  = band_freqs[peak_idx]
    rr_bpm     = peak_freq * 60.0

    # SNR in band
    snr = band_power[peak_idx] / (np.mean(band_power) + 1e-9)
    if snr >= 5.0:
        confidence = "HIGH"
    elif snr >= 2.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return float(rr_bpm), confidence


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Minimal config stub
    class _Cfg:
        output_dir = Path("phase1/output")

    name = sys.argv[1] if len(sys.argv) > 1 else "Polar"
    print(f"Knight Vision — BLE HR driver (looking for: {name!r})")
    print("Make sure the device is broadcasting (Polar: electrodes wet on skin;")
    print("Garmin: activity recording), then press Enter.")
    try:
        input()
    except EOFError:
        pass

    drv = GarminDriver(_Cfg(), log_csv=True, device_name=name)
    drv.start()

    print("Collecting R-R intervals — press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(2)
            print(drv.status_line())
    except KeyboardInterrupt:
        pass
    finally:
        drv.stop()
        rri, ts = drv.get_rri_buffer()
        if len(rri) >= _MIN_RRI_COUNT:
            rr, conf = _compute_rr_from_rri(rri)
            print(f"\nFinal estimate — RR: {rr:.1f} BPM  Confidence: {conf}")
            print(f"R-R intervals collected: {len(rri)}")
            if drv._csv_path:
                print(f"Data saved to: {drv._csv_path}")
        else:
            print(f"\nNot enough data ({len(rri)} intervals — need {_MIN_RRI_COUNT})")
            print("Was an activity recording on the watch?")
        sys.exit(0)
