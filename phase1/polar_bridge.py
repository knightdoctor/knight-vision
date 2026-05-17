"""Knight Vision · Polar H10 → viewer bridge (Mac edition).

Runs on the Mac (CoreBluetooth via bleak — stable, unlike Jetson bluez)
and POSTs HR + RR to the phase1 viewer's /gt endpoint on the Jetson.

RR is computed from R-R intervals using the **HF-band power-weighted
centroid method** (literature standard for HRV-derived respiration):

  - Resample RRIs to uniform 4 Hz grid
  - Detrend (linear) + Hanning window
  - FFT
  - Within the HF band (0.15-0.40 Hz = 9-24 BPM, the RSA frequencies),
    compute the power-weighted centroid frequency
  - Report centroid as the RR estimate

This is more stable than "pick the max peak" because the centroid
integrates power across the whole HF band — single noisy peaks (LF
baroreflex tails) don't capture the result.

Usage:
    venv/bin/python polar_bridge.py [--url URL] [--interval SEC]
                                    [--rr-window-s SEC] [--name PREFIX]
"""
import argparse
import asyncio
import collections
import csv
import json
import logging
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── BLE UUIDs (BLE Heart Rate Service, standard) ─────────────────────
HR_SERVICE_UUID     = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# ── RR computation tunables ──────────────────────────────────────────
HF_LO_HZ = 0.15   # 9 BPM   — RSA HF band lower bound
HF_HI_HZ = 0.40   # 24 BPM  — RSA HF band upper bound
INTERP_FS_HZ = 4.0
MIN_RRI_FOR_RR = 25     # need at least N RRIs in window before publishing RR

log = logging.getLogger("polar-bridge")


def compute_rr_hf_centroid(rri_ms_window: list) -> Optional[float]:
    """HF-band power-weighted centroid frequency → BPM.

    Returns None if there's not enough data or no power in the HF band.
    """
    if len(rri_ms_window) < MIN_RRI_FOR_RR:
        return None
    rri = np.asarray(rri_ms_window, dtype=float)
    # cumulative time (s)
    cumtime = np.cumsum(rri) / 1000.0
    if cumtime[-1] - cumtime[0] < 8.0:
        return None  # need at least ~8s of data for the lowest freq we want
    t = np.arange(cumtime[0], cumtime[-1], 1.0 / INTERP_FS_HZ)
    x = np.interp(t, cumtime, rri)
    # linear detrend (subtract best-fit line)
    coef = np.polyfit(np.arange(len(x)), x, 1)
    x = x - np.polyval(coef, np.arange(len(x)))
    x = x * np.hanning(len(x))
    n_fft = max(len(x), 4096)
    power = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=1.0 / INTERP_FS_HZ)
    mask = (freq >= HF_LO_HZ) & (freq <= HF_HI_HZ)
    if not mask.any():
        return None
    pf = power[mask]
    ff = freq[mask]
    total = pf.sum()
    if total <= 0:
        return None
    centroid_hz = float((ff * pf).sum() / total)
    return centroid_hz * 60.0


def compute_rr_peak_pick(rri_ms_window: list) -> Optional[float]:
    """Original max-peak method, kept for A/B comparison."""
    if len(rri_ms_window) < MIN_RRI_FOR_RR:
        return None
    rri = np.asarray(rri_ms_window, dtype=float)
    cumtime = np.cumsum(rri) / 1000.0
    if cumtime[-1] - cumtime[0] < 8.0:
        return None
    t = np.arange(cumtime[0], cumtime[-1], 1.0 / INTERP_FS_HZ)
    x = np.interp(t, cumtime, rri)
    x = x - x.mean()
    x = x * np.hanning(len(x))
    n_fft = max(len(x), 4096)
    power = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=1.0 / INTERP_FS_HZ)
    mask = (freq >= HF_LO_HZ) & (freq <= HF_HI_HZ)
    if not mask.any():
        return None
    return float(freq[mask][np.argmax(power[mask])] * 60.0)


# ── Polar H10 BLE driver ─────────────────────────────────────────────

class PolarH10:
    """Connect to Polar H10 (or any BLE HR device with RR-Interval support).

    Uses the BLE standard Heart Rate Service / HR Measurement char.
    Stores RRIs in a rolling buffer; caller polls via get_recent_rri().
    """

    def __init__(self, name_prefix: str = "Polar",
                 rri_buffer_size: int = 600,
                 csv_path: Optional[Path] = None):
        self.name_prefix = name_prefix
        self.csv_path = csv_path
        self._rri = collections.deque(maxlen=rri_buffer_size)   # ms
        self._rri_ts = collections.deque(maxlen=rri_buffer_size)
        self._hr_bpm: Optional[float] = None
        self._csv_fh = None
        self._csv_w = None
        self._connected = False
        self._first_packet_logged = False

    def _on_hr_notify(self, _sender, data: bytearray) -> None:
        flags = data[0]
        if not self._first_packet_logged:
            self._first_packet_logged = True
            rri_present = bool(flags & 0x10)
            log.info("first HR packet  raw=[%s]  flags=0x%02x  RRI=%s",
                     " ".join(f"{b:02x}" for b in data), flags,
                     "YES" if rri_present else "NO (cannot compute RR)")
        # HR field
        if flags & 0x01:
            hr_val = struct.unpack_from("<H", data, 1)[0]
            offset = 3
        else:
            hr_val = data[1]
            offset = 2
        self._hr_bpm = float(hr_val)
        # Skip Energy Expended (bit 3) if present
        if flags & 0x08:
            offset += 2
        # RR-Intervals (bit 4)
        if flags & 0x10:
            now = time.time()
            while offset + 1 < len(data):
                rri_raw = struct.unpack_from("<H", data, offset)[0]
                offset += 2
                rri_ms = rri_raw * 1000.0 / 1024.0   # 1/1024 s units
                if 300 < rri_ms < 2000:
                    self._rri.append(rri_ms)
                    self._rri_ts.append(now)
                    if self._csv_w:
                        self._csv_w.writerow([f"{now:.3f}", f"{rri_ms:.1f}",
                                              f"{hr_val}"])

    def get_hr(self) -> Optional[float]:
        return self._hr_bpm

    def get_recent_rri_by_seconds(self, seconds: float) -> list:
        """RRIs from the last `seconds` of wall time."""
        if not self._rri:
            return []
        cutoff = time.time() - seconds
        return [r for r, t in zip(self._rri, self._rri_ts) if t >= cutoff]

    def get_all_rri(self) -> list:
        return list(self._rri)

    async def run(self) -> None:
        from bleak import BleakClient, BleakScanner
        # Open CSV
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(self.csv_path, "w", buffering=1, newline="")
            self._csv_w = csv.writer(self._csv_fh)
            self._csv_w.writerow(["wall_time_s", "rri_ms", "hr_bpm"])
            log.info("logging RRIs to %s", self.csv_path)

        attempt = 0
        try:
            while True:
                attempt += 1
                log.info("scanning for '%s' (attempt %d)…",
                         self.name_prefix, attempt)
                devs = await BleakScanner.discover(timeout=8.0)
                target = next((d for d in devs
                               if d.name and self.name_prefix in d.name), None)
                if target is None:
                    log.warning("no device matching '%s' found; retrying",
                                self.name_prefix)
                    await asyncio.sleep(2)
                    continue
                log.info("connecting to %s (%s)", target.name, target.address)
                try:
                    async with BleakClient(target.address) as client:
                        log.info("connected")
                        await client.start_notify(HR_MEASUREMENT_UUID,
                                                  self._on_hr_notify)
                        self._connected = True
                        last_rri_count = len(self._rri)
                        last_progress_t = time.time()
                        while True:
                            await asyncio.sleep(2.0)
                            cur_count = len(self._rri)
                            if cur_count != last_rri_count:
                                last_rri_count = cur_count
                                last_progress_t = time.time()
                            elif time.time() - last_progress_t > 8.0:
                                log.warning("no new RRI for 8s — reconnecting")
                                break
                        try:
                            await client.stop_notify(HR_MEASUREMENT_UUID)
                        except Exception:
                            pass
                except Exception as e:
                    log.error("BLE error: %s", e)
                self._connected = False
                await asyncio.sleep(1.5)
        finally:
            if self._csv_fh:
                self._csv_fh.close()


# ── HTTP POST helper ─────────────────────────────────────────────────

def post_gt(url: str, hr: Optional[float], rr: Optional[float],
            source: str) -> bool:
    body: dict = {"source": source}
    if hr is not None and hr > 0:
        body["hr"] = float(hr)
    if rr is not None and rr > 0:
        body["rr"] = float(rr)
    if "hr" not in body and "rr" not in body:
        return False
    try:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        # OSError covers socket.timeout (raised as TimeoutError on Py3.10+
        # but socket.timeout/OSError on older). Either way, just drop the
        # post and try again next interval — don't kill the daemon.
        return False


# ── Daemon loop (POST to /gt every interval) ─────────────────────────

async def daemon_loop(polar: PolarH10, url: str, interval: float,
                      rr_window_s: float, method: str) -> None:
    compute = (compute_rr_hf_centroid if method == "hf-centroid"
               else compute_rr_peak_pick)
    last_post_ok = False
    log.info("posting every %.1fs  ·  RR window %.0fs  ·  method=%s",
             interval, rr_window_s, method)
    while True:
        await asyncio.sleep(interval)
        hr = polar.get_hr()
        rri_window = polar.get_recent_rri_by_seconds(rr_window_s)
        rr = compute(rri_window) if rri_window else None
        ok = post_gt(url, hr, rr, source="polar_h10")
        tag = "✓" if ok else "·"
        hr_s = f"{hr:5.1f}" if hr else "  --."
        rr_s = f"{rr:5.1f} BPM" if rr else "  --       "
        log.info("%s HR=%s  RR=%s  (RRI in %.0fs=%d)",
                 tag, hr_s, rr_s, rr_window_s, len(rri_window))
        if ok and not last_post_ok:
            log.info("viewer reachable")
        elif not ok and last_post_ok:
            log.warning("viewer unreachable")
        last_post_ok = ok


async def main_async(args) -> None:
    csv_path = Path(args.csv_dir) / f"polar_rri_{datetime.now():%Y%m%d_%H%M%S}.csv"
    polar = PolarH10(name_prefix=args.name, csv_path=csv_path)
    # Run BLE + daemon loop concurrently
    ble_task = asyncio.create_task(polar.run())
    daemon_task = asyncio.create_task(
        daemon_loop(polar, args.url, args.interval, args.rr_window_s, args.method)
    )
    try:
        await asyncio.gather(ble_task, daemon_task)
    except asyncio.CancelledError:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.168.1.90:5005/gt",
                   help="Jetson viewer /gt endpoint")
    p.add_argument("--interval", type=float, default=2.0,
                   help="POST interval seconds")
    p.add_argument("--rr-window-s", type=float, default=45.0,
                   help="Seconds of RRI history feeding each RR estimate "
                        "(longer = more stable but slower to react). "
                        "Literature standard 30-60s.")
    p.add_argument("--name", default="Polar")
    p.add_argument("--method", default="hf-centroid",
                   choices=("hf-centroid", "peak-pick"),
                   help="RR computation method. hf-centroid is the literature "
                        "standard for HRV→RR; peak-pick is the original.")
    p.add_argument("--csv-dir", default=str(Path.home() / "knight-vision-mac"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("stopping")


if __name__ == "__main__":
    main()
