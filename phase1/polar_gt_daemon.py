"""Polar H10 → viewer /gt daemon.

Connects to the Polar H10 chest strap via BLE, computes RR via RSA, and
POSTs HR + RR to the viewer's /gt endpoint at a configurable cadence.
Runs alongside a live phase1 session.

Usage:
    phase1/run.sh polar_gt_daemon.py [--url http://127.0.0.1:5005/gt]
                                     [--interval 2.0]
                                     [--name Polar]
"""
import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.drivers.garmin_driver import GarminDriver, _compute_rr_from_rri


class _Cfg:
    output_dir = Path("phase1/output")


def post_gt(url: str, hr: float, rr: float, source: str) -> bool:
    body = {"source": source}
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
    except (urllib.error.URLError, ConnectionError):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5005/gt",
                   help="Viewer /gt endpoint")
    p.add_argument("--interval", type=float, default=2.0,
                   help="POST interval in seconds")
    p.add_argument("--name", default="Polar",
                   help="BLE device name prefix to match")
    p.add_argument("--min-rri", type=int, default=15,
                   help="Min RRI count before posting an RR estimate (default 15)")
    p.add_argument("--rr-window", type=int, default=60,
                   help="Number of MOST RECENT RRIs used for live RR estimate "
                        "(default 60 ≈ 1 min at HR=60). Smaller = more responsive "
                        "but noisier; bigger = stable but slow to react.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("polar-gt")

    drv = GarminDriver(_Cfg(), log_csv=True, device_name=args.name)
    drv.start()
    log.info("polar-gt daemon → %s (every %.1fs)", args.url, args.interval)

    last_post_ok = False
    try:
        while True:
            time.sleep(args.interval)
            with drv._lock:
                hr  = drv._hr_bpm
                rri = list(drv._rri_buf)
            rr = None
            if len(rri) >= args.min_rri:
                # Use only the trailing window so transient breathing
                # changes (breath hold, fast breathing) are visible.
                recent = rri[-args.rr_window:]
                rr, _ = _compute_rr_from_rri(recent)
            ok = post_gt(args.url, hr, rr, source="polar_h10")
            tag = "✓" if ok else "·"
            log.info("%s HR=%5.1f BPM  RR=%s  RRI=%d (last %d used)",
                     tag,
                     hr if hr else float("nan"),
                     f"{rr:5.1f} BPM" if rr else "—  ",
                     len(rri), args.rr_window)
            if ok and not last_post_ok:
                log.info("viewer reachable")
            elif not ok and last_post_ok:
                log.warning("viewer unreachable")
            last_post_ok = ok
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        drv.stop()


if __name__ == "__main__":
    main()
