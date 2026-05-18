"""Radar smoke capture — runs alongside a LiDAR paired session.

Captures the IWR6843 detected-points stream for a duration, saves to
.npz for offline analysis. Goal: see whether the sparse ~2 pts/frame
output shows any breathing-rate periodicity in Z (depth) coordinate.

If yes: the existing LiDAR pipeline can be extended to ingest radar
points as another centroid source. If no (likely): radar respiratory
needs a different stack (phase-on-chest-bin), per the apnoea arch doc.

Usage:
  phase1/run.sh radar_smoke_capture.py [--duration 60] [--out PATH]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.drivers.radar_driver import RadarDriver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="/tmp/radar_smoke.npz")
    args = ap.parse_args()

    cfg = KVConfig()
    drv = RadarDriver(cfg)
    # set_live_mode is a no-op for some drivers; safe to call if present.
    if hasattr(drv, "set_live_mode"):
        drv.set_live_mode(True)

    print(f"[radar-smoke] capturing for {args.duration:.0f}s "
          f"→ {args.out}")
    frames = []
    timestamps = []
    t0 = time.time()
    n_empty = 0
    n_nonempty = 0
    last_print = t0

    try:
        while time.time() - t0 < args.duration:
            pts = drv.capture_frame()
            if pts is None:
                pts = np.zeros((0, 3), dtype=np.float32)
            frames.append(pts.astype(np.float32))
            timestamps.append(time.time() - t0)
            if len(pts) == 0:
                n_empty += 1
            else:
                n_nonempty += 1
            now = time.time()
            if now - last_print > 5.0:
                print(f"  t={now-t0:5.1f}s  frames={len(frames)}  "
                      f"empty={n_empty}  nonempty={n_nonempty}  "
                      f"last_pts={len(pts)}")
                last_print = now
    finally:
        try:
            drv.close()
        except Exception as e:
            print(f"[radar-smoke] close warning: {e}")

    print(f"[radar-smoke] capture done. {len(frames)} frames, "
          f"{n_nonempty} with detections, {n_empty} empty.")

    # Save as npz: per-frame point clouds + timestamp array
    save_dict = {f"f{i:05d}": f for i, f in enumerate(frames)}
    save_dict["timestamps"] = np.array(timestamps, dtype=np.float32)
    np.savez_compressed(args.out, **save_dict)
    print(f"[radar-smoke] wrote {args.out}")


if __name__ == "__main__":
    main()
