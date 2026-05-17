"""Offline validation of PR M against today's runs."""
import sys, json
sys.path.insert(0, ".")
from phase1.config import KVConfig
from phase1.respiratory import settled_median

RUNS = [
    ("20260516_165647", "Run 4 (yesterday) — known good, LiDAR ~14.5, Polar 14.1"),
    ("20260516_170653", "Run 5 (yesterday) — breath hold, LiDAR ~14.6, Polar 14.1"),
    ("20260517_064421", "Today AM #1 — LiDAR 17.7 (OLD), Polar 13"),
    ("20260517_065958", "Today AM #2 — LiDAR 20.2 (OLD), Polar 11.8"),
]

cfg = KVConfig()
print(f"Config: min_snr={cfg.settled_median_min_snr}, "
      f"std<{cfg.settled_median_std_threshold_bpm}, "
      f"min_window={cfg.settled_median_min_window}\n")

for run, desc in RUNS:
    try:
        m = json.load(open(f"/home/phil/knight-vision/phase1/runs/{run}/meta.json"))
    except Exception as e:
        print(f"=== {run}: SKIPPED ({e}) ===\n")
        continue
    ws = m.get("rr_windows", [])
    old = m.get("result_final", {})
    new = settled_median(ws, cfg)
    print(f"=== {run}: {desc} ===")
    print(f"  OLD (trailing-only): {old.get('rr_bpm',0):5.2f} BPM "
          f"SNR {old.get('snr',0):5.2f} {old.get('confidence','?'):6s} "
          f"({old.get('settle_note','')})")
    print(f"  NEW (SNR-weighted):  {new['rr_bpm']:5.2f} BPM "
          f"SNR {new['snr']:5.2f} {new['confidence']:6s} "
          f"({new['settle_note']})")
    print()
