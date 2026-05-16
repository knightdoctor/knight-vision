import sys, csv, bisect
sys.path.insert(0, ".")
import numpy as np
from drivers.garmin_driver import _compute_rr_from_rri

rows = []
with open("phase1/output/garmin_rri_20260516_161737.csv") as f:
    for row in csv.DictReader(f):
        rows.append((float(row["wall_time_s"]), float(row["rri_ms"])))

times = [r[0] for r in rows]
rris  = [r[1] for r in rows]
t0 = times[0]

print("Time-walk RR estimate at different window sizes:")
print("t-rel     60-RRI     90-RRI    120-RRI    180-RRI")
print("------  --------  --------  --------  --------")
for sec in range(60, int(times[-1]-t0)+1, 20):
    t_now = t0 + sec
    n = bisect.bisect_right(times, t_now)
    parts = [f"{sec:4d}s "]
    for w in (60, 90, 120, 180):
        if n < w:
            parts.append("    --   ")
        else:
            recent = rris[n-w:n]
            rr, _ = _compute_rr_from_rri(recent)
            parts.append(f" {rr:6.1f}  ")
    print(" ".join(parts))
