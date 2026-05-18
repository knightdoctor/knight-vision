"""Quick analysis of radar smoke capture for periodic Z motion."""
import sys
import numpy as np

if len(sys.argv) > 1:
    NPZ = sys.argv[1]
else:
    NPZ = "/tmp/radar_smoke.npz"

d = np.load(NPZ)
ts = d["timestamps_rel"] if "timestamps_rel" in d.files else d["timestamps"]
frame_keys = sorted(k for k in d.files if k.startswith("f"))
print(f"frames: {len(frame_keys)}; duration: {ts[-1]:.1f}s; "
      f"fps_effective: {len(frame_keys)/ts[-1]:.2f}")

# Per-frame stats
pts_counts = [len(d[k]) for k in frame_keys]
print(f"points per frame: min={min(pts_counts)}, "
      f"max={max(pts_counts)}, median={int(np.median(pts_counts))}")

# Per-frame summaries of Z (depth)
mean_z   = []
median_z = []
min_z    = []
for k in frame_keys:
    pts = d[k]
    if len(pts) == 0:
        mean_z.append(np.nan); median_z.append(np.nan); min_z.append(np.nan)
    else:
        mean_z.append(float(pts[:, 2].mean()))
        median_z.append(float(np.median(pts[:, 2])))
        min_z.append(float(pts[:, 2].min()))

mean_z = np.array(mean_z); median_z = np.array(median_z); min_z = np.array(min_z)
print(f"\nZ-coord stats across recording:")
print(f"  mean_z: {np.nanmean(mean_z):.2f} ± {np.nanstd(mean_z):.2f} m "
      f"(range {np.nanmin(mean_z):.2f}..{np.nanmax(mean_z):.2f})")
print(f"  median_z: range {np.nanmin(median_z):.2f}..{np.nanmax(median_z):.2f}")
print(f"  min_z (nearest detection per frame): "
      f"{np.nanmean(min_z):.2f} ± {np.nanstd(min_z):.2f}")

# Resample to uniform grid for FFT
fps = len(frame_keys) / ts[-1]
n   = len(frame_keys)
print(f"\nFFT analysis (sample rate {fps:.2f} Hz, n={n}):")

for label, sig in [("mean_z", mean_z), ("median_z", median_z),
                   ("min_z", min_z)]:
    # NaN-fill then detrend
    s = sig.copy()
    if np.isnan(s).any():
        m = np.nanmean(s)
        s[np.isnan(s)] = m
    s = s - s.mean()
    s = s * np.hanning(len(s))
    n_fft = max(len(s), 4096)
    power = np.abs(np.fft.rfft(s, n=n_fft)) ** 2
    freq  = np.fft.rfftfreq(n_fft, d=1/fps)
    bpm   = freq * 60

    # Top 5 peaks in 6-40 BPM band (physiological respiratory)
    in_band = (bpm >= 6) & (bpm <= 40)
    b = bpm[in_band]; p = power[in_band]
    peaks = [(b[i], p[i]) for i in range(1, len(p)-1)
             if p[i] > p[i-1] and p[i] > p[i+1]]
    peaks.sort(key=lambda x: -x[1])
    mx = peaks[0][1] if peaks else 1.0
    print(f"\n  {label}: top peaks in 6-40 BPM:")
    for bpm_v, pw in peaks[:5]:
        print(f"    {bpm_v:5.2f} BPM  rel-power {pw/mx:.3f}")
    if peaks:
        snr = peaks[0][1] / np.mean(p)
        print(f"    SNR (top peak / mean in band): {snr:.2f}")
