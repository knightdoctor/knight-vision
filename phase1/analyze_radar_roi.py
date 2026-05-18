"""Same as analyze_radar_smoke but spatial-filter to ROI before stats."""
import sys
import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else "/tmp/radar_smoke.npz"

# ROI: chest-region around Phil's typical seated position (per earlier
# LiDAR runs). Phil at ~1m forward, chest height ~0 in shared Y frame.
X_LO, X_HI = -0.5, 0.5
Y_LO, Y_HI = -0.3, 0.5
Z_LO, Z_HI = 0.7, 1.8

d = np.load(NPZ)
ts = d["timestamps_rel"] if "timestamps_rel" in d.files else d["timestamps"]
keys = sorted(k for k in d.files if k.startswith("f"))

filt_means  = []
filt_counts = []
for k in keys:
    p = d[k]
    if len(p) == 0:
        filt_means.append(np.nan); filt_counts.append(0); continue
    m = ((p[:,0] >= X_LO) & (p[:,0] <= X_HI) &
         (p[:,1] >= Y_LO) & (p[:,1] <= Y_HI) &
         (p[:,2] >= Z_LO) & (p[:,2] <= Z_HI))
    if m.sum() == 0:
        filt_means.append(np.nan); filt_counts.append(0)
    else:
        filt_means.append(float(p[m, 2].mean()))
        filt_counts.append(int(m.sum()))

filt_means = np.array(filt_means)
filt_counts = np.array(filt_counts)
total_in_roi = (filt_counts > 0).sum()
print(f"frames: {len(keys)};  with >=1 point in ROI: {total_in_roi} "
      f"({100*total_in_roi/len(keys):.0f}%);  avg pts in ROI: "
      f"{filt_counts[filt_counts>0].mean():.1f}")

if total_in_roi < 10:
    print("ROI too sparse for FFT")
    sys.exit(0)

valid = ~np.isnan(filt_means)
s = filt_means.copy()
s[~valid] = np.nanmean(filt_means)
print(f"Z signal: mean {np.nanmean(filt_means):.3f}m, "
      f"span {np.nanmax(filt_means)-np.nanmin(filt_means):.3f}m, "
      f"std {np.nanstd(filt_means)*1000:.1f}mm")

fps = len(keys) / ts[-1]
s = s - s.mean()
s = s * np.hanning(len(s))
n_fft = max(len(s), 4096)
power = np.abs(np.fft.rfft(s, n=n_fft)) ** 2
freq = np.fft.rfftfreq(n_fft, d=1/fps)
bpm = freq * 60

in_band = (bpm >= 6) & (bpm <= 40)
b = bpm[in_band]; p = power[in_band]
peaks = [(b[i], p[i]) for i in range(1, len(p)-1)
         if p[i] > p[i-1] and p[i] > p[i+1]]
peaks.sort(key=lambda x: -x[1])
if not peaks:
    print("no peaks"); sys.exit(0)
mx = peaks[0][1]
print(f"\ntop peaks 6-40 BPM (chest-ROI Z mean signal):")
for v, pw in peaks[:5]:
    print(f"  {v:5.2f} BPM  rel-power {pw/mx:.3f}")
print(f"SNR (top peak / mean in band): {peaks[0][1] / np.mean(p):.2f}")
