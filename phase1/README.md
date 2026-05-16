# Knight Vision — Phase 1: Spatial-Prior Respiratory Monitoring

Geometry-first, hardware-stubbed Python scaffold for non-contact respiratory
rate estimation using LiDAR, mmWave radar, and RGB camera.

---

## Architecture

```
LiDAR / Radar / Camera
        │
        ▼
 BackgroundModel.subtract()    ← pre-scan voxel statistics; no ML in hot path
        │
        ▼
 cluster_residuals()            ← DBSCAN on residual point cloud only
        │
        ▼
 select_subject_cluster()       ← largest cluster inside monitoring volume
        │
        ▼
 extract_rr()                   ← centroid-Z → Hanning FFT → peak in RR band
        │
        ▼
 [HH:MM:SS] RR: 15.2 BPM | SNR: 48.3 | Conf: HIGH | Clusters: 2 | Subject pts: 120
```

---

## Module guide

| File | Purpose |
|---|---|
| `config.py` | `KVConfig` dataclass — all tunable parameters |
| `drivers/lidar_driver.py` | LiDAR stub (flip `STUB_MODE = False` for hardware) |
| `drivers/radar_driver.py` | mmWave radar stub |
| `drivers/camera_driver.py` | RGB camera stub (wraps OpenCV) |
| `background.py` | `BackgroundModel` — voxel-grid background subtraction |
| `clustering.py` | DBSCAN clustering + subject-cluster selection |
| `respiratory.py` | `extract_rr()` — centroid-Z FFT → BPM + SNR + confidence |
| `pipeline.py` | `Phase1Pipeline` — orchestrates everything |
| `visualise.py` | Matplotlib spectrum / history plots; Open3D 3-D views |
| `run_phase1.py` | CLI entry point |

---

## Quick start

### Install dependencies

```bash
pip install -r phase1/requirements.txt
```

`open3d` and `opencv-python` are optional — core pipeline runs without them.

### Run the demo (no hardware required)

From the **Knight Vision** directory:

```bash
python phase1/run_phase1.py --mode demo
```

This:
1. Builds a background model from 5 s of synthetic empty-room frames.
2. Runs 30 s of synthetic live data with a sinusoidal breathing signal at
   **0.25 Hz (15 BPM)** injected into the point cloud.
3. Prints periodic RR estimates and a final summary.
4. Saves `phase1/output/rr_spectrum.png` and `phase1/output/rr_history.png`.

Expected output (exact values may vary slightly due to FFT bin resolution):

```
[HH:MM:SS] RR:  15.2 BPM | SNR:  48.3 | Conf: HIGH   | Clusters: 1 | Subject pts: 120
...
══════════════════════════════════════════════════════════
  DEMO RESULT SUMMARY
  Estimated RR :  15.23 BPM
  Target RR    :  15.00 BPM  (0.25 Hz synthetic signal)
  SNR          :  52.1
  Confidence   :  HIGH
══════════════════════════════════════════════════════════
✓ Pipeline validated — RR estimate within 0.2 BPM of 15 BPM target.
```

### Capture background (real hardware)

```bash
python phase1/run_phase1.py --mode background --duration 60 --lidar
```

### Live monitoring (real hardware)

```bash
python phase1/run_phase1.py --mode live --duration 120 --lidar
```

---

## Wiring in real hardware

### LiDAR

1. Open `phase1/drivers/lidar_driver.py`.
2. Set `STUB_MODE = False` (line 42).
3. Fill in `LidarDriver._real_capture()` using the Open3D sensor API or
   your sensor's SDK.  The method must return `np.ndarray` of shape `(N, 3)`,
   dtype `float64`, units **metres**.

### mmWave Radar (TI IWR6843AOP)

1. Open `phase1/drivers/radar_driver.py`.
2. Set `STUB_MODE = False`.
3. Fill in `RadarDriver._real_capture()`.  Parse the mmWave SDK UART TLV
   frame to extract detected XYZ point positions.

### RGB Camera

1. Open `phase1/drivers/camera_driver.py`.
2. Set `STUB_MODE = False`.
3. The driver will open `cv2.VideoCapture(device_index)` automatically.
   Pass `device_index=N` to `CameraDriver(config, device_index=N)` in
   `pipeline.py` if the camera is not device 0.

---

## Configuration

All tunable parameters are in `phase1/config.py` (`KVConfig` dataclass).
Key values:

| Parameter | Default | Notes |
|---|---|---|
| `voxel_size` | 0.02 m (2 cm) | Background voxel grid resolution |
| `background_sigma` | 2.0 | σ threshold for residual detection |
| `dbscan_eps` | 0.08 m | DBSCAN neighbourhood radius |
| `rr_freq_min/max` | 0.1 – 2.0 Hz | 6 – 120 BPM |
| `fft_window_length` | 256 samples | Input samples for FFT |
| `fft_zero_pad_factor` | 16 | Zero-pad multiplier for frequency interpolation |
| `monitoring_volume` | ±1.5 m XY, 0–2.5 m Z | Subject must be inside this box |

---

## Phase 2 integration

`Phase1Pipeline.run_from_recording(frames)` accepts a pre-captured list of
point-cloud arrays for offline replay — useful for validating algorithmic
changes against recorded sessions before going live.
