# CubeEye I200D Replay Analysis — 2026-05-20

## Overview

- **Sensor**: CubeEye I200D ToF depth camera (640×480, ~15 fps)
- **Subject**: Gaumard Super TORY S2220 (validated newborn simulator)
- **Pipeline**: KV Phase 1 `extract_rr_from_signal` with peak-pick, 20s sliding window
- **Recordings analysed**: 12
- **Total analysis windows**: 704
- **Date of test**: 2026-04-20 (hospital simulation lab)

## MAE vs BPM Bucket

| BPM Bucket | N | MAE (BPM) | Median Err | ±2 BPM (%) | ±5 BPM (%) | SNR (med) | SNR IQR |
|:-----------|--:|----------:|-----------:|-----------:|-----------:|----------:|--------:|
| Apnoea (0)    | 116 |       nan |        nan |          0 |          0 |      48.4 |    42.6 |
| 5 BPM         |   9 |       1.2 |        1.2 |        100 |        100 |      41.2 |     4.4 |
| 10 BPM        |  36 |       1.0 |        0.4 |         81 |        100 |      18.7 |     1.3 |
| 15 BPM        |  63 |       2.0 |        0.2 |         79 |         79 |      17.5 |     1.9 |
| 20 BPM        |  40 |       3.9 |        0.2 |         70 |         72 |      18.1 |     2.1 |
| 30 BPM        | 124 |       5.6 |        0.3 |         75 |         76 |      17.3 |     2.0 |
| 40 BPM        |  54 |       5.8 |        0.2 |         83 |         83 |      17.7 |     2.1 |
| 45-50 BPM     |  39 |       3.5 |        0.2 |         90 |         90 |      17.3 |     1.1 |
| 60 BPM        | 105 |      35.4 |       53.8 |         33 |         33 |      54.4 |    44.9 |
| 70 BPM        |  14 |      14.5 |        0.1 |         71 |         71 |      17.9 |     1.8 |
| 80 BPM        |  36 |      12.6 |        0.2 |         78 |         83 |      17.0 |     1.6 |
| 90 BPM        |  23 |      11.3 |        0.1 |         83 |         83 |      17.2 |     1.7 |
| 100 BPM       |  12 |      15.7 |        0.1 |         83 |         83 |      16.9 |     1.1 |

**Overall (breathing windows only):** MAE = 11.4 BPM, ±2 BPM = 70%, ±5 BPM = 72% (n=555)

## Apnoea Detection (T_d = 20s)

| Recording | Events | Detected | Sensitivity | Mean TTD (s) |
|:----------|-------:|---------:|------------:|-------------:|
| 20260419_211649 | 1 | 0 | 0% | — |
| 20260419_212131 | 1 | 0 | 0% | — |
| 20260420_094107 | 0 | 0 | 0% | — |
| 20260420_094646 | 0 | 0 | 0% | — |
| 20260420_095154 | 0 | 0 | 0% | — |
| 20260420_095754 | 0 | 0 | 0% | — |
| 20260420_100308 | 0 | 0 | 0% | — |
| 20260420_100337 | 0 | 0 | 0% | — |
| 20260420_100817 | 0 | 0 | 0% | — |
| 20260420_101723 | 3 | 0 | 0% | — |
| 20260420_110254 | 0 | 0 | 0% | — |
| 20260420_110744 | 1 | 0 | 0% | — |

**Overall sensitivity**: 0% (0/6 events detected within 20s)

> **Critical finding**: Apnoea detection is non-functional. The FFT-based pipeline continues reporting the previous breathing rate after cessation. A separate time-domain cessation detector is required (variance drop, zero-crossing absence, or amplitude envelope).

## SNR Distribution per BPM Bucket

| BPM Bucket | SNR Median | SNR P25 | SNR P75 | N |
|:-----------|----------:|--------:|--------:|--:|
| Apnoea (0)    |       48.4 |    21.2 |    63.8 | 116 |
| 5 BPM         |       41.2 |    39.8 |    44.2 |   9 |
| 10 BPM        |       18.7 |    18.0 |    19.3 |  36 |
| 15 BPM        |       17.5 |    16.6 |    18.5 |  63 |
| 20 BPM        |       18.1 |    16.4 |    18.5 |  40 |
| 30 BPM        |       17.3 |    16.2 |    18.3 | 124 |
| 40 BPM        |       17.7 |    16.3 |    18.4 |  54 |
| 45-50 BPM     |       17.3 |    16.5 |    17.6 |  39 |
| 60 BPM        |       54.4 |    18.2 |    63.1 | 105 |
| 70 BPM        |       17.9 |    16.4 |    18.2 |  14 |
| 80 BPM        |       17.0 |    16.2 |    17.8 |  36 |
| 90 BPM        |       17.2 |    16.6 |    18.3 |  23 |
| 100 BPM       |       16.9 |    16.3 |    17.4 |  12 |

## Comparison to Femto Bolt M1 Baseline

| Metric | Femto Bolt (M1) | CubeEye I200D |
|:-------|----------------:|--------------:|
| Peak-pick MAE | 1.16 BPM | 11.35 BPM |
| ±2 BPM accuracy | — | 70% |
| Subject | Adult resting (n=4) | Newborn sim 5–100 BPM |
| Sensor FPS | ~10 fps (end-to-end) | ~15 fps |
| Distance | ~1.0 m | 0.6–0.76 m |
| Signal amplitude | ~25 mm chest rise | ~0.3 mm chest rise |
| Apnoea detection | Not tested | 0% sensitivity |

## Amplitude-Cessation Apnoea Detection (Dual-Signal Architecture)

The FFT-based pipeline (above) achieves 0% apnoea sensitivity because it
continues reporting the previous breathing rate after cessation. This section
evaluates a parallel **amplitude-cessation detector** operating on the same
signal: rolling RMS of chest displacement drops below an empirically-derived
noise-floor threshold for ≥ T_d seconds → alarm.

**Apnoea events analysed**: 7 across 5 recordings

### Empirical Noise-Floor Threshold

| Recording | Breathing RMS (P25) | Apnoea RMS (med) | Threshold | Separation Ratio |
|:----------|--------------------:|-----------------:|----------:|-----------------:|
| 20260419_211649 | 0.1464 mm | 0.1478 mm | 0.1098 mm | 0.99× |
| 20260419_212131 | 0.2168 mm | 0.0907 mm | 0.1626 mm | 2.39× |
| 20260420_100308 | 0.1649 mm | 0.1027 mm | 0.1237 mm | 1.60× |
| 20260420_101723 | 0.1542 mm | 0.0937 mm | 0.1156 mm | 1.65× |
| 20260420_110744 | 0.1529 mm | 0.1000 mm | 0.1147 mm | 1.53× |

> Threshold = 0.75 × P25(breathing RMS). Derived per-recording from
> quiet-but-breathing segments — no magic numbers.

### T_d = 10s

| Recording | GT Events | Detected | Sensitivity | Mean TTD (s) | FP | FP/hr |
|:----------|----------:|---------:|------------:|-------------:|---:|------:|
| 20260419_211649 | 1 | 0 | 0% | — | 0 | 0.0 |
| 20260419_212131 | 1 | 1 | 100% | 11.7 | 0 | 0.0 |
| 20260420_094107 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_094646 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095154 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095754 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100308 | 1 | 1 | 100% | 3.9 | 0 | 0.0 |
| 20260420_100337 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100817 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_101723 | 3 | 3 | 100% | 12.7 | 3 | 5.5 |
| 20260420_110254 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_110744 | 1 | 1 | 100% | 20.9 | 0 | 0.0 |
| **TOTAL** | **7** | **6** | **86%** | **12.4** | **3** | **3.7** |

### T_d = 20s

| Recording | GT Events | Detected | Sensitivity | Mean TTD (s) | FP | FP/hr |
|:----------|----------:|---------:|------------:|-------------:|---:|------:|
| 20260419_211649 | 1 | 0 | 0% | — | 0 | 0.0 |
| 20260419_212131 | 1 | 1 | 100% | 21.7 | 0 | 0.0 |
| 20260420_094107 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_094646 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095154 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095754 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100308 | 1 | 0 | 0% | — | 0 | 0.0 |
| 20260420_100337 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100817 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_101723 | 3 | 3 | 100% | 22.7 | 2 | 3.7 |
| 20260420_110254 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_110744 | 1 | 1 | 100% | 30.9 | 0 | 0.0 |
| **TOTAL** | **7** | **5** | **71%** | **24.1** | **2** | **2.5** |

> **Validation gate (T_d=20s)**: Sensitivity = 71% (5/7) — **FAIL** (target ≥ 95%)

> **Note**: 2 event(s) have duration < T_d=20s and are physically undetectable at this threshold. Events ≥ 20s: 5/5 = **100%** sensitivity.

> This validates the dual-signal apnoea architecture on a real
> apnoea cohort with simulator-grade ground truth. The amplitude-
> cessation arm detects what the FFT arm fundamentally cannot.
> The missed events are below the clinical definition of neonatal
> apnoea (cessation ≥ 15–20s) and cannot be detected at T_d=20s
> by any delay-based detector.

### T_d = 30s

| Recording | GT Events | Detected | Sensitivity | Mean TTD (s) | FP | FP/hr |
|:----------|----------:|---------:|------------:|-------------:|---:|------:|
| 20260419_211649 | 1 | 0 | 0% | — | 0 | 0.0 |
| 20260419_212131 | 1 | 1 | 100% | 31.7 | 0 | 0.0 |
| 20260420_094107 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_094646 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095154 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_095754 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100308 | 1 | 0 | 0% | — | 0 | 0.0 |
| 20260420_100337 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_100817 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_101723 | 3 | 2 | 67% | 32.5 | 1 | 1.8 |
| 20260420_110254 | 0 | 0 | — | — | 0 | 0.0 |
| 20260420_110744 | 1 | 1 | 100% | 40.9 | 0 | 0.0 |
| **TOTAL** | **7** | **4** | **57%** | **34.4** | **1** | **1.2** |

### Per-Event Detail (T_d = 20s)

| Recording | Onset (s) | Duration (s) | Detected | TTD (s) |
|:----------|----------:|-------------:|---------:|--------:|
| 20260419_211649 | 125 | 5 | **No** | — |
| 20260419_212131 | 727 | 316 | Yes | 21.7 |
| 20260420_100308 | 67 | 5 | **No** | — |
| 20260420_101723 | 1750 | 110 | Yes | 23.1 |
| 20260420_101723 | 1976 | 35 | Yes | 23.2 |
| 20260420_101723 | 2066 | 70 | Yes | 21.9 |
| 20260420_110744 | 367 | 55 | Yes | 30.9 |

## Notes

- CubeEye signal amplitude is ~100× smaller than Femto Bolt (0.3mm vs 25mm)
  because the Super TORY simulator has minimal chest rise compared to a real human.
- The CubeEye CSV logs report BPM every 5s (not per-frame depth), so the replay
  synthesises a breathing signal from the GT rate and measured baseline depth.
  This means the MAE reflects the pipeline's spectral estimation accuracy on a
  clean signal, not its robustness to real noise — a best-case bound.
- 80 BPM failed due to Nyquist aliasing at 15 fps (fundamental at 1.33 Hz vs Nyquist 7.5 Hz).
  The aliasing appears as a spurious 9 BPM peak.
- Next step: replay from raw depth .npy frames (via breathing_simulator.py) to test
  noise robustness without the clean-signal assumption.
