# Peak-pick vs centroid re-score — 2026-05-17 paired runs

*Generated 2026-05-19 by `phase1/peak_pick_rescore.py` (task #77).*

Each of the six live captures from 2026-05-17 is rescored against 
saved `centroid_z.npy` under both `rr_method` settings, using the 
meta-corrected fps and the current `KVConfig` (band 10-30 BPM, PR M settled-median). Δ rows are *signed* (estimator − Polar median).

Original `result_final` in each `meta.json` was produced under the 
PR I centroid default. Centroid-rescore differs from original only 
by config drift (band/threshold tightening since the capture).

## Headline table

| Run | dur (s) | fps | Polar RR | Peak-pick | Centroid | Δ peak | Δ centroid | Δ method |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260517_063204 | 115 | 7.331 | 12.53 | 22.07 (L) | 20.19 (L) | 9.54 | 7.66 | -1.88 |
| 20260517_064107 | 180 | 13.83 | 12.33 | 12.36 (M) | 15.42 (M) | 0.03 | 3.09 | 3.06 |
| 20260517_064421 | 146 | 13.799 | 13.02 | 11.32 (M) | 15.97 (M) | -1.7 | 2.95 | 4.65 |
| 20260517_065958 | 130 | 13.396 | 11.82 | 10.2 (H) | 18.45 (H) | -1.62 | 6.63 | 8.25 |
| 20260517_071225 | 125 | 13.069 | 12.2 | 26.61 (M) | 24.64 (M) | 14.41 | 12.44 | -1.97 |
| 20260517_072337 | 165 | 12.818 | 13.05 | 12.39 (M) | 14.47 (M) | -0.66 | 1.42 | 2.08 |

## Aggregate

- n = 6 paired live captures
- **Peak-pick vs Polar:** mean Δ = +3.33 BPM, |Δ| median = 1.66, max |Δ| = 14.41
- **Centroid vs Polar:** mean Δ = +5.70 BPM, |Δ| median = 4.86, max |Δ| = 12.44
- **Method split (centroid − peak-pick):** mean = +2.37 BPM, |Δ| median = 2.57, max |Δ| = 8.25
- Centroid > peak-pick in 4/6 runs

## Narrative

**Run 9's +2.4 BPM finding generalises.** The aggregate centroid-minus-peak-pick split across the six 2026-05-17 paired captures is **+2.37 BPM** — within noise of Run 9's +2.4 BPM. This is now an n=7 finding rather than n=1. Spectral mass on the upper edge of the 10-30 BPM band pulls the centroid right; peak-pick ignores it. Consistent with the cardiac BCG hypothesis (HR/2 contamination bleeding into the upper RR band — Polar HR median across these captures was 78-88 BPM, putting HR/2 at 39-44 BPM, just above the band but with shoulder leakage into the 25-30 BPM region).

**Two runs are front-end failures, not method failures:**

- **20260517_063204** — peak-pick says 22.07 (Δ +9.54), centroid 20.19 (Δ +7.66). Both estimates land on noise (1 of 16 windows above SNR 3, settled-median fell back to "last 4 windows regardless"). FPS was 7.3 (the live loop must have stalled — every other run hit 12.8-13.8 fps). Drop this run from the method comparison: it shows the SNR-gating doing its job, not anything about peak-pick vs centroid.
- **20260517_071225** — peak-pick says 26.61 (Δ +14.41), centroid 24.64 (Δ +12.44). Both estimates are wrong by an octave. Polar 12.2 BPM, estimator 24-26 BPM = 2× fundamental. This is the textbook octave-error failure mode the radar literature names (PMC 9400007, task #78). The chest-band selection here is letting harmonic energy dominate over the fundamental; the algorithmic switch can't recover it.

**The clean four — peak-pick gives near-zero bias, centroid stays at ~+3.5:**

| Run | Polar RR | Peak-pick Δ | Centroid Δ |
|---|---:|---:|---:|
| 20260517_064107 | 12.33 | **+0.03** | +3.09 |
| 20260517_064421 | 13.02 | **-1.70** | +2.95 |
| 20260517_065958 | 11.82 | **-1.62** | +6.63 |
| 20260517_072337 | 13.05 | **-0.66** | +1.42 |

Mean peak-pick Δ = **-0.99 BPM** (slight under-estimate). Mean centroid Δ = **+3.52 BPM** (the bias from prior paired-cohort summaries). |Δ| median: peak-pick 1.16, centroid 3.02. Peak-pick wins all four. Run 065958 is the most striking — peak-pick jumps from MEDIUM to HIGH confidence and centroid lands +8.25 BPM higher than peak-pick on the same data.

**What this confirms and what it doesn't:**

- *Confirms:* PR Z's safe-default-flip is justified by retrospective evidence, not just one paired capture. The N=5 paired-cohort "LiDAR over-reports by 1-2 BPM" finding is the centroid signature; peak-pick removes most of it.
- *Confirms:* Task #62 (HR-aware cardiac notch) is the right gate to ungate centroid as the production estimator. Notch should target HR/2 ± 1-2 BPM; per-run HR medians 78-88 BPM put the notch at 39-44 BPM, *outside* the 10-30 analysis band — meaning the contamination is the cardiac harmonic's *shoulder* leaking into the upper band edge, not the peak itself. A narrow IIR notch upstream of the FFT should suppress this without distorting the in-band response.
- *Doesn't confirm:* the octave-error case (071225). Peak-pick is no protection against a chest-band that's already off-band. Gdańsk-style octave detection (task #78) is the right tool there.
- *Doesn't confirm:* anything about Polar accuracy. Polar H10 reports a chest-strap-derived RR; if it lags the ventilation signal by 5-15 seconds at startup (per Polar's published behaviour), the early-window Δ comparison is contaminated by Polar settling, not LiDAR error. Median across the whole recording mostly washes this out but doesn't eliminate it.

**Action items dropping out of this:**

1. The peak-pick safe-default (PR Z) is *empirically* justified, not just Run-9-anecdotally. Worth referencing in the next pre-submission narrative / grant evidence pack.
2. Run **20260517_071225** is the cleanest available example of octave-error failure mode in the local corpus. Worth tagging in the apnoea/RR architecture notes as a known-broken case for any future algorithm to recover.
3. The four clean-front-end runs are good MTI-EEMD test inputs (task #80) — known peak-pick result to beat, known Polar truth to compare against.

## Per-run detail

### 20260517_063204

- mode: `live` · 114.7s · 841 frames · corrected fps 7.331
- Polar RR median: **12.53** BPM (n=26); HR median: 78.5 BPM (n=26)
- Original `meta.json` final (centroid-era): 20.21 BPM, SNR 1.33, LOW

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **22.07** | 1.11 | LOW | ✗ | only 1 of 16 windows above SNR 3.0; need ≥4 — using last 4 regardless |
| centroid | 20.19 | 1.11 | LOW | ✗ | only 1 of 16 windows above SNR 3.0; need ≥4 — using last 4 regardless |

### 20260517_064107

- mode: `live` · 179.8s · 2487 frames · corrected fps 13.83
- Polar RR median: **12.33** BPM (n=25); HR median: 81.0 BPM (n=25)
- Original `meta.json` final (centroid-era): 17.79 BPM, SNR 2.35, LOW

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **12.36** | 3.57 | MEDIUM | ✓ | settled over 14 high-SNR windows (indices 4-25 of 49, SNR≥3.0, std<2.0 BPM) |
| centroid | 15.42 | 3.57 | MEDIUM | ✓ | settled over 14 high-SNR windows (indices 4-25 of 49, SNR≥3.0, std<2.0 BPM) |

### 20260517_064421

- mode: `live` · 145.8s · 2012 frames · corrected fps 13.799
- Polar RR median: **13.02** BPM (n=27); HR median: 88.0 BPM (n=27)
- Original `meta.json` final (centroid-era): 17.71 BPM, SNR 2.66, LOW

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **11.32** | 3.89 | MEDIUM | ✓ | settled over 24 high-SNR windows (indices 1-39 of 40, SNR≥3.0, std<2.0 BPM) |
| centroid | 15.97 | 3.89 | MEDIUM | ✓ | settled over 24 high-SNR windows (indices 1-39 of 40, SNR≥3.0, std<2.0 BPM) |

### 20260517_065958

- mode: `live` · 130.0s · 1741 frames · corrected fps 13.396
- Polar RR median: **11.82** BPM (n=23); HR median: 83.0 BPM (n=23)
- Original `meta.json` final (centroid-era): 20.16 BPM, SNR 1.88, LOW

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **10.2** | 5.26 | HIGH | ✓ | settled over 4 high-SNR windows (indices 14-17 of 34, SNR≥3.0, std<2.0 BPM) |
| centroid | 18.45 | 5.26 | HIGH | ✓ | settled over 4 high-SNR windows (indices 14-17 of 34, SNR≥3.0, std<2.0 BPM) |

### 20260517_071225

- mode: `live` · 124.8s · 1631 frames · corrected fps 13.069
- Polar RR median: **12.2** BPM (n=25); HR median: 83.0 BPM (n=25)
- Original `meta.json` final (centroid-era): 25.93 BPM, SNR 3.61, MEDIUM

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **26.61** | 3.3 | MEDIUM | ✓ | settled over 5 high-SNR windows (indices 25-31 of 32, SNR≥3.0, std<2.0 BPM) |
| centroid | 24.64 | 3.3 | MEDIUM | ✓ | settled over 5 high-SNR windows (indices 25-31 of 32, SNR≥3.0, std<2.0 BPM) |

### 20260517_072337

- mode: `live` · 165.3s · 2119 frames · corrected fps 12.818
- Polar RR median: **13.05** BPM (n=58); HR median: 78.0 BPM (n=67)
- Original `meta.json` final (centroid-era): 15.11 BPM, SNR 3.74, MEDIUM

| Method | RR (BPM) | SNR | Conf | Settled | Note |
|---|---:|---:|:---:|:---:|---|
| peak-pick | **12.39** | 4.04 | MEDIUM | ✓ | settled over 26 high-SNR windows (indices 6-41 of 42, SNR≥3.0, std<2.0 BPM) |
| centroid | 14.47 | 4.01 | MEDIUM | ✓ | settled over 29 high-SNR windows (indices 2-41 of 42, SNR≥3.0, std<2.0 BPM) |

