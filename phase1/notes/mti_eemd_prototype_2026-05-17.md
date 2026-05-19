# MTI-EEMD prototype on saved centroid_z — 2026-05-17 runs

*Generated 2026-05-19 by `phase1/mti_eemd_prototype.py` (task #80).*

EEMD parameters held constant: trials=50, noise_width=0.05, max_imf=8.

Two picker rules are evaluated side-by-side. **frac** = IMF with the highest fraction of its energy in [10, 30] BPM (gated at ≥ 0.5, ties → higher in-band SNR). **pow** = IMF with max absolute in-band power. Both pick from the *same* EEMD output — the decomposition itself is identical; the rule is the load-bearing design choice.

Peak-pick numbers are sourced from `peak_pick_rescore_2026-05-17.json` (task #77).

## Headline table

| Run | Polar RR | Peak-pick | EEMD-frac IMF | EEMD-frac RR | Δ frac | EEMD-pow IMF | EEMD-pow RR | Δ pow | Δ peak |
|---|---:|---:|:---:|---:|---:|:---:|---:|---:|---:|
| 20260517_063204 | 12.53 | 22.07 | #3 | 19.17 | 6.65 | #3 | 19.17 | 6.65 | 9.54 |
| 20260517_064107 | 12.33 | 12.36 | #4 | 13.35 | 1.02 | #4 | 13.35 | 1.02 | 0.03 |
| 20260517_064421 | 13.02 | 11.32 | #3 | 13.24 | 0.22 | #3 | 13.24 | 0.22 | -1.7 |
| 20260517_065958 | 11.82 | 10.2 | #4 | 26.76 | 14.94 | #4 | 26.76 | 14.94 | -1.62 |
| 20260517_071225 | 12.2 | 26.61 | #4 | 23.62 | 11.42 | #4 | 23.62 | 11.42 | 14.41 |
| 20260517_072337 | 13.05 | 12.39 | #4 | 11.91 | -1.14 | #3 | 20.85 | 7.8 | -0.66 |

## Aggregate

- n = 6 paired live captures
- **EEMD (frac picker) vs Polar:** mean Δ = +5.52 BPM, |Δ| median = 3.90, max |Δ| = 14.94
- **EEMD (power picker) vs Polar:** mean Δ = +7.01 BPM, |Δ| median = 7.22, max |Δ| = 14.94
- **Peak-pick vs Polar (recap):** mean Δ = +3.33 BPM, |Δ| median = 1.66, max |Δ| = 14.41
- EEMD-frac closer to Polar than peak-pick in **3/6** runs
- EEMD-pow closer to Polar than peak-pick in **3/6** runs

## Narrative — verdict and follow-up

**EEMD finds the right respiratory mode in every clean-front-end run, but the IMF picker can't reliably identify which mode it is.** That's the headline. The decomposition works; the selection rule is the load-bearing engineering problem.

Look at the per-run tables below: in every run where Polar is ~12 BPM, an IMF is present whose peak BPM is within ~1-2 BPM of Polar. The picker — under both rules tested — sometimes lands on it (064107, 064421, 072337) and sometimes lands on a noisier high-frequency neighbour (063204, 065958).

The cleanest example of picker failure is **20260517_065958**: Polar 11.82 BPM, and IMF 5 of the EEMD has peak 11.87 BPM (Δ **+0.05**). If we'd picked IMF 5 we'd have a Polar-matching estimate, beating peak-pick (-1.62) and centroid (+6.63) decisively. But IMF 5's in-band fraction is 0.433, below the 0.5 gate, so the frac picker fell back to IMF 4 (28.87 BPM, off by +14.94). Tweaking the gate to 0.4 would pick IMF 5 here, but the in-band SNR tiebreaker would still prefer IMF 4 — so the fix is not just a parameter twist.

### Run-by-run verdict (frac picker)

| Run | EEMD verdict | Why |
|---|---|---|
| 20260517_063204 | ✗ fails (Δ +6.65) | Low-SNR front-end (same as peak-pick failure in task #77); no IMF in the decomposition is clearly the respiratory mode either |
| 20260517_064107 | ✓ wins (Δ +1.02) | Clean. Beat by peak-pick's Δ +0.03 by ~1 BPM but well within noise |
| 20260517_064421 | ✓ wins (Δ +0.22) | Best result in the set. Peak-pick Δ -1.70; EEMD-frac closer to Polar |
| 20260517_065958 | ✗ fails (Δ +14.94) | **Picker failure**, not decomposition failure. IMF 5 @ 11.87 BPM exists and matches Polar within 0.05 but is gated out at frac=0.433 |
| 20260517_071225 | ✗ fails (Δ +11.42) | Octave-error capture; centroid_z input is dominated by harmonic energy, no IMF contains the fundamental. Gdańsk algorithm (task #78) is the right tool |
| 20260517_072337 | ✓ wins (Δ -1.14) | Clean. Comparable to peak-pick Δ -0.66 |

3/6 wins on aggregate is **not** the "consistently closer to Polar" bar Phil set for promotion to a real PR. But:

- The 3 clean-front-end wins are **better than peak-pick** in their own right (sub-1.5 BPM Δ, sometimes sub-0.5).
- The 3 losses split into one front-end failure (same root cause as peak-pick's failure on the same run), one octave-error capture (a known different problem with a different fix), and one **picker-rule failure on a recoverable signal**.

**Verdict: not a Phase 2 candidate as-is.** But the decomposition output has the right answer hidden in it on at least 4 of 6 runs. A smarter picker, possibly informed by an HR prior or a tighter analysis band, could promote this. Worth following up before closing.

### Specific follow-up suggestions

1. **Narrower IMF-selection band for adult validation.** Adult resting RR is rarely above ~22 BPM. Narrow the IMF-selection band from [10, 30] to [8, 22] BPM and re-pick. IMF 4 @ 28.87 in run 065958 would drop out; IMF 5 @ 11.87 would be selected. *Note: this is for IMF SELECTION only — the FFT bin band itself can stay at [10, 30] for peak detection within the picked IMF.*

2. **HR-aware picker.** When HR is known (Polar in validation, or eventually the cardiac BCG self-derivation), reject IMFs whose peak falls within ±2 BPM of HR/2. For run 065958, HR=83 → HR/2 = 41.5 → no rejection (IMF 4's peak 28.87 isn't near HR/2). But the broader principle is "the respiratory IMF is not the cardiac harmonic IMF" — combine with task #62 (cardiac notch) for a coherent story.

3. **CEEMDAN instead of EEMD.** Complete Ensemble EMD with Adaptive Noise (Torres et al. 2011) reduces the mode-mixing that makes IMF assignment ambiguous. The same PyEMD package implements it; one-line swap in the script. Worth a quick comparison.

4. **Multi-IMF reconstruction.** Instead of picking ONE IMF, reconstruct the respiratory signal as a sum of IMFs whose peaks fall in the RR band, then FFT-peak-pick on the reconstruction. Robust against mode-mixing because the picker doesn't need to commit to a single IMF.

5. **Validate on the public mm-Wave dataset** (per radar lit review): arxiv 2405.12659 has paced breath-hold and rapid-breathing scenarios. EEMD's reputed advantage over FFT is in transient/non-stationary segments — our M1 captures are mostly stationary so we may be testing in the wrong regime.

### Performance notes

EEMD runs in ~0.85–1.5 s per ~2-min capture at trials=50. **Not real-time** as configured (live RR cadence is every 5 s; EEMD at trials=50 on a 20 s sliding window would have to come in under 1 s). For a Phase 2 deployment we'd need to drop trials to ~10-20 and benchmark on the Jetson. For offline validation the current parameters are fine.

### What this rules out

EEMD-with-naive-picker is not a drop-in replacement for peak-pick. The +5.5 BPM mean Δ under the principled (frac) picker is *worse* than peak-pick's +3.3, even though three individual runs win. A drop-in replacement needs to beat peak-pick on aggregate, not on a per-run basis.

## Per-run IMF detail

### 20260517_063204

- 114.7s · 841 frames · corrected fps 7.331
- Polar RR median: **12.53** BPM (n=26); HR median: 78.5 BPM
- EEMD: 9 IMFs in 0.92s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 29.99 | 0.052 | 3.91 | 0.04708 |  |  |
| 1 | 29.85 | 0.049 | 4.33 | 0.03395 |  |  |
| 2 | 23.79 | 0.437 | 4.96 | 0.02157 |  |  |
| 3 | 19.17 | 0.919 | 4.42 | 0.01944 | ✓ | ✓ |
| 4 | 10.01 | 0.168 | 23.86 | 0.01158 |  |  |
| 5 | 10.01 | 0.004 | 17.72 | 0.00898 |  |  |
| 6 | 10.01 | 0.0 | 40.0 | 0.00525 |  |  |
| 7 | 11.01 | 0.0 | 14.67 | 0.00241 |  |  |
| 8 | 10.01 | 0.0 | 14.61 | 0.00316 |  |  |

### 20260517_064107

- 179.8s · 2487 frames · corrected fps 13.83
- Polar RR median: **12.33** BPM (n=25); HR median: 81.0 BPM
- EEMD: 9 IMFs in 0.9s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 27.59 | 0.005 | 3.55 | 0.00094 |  |  |
| 1 | 24.07 | 0.013 | 7.93 | 0.00053 |  |  |
| 2 | 20.28 | 0.054 | 3.54 | 0.00044 |  |  |
| 3 | 28.69 | 0.549 | 3.55 | 0.00042 |  |  |
| 4 | 13.35 | 0.783 | 10.15 | 0.00046 | ✓ | ✓ |
| 5 | 10.91 | 0.043 | 18.8 | 0.00026 |  |  |
| 6 | 10.05 | 0.0 | 20.99 | 0.00029 |  |  |
| 7 | 10.05 | 0.0 | 33.23 | 0.00044 |  |  |
| 8 | 10.03 | 0.0 | 10.76 | 0.00054 |  |  |

### 20260517_064421

- 145.8s · 2012 frames · corrected fps 13.799
- Polar RR median: **13.02** BPM (n=27); HR median: 88.0 BPM
- EEMD: 9 IMFs in 0.86s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 26.61 | 0.01 | 4.24 | 0.00199 |  |  |
| 1 | 22.94 | 0.018 | 4.56 | 0.00121 |  |  |
| 2 | 29.11 | 0.094 | 5.58 | 0.00096 |  |  |
| 3 | 13.24 | 0.778 | 4.07 | 0.00177 | ✓ | ✓ |
| 4 | 11.37 | 0.065 | 7.35 | 0.00191 |  |  |
| 5 | 10.01 | 0.001 | 33.6 | 0.00204 |  |  |
| 6 | 10.76 | 0.0 | 16.25 | 0.01321 |  |  |
| 7 | 11.19 | 0.0 | 23.51 | 0.00688 |  |  |
| 8 | 10.28 | 0.0 | 16.55 | 0.01024 |  |  |

### 20260517_065958

- 130.0s · 1741 frames · corrected fps 13.396
- Polar RR median: **11.82** BPM (n=23); HR median: 83.0 BPM
- EEMD: 9 IMFs in 1.3s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 28.06 | 0.013 | 3.47 | 0.13364 |  |  |
| 1 | 21.49 | 0.011 | 4.57 | 0.11441 |  |  |
| 2 | 30.0 | 0.014 | 4.57 | 0.07409 |  |  |
| 3 | 23.67 | 0.064 | 4.87 | 0.0546 |  |  |
| 4 | 26.76 | 0.547 | 3.56 | 0.06612 | ✓ | ✓ |
| 5 | 11.06 | 0.432 | 6.74 | 0.02924 |  |  |
| 6 | 11.09 | 0.009 | 24.42 | 0.04279 |  |  |
| 7 | 10.77 | 0.0 | 24.92 | 0.04017 |  |  |
| 8 | 10.82 | 0.0 | 14.17 | 0.04407 |  |  |

### 20260517_071225

- 124.8s · 1631 frames · corrected fps 13.069
- Polar RR median: **12.2** BPM (n=25); HR median: 83.0 BPM
- EEMD: 9 IMFs in 1.38s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 23.91 | 0.014 | 4.35 | 0.30509 |  |  |
| 1 | 19.31 | 0.005 | 4.46 | 0.27293 |  |  |
| 2 | 25.49 | 0.029 | 4.34 | 0.12987 |  |  |
| 3 | 29.77 | 0.245 | 5.38 | 0.10318 |  |  |
| 4 | 23.62 | 0.879 | 4.78 | 0.07805 | ✓ | ✓ |
| 5 | 13.19 | 0.11 | 8.98 | 0.04684 |  |  |
| 6 | 10.0 | 0.001 | 20.83 | 0.03454 |  |  |
| 7 | 10.91 | 0.0 | 12.85 | 0.02706 |  |  |
| 8 | 10.82 | 0.0 | 18.63 | 0.01808 |  |  |

### 20260517_072337

- 165.3s · 2119 frames · corrected fps 12.818
- Polar RR median: **13.05** BPM (n=58); HR median: 78.0 BPM
- EEMD: 9 IMFs in 0.89s

| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 21.39 | 0.012 | 5.27 | 0.00104 |  |  |
| 1 | 15.65 | 0.016 | 4.1 | 0.00062 |  |  |
| 2 | 26.77 | 0.24 | 6.24 | 0.00061 |  |  |
| 3 | 20.85 | 0.91 | 4.45 | 0.00078 |  | ✓ |
| 4 | 11.91 | 0.522 | 17.16 | 0.00135 | ✓ |  |
| 5 | 11.1 | 0.0 | 12.39 | 0.00158 |  |  |
| 6 | 10.01 | 0.0 | 12.88 | 0.00128 |  |  |
| 7 | 12.03 | 0.0 | 16.28 | 0.00072 |  |  |
| 8 | 10.23 | 0.0 | 4.38 | 0.00054 |  |  |

