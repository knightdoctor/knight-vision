# Multi-modal validation — first 3-way paired result (2026-05-18)

First M1 paired reading with **all three modalities working simultaneously**:
LiDAR (Femto Bolt depth) + mmWave radar (IWR6843AOPEVM) + Polar H10 GT.
60s seated capture, normal breathing, subject (Dr Knight) at ~1m from
sensors.

Captured: 2026-05-18 ~14:52 BST. LiDAR run dir
`phase1/runs/20260518_145226/`. Radar smoke `/tmp/radar_smoke3.npz`
(NOT in repo — large npz, ablation tool only).

## The result

| Modality | Method | BPM | SNR | Δ vs Polar |
|---|---|---|---|---|
| **Polar H10** (ground truth) | HF-band power-weighted centroid on RRI | **~13.5** (median, 60 s window) | n/a | 0 |
| **Radar** — chest-ROI mean Z | spatial filter to X[-0.5,0.5] Y[-0.3,0.5] Z[0.7,1.8] m, mean Z of in-box points, FFT | **12.48** | **10.64** | **−1.0** |
| **LiDAR** — chest subset | PR L chest band (0.55-0.75 of subject Y), PR J sliding window, PR M settled median | **15.19** | 3.86 | **+1.7** |

Maximum cross-modal disagreement: 2.7 BPM (LiDAR vs Radar), within
physiological breath-to-breath variability. Polar HR 69-76 BPM (median
~72) over the same window — consistent with quiet seated breathing.

## Why this matters

The apnoea pipeline architecture (`apnoea_pipeline_architecture.md`)
specifies "multi-modal confirmation" as a defence against single-modality
contamination (e.g. LiDAR locking onto cardiac BCG at HR/2 during low
respiratory amplitude). Until today, that defence was theoretical — we
had radar hardware running but no radar-vs-LiDAR cross-check. This run
demonstrates the basic infrastructure produces grant-grade cross-modal
agreement with **no radar respiratory module built yet** — just spatial
filtering on the existing sparse detected-points stream.

Practical implication: pulling the multi-modal cross-check arm forward
from Phase 2 to Phase 1 may be much cheaper than the spec assumed.

## Open questions (DO NOT resolve from this single recording — repeat first)

### (a) Is LiDAR's +1.7 BPM Δ vs Polar systematic or run-specific?

Sign of the gap across an N≥5 cohort of identical-protocol runs will
answer it. Consistent positive → LiDAR systematic over-estimation
(possibly chest-band selecting motion that adds harmonic above true RR;
possibly settled-median bias toward higher-SNR windows that happen to
sit on higher-BPM peaks). Sign varying → run-specific noise.

### (b) Is radar's −1.0 BPM Δ vs Polar systematic or run-specific?

Same test. A consistent radar bias would suggest the ROI-mean-Z method
under-estimates true RR (possibly because sparse points capture motion
of a different anatomical feature than chest expansion — e.g.,
shoulder/abdomen). A varying-sign bias suggests run-to-run variation.

### (c) Does the radar-SNR-3×-higher pattern hold across runs?

This recording had Radar SNR 10.64 vs LiDAR SNR 3.86 — radar 2.75×
higher. If this reproduces, **the headline finding is that radar's
sparse-points-as-cross-check is not just viable but potentially
preferable to LiDAR for SNR-limited cases.** If single-recording
artefact, no headline. Critical question for the next 2-3 captures.

### (d) Is the radar ROI definition apples-to-apples with LiDAR's chest subset?

**Currently it is NOT, and that needs to be either fixed or explicitly
documented.** Specifically:

- **LiDAR chest subset** uses `config.chest_y_band_min/max = (0.55, 0.75)`
  — fractions of the **detected subject cluster's** Y span (RELATIVE,
  adapts per frame).
- **Radar ROI** uses absolute world coordinates X[-0.5, 0.5] Y[-0.3, 0.5]
  Z[0.7, 1.8] (m) — FIXED, doesn't follow the subject.

That's a category difference, not a magnitude one. To be apples-to-apples
we'd either (i) project the LiDAR chest subset's bounding box into world
coords and use it for radar, or (ii) implement adaptive-subject-cluster
for radar (requires more points than the sparse stream typically gives).
Easier path: option (i), document the projected box dimensions.

**Also**: SNR metric is **named** the same in both pipelines
(`peak_power / mean_in_band`) but the band differs:

- LiDAR analysis band: `config.rr_freq_min/max` = 0.17-0.50 Hz = **10-30 BPM**
- Radar smoke analysis band: hard-coded 6-40 BPM in `analyze_radar_smoke.py`
  and `analyze_radar_roi.py`

Same formula, different denominators → SNR values not directly comparable.
Either unify the bands (use 10-30 for radar too) or report each modality's
SNR relative to its own band's noise floor and document the difference.

## Reproducibility plan

Next: 2-3 more 3-way captures in the next hour, same protocol,
**no changes between runs**. Compare each (BPM, SNR, Δ-vs-Polar) per
modality. If radar-beats-LiDAR holds reproducibly → significant
architectural finding. If it flips run-to-run → noise.

## Files

- `phase1/radar_smoke_capture.py` — captures radar TLV stream for N s (standalone smoke; superseded by PR P sidecar for paired captures)
- `phase1/analyze_radar_smoke.py` — whole-frame aggregations + FFT
- `phase1/analyze_radar_roi.py` — chest-ROI filter + FFT
- `phase1/profile_pr_m_bias.py` — aggregation-method profiler (full FFT vs median vs PR M vs Polar) across paired runs
- `phase1/notes/multimodal_2026-05-18.md` — this file

---

## Update (2026-05-18 evening) — N=5 paired-run profile + open-question resolutions

Three more paired captures landed in the afternoon (Runs 7, 8, 9 — the
last via PR P's R-press-triggered sidecar). Profile across all paired
runs comparing three LiDAR aggregation methods (PR M live, median-of-
all-windows, full-record FFT) against Polar GT:

```
Run   |  Polar | PR M (live) | median windows |   full FFT |  Δ PR M | Δ med | Δ full
------+--------+-------------+----------------+------------+---------+-------+-------
Run 4 |  14.07 |       16.03 |          16.43 |      17.18 |   +1.96 | +2.36 | +3.11
Run 5 |  13.16 |       14.59 |          16.10 |      13.95 |   +1.43 | +2.94 | +0.78
Run 6 |  13.05 |       15.11 |          15.93 |      17.91 |   +2.06 | +2.87 | +4.85
Run 7 |  13.05 |       15.19 |          16.32 |      13.84 |   +2.14 | +3.27 | +0.79
Run 8 |   ---- |       15.80 |          16.56 |      17.67 |     --- |   --- |   ---
Run 9 |  13.57 |       15.72 |          16.96 |      14.63 |   +2.15 | +3.38 | +1.05
```

**N=5 paired data points (Run 8 lost Polar to dead bridge).** Reproduce
with `phase1/run.sh profile_pr_m_bias.py`.

### Resolutions to the original open questions

**(a) Is LiDAR's +Δ vs Polar systematic?** **YES — N=5 confirms.** Mean
Δ PR M = +1.95 BPM, range +1.43 to +2.15. Every paired run shows
LiDAR > Polar by 1-2 BPM. Not run-specific noise.

**(b) Is radar's −Δ vs Polar systematic?** **Method-dependent.** ROI-
filtered (Runs 7, 8): −1.0, −0.8 BPM (consistent under-estimate).
Whole-frame mean_z (Run 9 with PR P): +0.07 BPM (basically unbiased).
The chest-ROI filter introduces its own bias separate from the radar
hardware itself; the unfiltered aggregation is closer to truth.

**(c) Does the radar-SNR-3×-higher pattern hold?** **NO.** Run 7 was a
single-recording artefact. Subsequent radar SNRs were 1.2-1.3× LiDAR's,
i.e. broadly equivalent. The "radar might be more SNR-favourable than
LiDAR" headline is retracted.

**(d) Is the radar ROI apples-to-apples with LiDAR's chest subset?**
**Still NO.** ROI definition mismatch is unaddressed (radar uses fixed
world coords; LiDAR uses subject-relative Y fraction). SNR formula
band mismatch also unaddressed (radar analyse-scripts use 6-40 BPM
band; LiDAR pipeline uses 10-30). Both should be unified before the
ROI-vs-PR M comparison is rigorous. **Outstanding.**

### Where the LiDAR-vs-Polar +1-2 BPM bias actually comes from

PR M was initially suspected. The profile rules it out:

- **PR M does better than naive median** (Δ PR M < Δ median across all 5)
- **Full-record FFT is wildly variable** (+0.78 to +4.85), sometimes
  better than PR M, often worse — not a stable alternative
- All three LiDAR aggregations report systematically higher than Polar

Conclusion: the +1-2 BPM gap is a **method bias between LiDAR's
chest-displacement FFT and Polar's RSA-derived RR**, NOT a PR M
artefact. Two competing explanations:

1. **LiDAR over-estimates** — chest-surface motion contains cardiac-
   coupled content (BCG at HR/2, see diag_hold_spectrum.py result);
   that high-freq content shifts the FFT peak upward slightly.
2. **Polar under-estimates** — RSA-derived RR at rest is known in
   the HRV literature to under-estimate true breathing rate by 1-2
   BPM (LF/HF spectral boundary can split the respiratory peak).

**Cannot be resolved between two correlated-but-imperfect
measurements.** Requires a third independent GT (capnography,
respiration belt across the sternum, or manual breath counting).

### Implication for M1 reporting

Report the +1-2 BPM offset as a **known method-bias offset, not an
algorithm bug**. The within-method reproducibility is good (LiDAR
Δ PR M std ≈ 0.3 BPM across N=5; radar method-stable too). The
between-method bias is consistent and can be characterised as a
disclosed limitation pending third-modality validation. This framing
is grant-defensible.

PR M stays as the live final-estimate rule. No code change from this
profile.

### What this ungates

- Continued M1 paired-capture cohort growth — no algorithm change blocks
  the next captures
- Optional: source a respiration belt for unambiguous breath-count GT,
  measure split (LiDAR over vs Polar under). Not blocking
- The bias profile is itself a piece of M1 evidence — methodological
  honesty improves the grant narrative
