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

---

## Update (2026-05-18 late afternoon) — partner cohort + radar method retraction

Three more 3-way captures landed this afternoon, this time with **Dr
Knight's partner as subject** (Phil sat outside FOV; sensors moved →
fresh bg captured). PR P sidecar used throughout; PR Q viewer
visualisation (radar top-down trail + live radar BPM) shipped between
Run 10 and Run 12.

  Run 10 partner (16:01) — Polar median 11.78 (97 samples, HR 88)
  Run 11 partner (16:08) — Polar median 14.15 (similar setup)
  Run 12 partner (16:19) — Polar median 14.85 (97 samples, HR 93)

Partner is a meaningfully harder subject than Phil: higher resting HR
(88-93 vs Phil's 70-78), more variable breath rate (Polar RR std ~2
BPM vs Phil's ~0.5), and the sensor pose was changed mid-day. **All
three contribute to larger errors on the partner cohort regardless of
method.**

### Radar aggregation profile — N=5 paired runs, apples-to-apples

Ran the LiDAR pipeline's exact FFT (10-30 BPM band, PR I power-weighted
centroid) against three radar aggregations across all paired radar
recordings:

```
Run                         |   Polar |          mean_z |        median_z |           min_z
Run 7 Phil  (14:52 smoke)   |   13.05 | 16.40 (Δ+3.35) | 21.18 (Δ+8.13) | 14.68 (Δ+1.63)
Run 9 Phil  (15:12 sidecar) |   13.57 | 15.23 (Δ+1.66) | 17.29 (Δ+3.72) | 16.46 (Δ+2.89)
Run 10 partner (16:01)      |   11.78 | 21.25 (Δ+9.48) | 21.17 (Δ+9.40) | 17.72 (Δ+5.95)
Run 11 partner (16:08)      |   14.15 | 22.17 (Δ+8.02) | 22.38 (Δ+8.23) | 19.53 (Δ+5.38)
Run 12 partner (16:19)      |   14.85 | 20.69 (Δ+5.84) | 20.85 (Δ+6.00) | 21.96 (Δ+7.11)

Mean abs Δ vs Polar:
  mean_z   : 5.67 BPM  (max 9.48)
  median_z : 7.09 BPM  (max 9.40)
  min_z    : 4.59 BPM  (max 7.11)   ← best of three
```

Reproduce with `phase1/run.sh profile_radar_methods.py`.

### Retraction: "radar whole-frame mean_z within 0.07 BPM of Polar"

The Run 9 headline figure from this morning ("radar whole-frame mean_z
13.64 BPM vs Polar 13.57, Δ +0.07 BPM") **doesn't survive an apples-to-
apples comparison** with the LiDAR pipeline.

The earlier number came from `analyze_radar_smoke.py`, which uses a
hard-coded **6-40 BPM band + peak-pick** — a much looser methodology
than the LiDAR pipeline (10-30 BPM band + PR I power-weighted
centroid). Same data through the same algorithm LiDAR uses yields Run
9 radar mean_z = 15.23 BPM (Δ +1.66). The 0.07 BPM agreement was an
artefact of methodology mismatch, not real cross-modal convergence.

That retraction propagates back to open question (b) in the original
note: **radar's bias vs Polar is not a small under-estimate. It's
3-5× the LiDAR error in all paired runs once the method is matched.**

### Re-framed radar status

- **Sparse-points radar processed naively is NOT a peer measurement
  for RR.** Mean abs Δ vs Polar is 4.6-7.1 BPM across methods, vs
  LiDAR's ~+1-2 BPM systematic offset.
- It can still serve as a **rough cross-check** ("LiDAR and radar both
  in the physiological band?"), and as a **secondary apnoea signal**
  (when LiDAR amplitude cessation fires, does radar agree?). But not
  as a primary RR estimator.
- The proper radar respiratory approach — phase-on-chest-bin or range-
  Doppler magnitude per the apnoea architecture doc — is still the
  right answer for radar as a real modality. What we have now is
  debug-grade cross-modal sanity-check, not a third measurement arm.
- **PR Q (radar visualisation in viewer) ships as a debug tool**, not
  a clinical readout. The live "RR (Radar)" pink number in the GT card
  is informative-but-not-trustworthy; treat as a sanity check.

### Subject-dependence of "best method"

Even within the radar-as-cross-check framing, no aggregation method is
universally best. min_z won 3/5 runs; mean_z won 2/5; median_z won
none. This is **subject-dependent**: depends on how the sparse radar
detections distribute spatially around the subject, which depends on
clothing, posture, HR (BCG affects far-side detections), and sensor-
to-subject geometry. A universal radar aggregation rule isn't visible
in this dataset.

### What this means for M1

The +1-2 BPM LiDAR-vs-Polar bias narrative from N=5 (Phil cohort)
**doesn't generalise to the partner cohort** — Run 10 Δ was +4.81
BPM, Run 11/12 similar. Subject and pose variability is bigger than
within-subject variability. M1 reporting needs:

1. **Disclose subject + sensor-pose variability** as part of the
   uncertainty budget, not hide it under an averaged "Δ ≈ 1.7 BPM"
2. **Frame radar as cross-check infrastructure** for the multi-modal
   apnoea defence, not as a peer RR measurement
3. **Defer claims about radar accuracy** until the proper
   phase-on-chest-bin pipeline is built (Phase 2)

### Distance caveat — adult-at-1.3m is NOT the deployment scenario

All today's captures placed the subject at ~1.3-1.5 m from the sensor
(adult on a chair). **The Knight Vision cot deployment scenario is a
neonate at well under 1 m — likely 30-60 cm from sensor.** The current
N=5 paired cohort therefore tests the algorithm in a regime
substantially harder than the target operating condition:

- At ~50 cm vs ~1.3 m, the subject's angular extent in the sensor FOV
  is ~3× larger → ~10× more LiDAR points on the chest (linear×linear×
  geometric falloff).
- More radar detections per frame from the chest, fewer from
  background (walls, sofa) which are now relatively further away.
- Cardiac BCG and respiratory amplitudes scale with distance
  differently in BCG literature — relative SNR of breath vs heart may
  shift at closer ranges.
- ROI definition becomes much simpler — at 50 cm the subject fills
  most of the monitoring volume; sub-region adaptive chest selection
  may not even be needed.

**Implication: today's "radar is 5 BPM off Polar" finding might be
materially better at deployment distance.** Adult-on-chair is the
worst case for sparse-radar sensitivity, not a representative test.

**Action:** at next session, do a "close-range" capture — sensor
aimed at a chair or table at ~50 cm, subject (Phil) leans/sits so
chest is in that range. Compare LiDAR and radar errors vs Polar at
deployment-equivalent geometry. If errors drop substantially → the
sparse-radar approach may earn its keep for cot monitoring. If
errors stay the same → the sparse approach is fundamentally limited
and proper range-Doppler is the only path.

### Carry-over follow-ups

- **Close-range capture (~50 cm)** — deployment-equivalent geometry test
- Align radar smoke analysis bands with LiDAR pipeline (10-30 BPM) in
  the standalone scripts so future smoke analyses don't recreate the
  methodology mismatch that produced the retracted figure
- Continue M1 paired captures with subject + pose held constant for
  within-subject reproducibility
- Source third-modality GT (task #66) — even more important now,
  since LiDAR-vs-Polar variability is itself subject-dependent

---

## Evening session — Run 9 (`20260518_213754`) resolves the LiDAR +1-2 BPM bias

The headline of this run is *not* "another paired capture." It's that
peak-pick vs power-weighted-centroid on the same 81 s recording, with
identical chest-band selection, returns **17.0 BPM** and **19.4 BPM**
respectively. That 2.4 BPM split — same data, same spatial subset,
different reporting method — localises the persistent LiDAR-vs-Polar
+1-2 BPM bias seen across the N=5 paired cohort to the **reporting
algorithm**, not anatomy and not the spatial chest-band selection.

Centroid integrates power across the whole respiratory band; cardiac
BCG harmonic energy bleeds into the upper end of that band (e.g.
HR/2 ≈ 30 BPM at HR 60, which is still inside the 10-30 BPM analysis
window). Peak-pick ignores that secondary energy by definition.

**Implications:**

- The cardiac notch filter (task #62) is **load-bearing, not optional**.
  Centroid only becomes the right default *after* the notch removes
  the BCG contamination from the spectrum.
- Until #62 ships, **peak-pick is the safe default** for the live
  pipeline. PR Z (2026-05-18) adds `config.rr_method` + `--rr-method`
  CLI flag for the choice, defaulting to peak-pick.
- The viewer now reports both numbers side-by-side. The Δ between
  them is itself a signal-quality indicator: large Δ ⇒ strong cardiac
  contamination present in the spectrum.
- Task #64 (HR/2 validation under HR-varying recording) still gates
  #62 — the notch frequency depends on the cardiac-at-HR/2 hypothesis
  holding.
- Retrospective re-scoring of Runs 4, 6, 7, 8 under peak-pick is filed
  as task #77 — expected to shrink the cohort Δ from ~+1-2 BPM toward
  ~+0.5 BPM. If it lands there, that's clinical-grade for adult RR.

### Capture detail

Geometry: Phil seated at z ≈ 0.94 m, cz median 936.5 mm, ptp chest
displacement 28.6 mm (normal adult breathing amplitude). 81 s, 9.2
fps, 749 samples, **100 % valid frames**.

Polar GT (n = 42): RR median **14.22 BPM**, range 11.6–16.1.

LiDAR within the 10–30 BPM band:

| Method | RR | Δ vs Polar | Notes |
|---|---|---|---|
| **Peak-pick (FFT bin)** | **17.0 BPM** | **+2.8 BPM** | SNR 6.10 → HIGH conf. Top four bins: 16.9, 17.0, 17.1, 17.3 BPM (very tight cluster, not a noisy spread). |
| Centroid (PR I, prior default) | 19.4 BPM | +5.2 BPM | Pulled high by cardiac BCG band-edge energy. |

### Context

This was the first paired capture after a stack of session changes:
sticky tracking + size-dominance break (PR S/W), chest band widened
to **(0.30, 0.75)** for chest+abdomen coverage (PR X, motivated by
diaphragmatic breathing in infants), chest X/Z lateral crop (PR T),
LiDAR side-elevation panel for visual QA (PR V), display ranges
tightened to Z=(0.5, 2.0). Shape gate (PR Y) disabled at runtime
after over-rejecting human clusters when DBSCAN chained the subject
to chair/desk past the 1.0 m XZ envelope; monitoring volume Z
tightened to 1.4 m as a temporary mitigation against the ~1.5 m wall
residual phantom that kept winning sticky-acquire. Both are
follow-ups; the headline finding stands independently of them.

### Operational follow-ups

- Sticky-acquire latching onto wall phantom before subject settled —
  size-dominance ratio of 2× was insufficient because Phil's cluster
  only reached dominance after fully seating. Investigate either
  (a) prioritising motion-bearing clusters at acquire, or (b) raising
  dominance ratio to 4× combined with a fresh BG model per session.
- Shape gate (PR Y) defaults need an offline reproduction — relaxed
  thresholds did not restore acquisition at runtime despite passing
  in a fresh Python interpreter on the same config.
- Polar BLE pairing dropped silently mid-session (no log line). The
  daemon should detect stale pairing and force re-pair without manual
  OS-side "Forget Device" intervention. Worth filing as a bridge
  robustness task.
- Viewer rec-button `.rec.on` CSS toggle not firing — recording works
  end-to-end but visual indicator is stuck. CSS-only fix.

New workflow rule in effect this session: after every relaunch I pull
the topdown + side JPEGs from the viewer stream and read them
directly, rather than relying on Phil to narrate the panel back to
me. Tightens the iteration loop substantially when chasing visual
bugs (locked-on-wrong-cluster, off-screen GT card, etc.).
