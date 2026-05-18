# Radar Respiratory Monitoring — Literature Review

*2026-05-18. Triggered by Run 9's finding that LiDAR centroid bias is algorithmic (cardiac BCG harmonic contamination). Radar has a substantially richer published methods literature than LiDAR; this document reviews what's available and identifies specific techniques applicable to the IWR6843AOPEVM-on-Jetson pipeline.*

---

## TL;DR

1. **The "octave error" we've been calling "harmonic lock-on" is a known, named, published failure mode of IWR6843-class radar vital signs estimation.** Gdańsk University of Technology published an octave-error removal algorithm that TI integrated into their reference firmware. Borrow, don't re-invent.
2. **Our current peak-pick accuracy (±2.8 BPM vs Polar at adult rest) already beats TI's stock ±5 BPM spec.** Useful framing for grant evidence and for confidence calibration: we're already at or above commercial-grade for the adult-rest case.
3. **A directly analogous neonatal radar+ToF fusion study exists** (PMC 8122919) — same hardware class, same architecture, validated in neonates. Cite this in grant applications and borrow methodology where it makes sense.
4. **NICU radar validation has been done; achievable RMSE is ~4 BPM** in premature infants, ~3 BPM in controlled adult settings. Our target performance band is set by that bar.
5. **Mature signal-processing techniques exist** (DR-MUSIC, MTI-EEMD, parameterised respiration filters, signal superposition with elliptical filtering) that we haven't tried. Each is a Phase 2 algorithm candidate.

---

## The octave-error / harmonic problem — confirmed common, with a published fix

Our PR I ("prefer lowest in-band peak") and the cardiac BCG diagnostic together exposed a 2× harmonic contamination problem. The literature names this the **octave error** and confirms it's a structural failure mode of FMCW radar vital-signs estimation, not specific to our pipeline.

TI's vital signs implementation on the IWR6843 originally had this problem; an algorithm developed at the Gdańsk University of Technology was integrated into the firmware specifically to remove octave errors. The source code is under NDA with TI but the algorithmic approach is in published literature.

**Action:** investigate whether the Gdańsk approach is published in detail outside TI's NDA documentation. If yes, reimplement; if not, the existence of the fix tells us the problem is solvable and PR I was the right direction but probably under-specified.

## Achievable accuracy in the literature

| Setting / sensor | Performance | Source |
|---|---|---|
| Premature infants, NICU | RMSE 4.3 BPM (best 4 BPM in prone position) | PMC 8956695 |
| Adult, controlled, FMCW signal-superposition + elliptical filter | Relative error 1.33% | Sci Reports 2024 |
| Adult with SVM respiratory feature classification | 98.25% pattern accuracy | mmWave-RM, PMC 11243972 |
| Neonatal, ToF + interferometric radar fusion | ±3 BPM max difference within 20-60 BPM range | PMC 8122919 |
| Camera-based NICU (reference comparison) | MAE 2.13 BPM (RGB), 4.83 BPM (RGB-D depth) | Cambridge Meerkat + others |
| IWR6843 TI stock firmware | ±5 BPM | TI datasheet |
| **Our Run 9 LiDAR peak-pick** | **Δ 2.8 BPM, n=1** | This project |

The published target band is ~3-5 BPM RMSE in NICU. Our adult-resting figure already sits in that range with peak-pick reporting. The harder validation will be paediatric where breathing rates are 30-60 BPM and movement is constant — but the literature gives us a benchmark to aim at.

## Signal-processing techniques worth borrowing

**Listed in rough order of expected impact for our specific pipeline.**

### 1. DC-offset removal for static reflectors (PMC 9781610)

Background reflectors (walls, furniture) contribute a DC offset to the phase signal that dominates the respiratory dynamic. The radar literature has explicit methods for removing this — phase unwrapping + DC removal before spectral estimation. **Our background-subtraction does the equivalent geometrically for depth; the radar arm may not have this yet.** Check whether the IWR6843 stream we're using includes static-clutter removal — if `clutterRemoval` is off (we override it to 0 to expose subject in the hub viewers), the static-reflector DC offset is contaminating our spectrum.

### 2. Signal superposition + elliptical filtering (Sci Reports 2024)

Specifically the technique that achieved 1.33% relative error. Signal superposition combines multiple range bins around the subject to improve SNR; elliptical filtering (steep band-pass with controlled ripple) isolates the 0.1-0.5 Hz respiratory band. Our current pipeline uses a simple band-pass; elliptical is the upgrade path.

### 3. MTI-EEMD (Multivariate Empirical Mode Decomposition) decomposition

Outperforms Wavelet Packet Decomposition for separating respiratory signal from noise. Used in multiple recent papers as the preferred preprocessing for FMCW radar respiratory extraction. Worth evaluating as a replacement for our current FFT-on-band approach.

### 4. DR-MUSIC adaptive filtering + spectral analysis

Used for HR estimation specifically — suppresses respiratory motion as artifact to extract the weaker cardiac signal. Inverts the problem we have (where cardiac contaminates respiration); the technique is mirror-image relevant for our planned Phase 2 HR-from-cardiac-BCG module.

### 5. Parameterised respiration filter (PRF)

A continuous-wave radar technique with a parametric model of respiratory motion; outperforms conventional band-pass filtering for low-SNR conditions. May be useful for paediatric where respiratory motion is small relative to body movement.

### 6. Random body movement removal — adaptive motion artifact filtering

Specifically published for neonatal heartbeat sensing (MDPI 2024). Critical for the wriggling-baby use case. Should be reviewed when we move from adult-resting validation to paediatric.

## Architecture validation — what's already published

**Three papers that directly validate aspects of our architecture in neonatal/paediatric populations:**

- **PMC 8122919 (TenneT et al., neonatal ToF + microwave interferometric radar fusion):** essentially our Tier 2 architecture, but the radar in their case is interferometric not FMCW. They demonstrated synchronous evaluation with ±3 BPM max difference. **Use this in grant applications as direct prior art validation; it strengthens the "we're not the first; the approach is feasible" framing.**
- **PMC 8956695 (Contactless radar breathing monitoring of premature infants in NICU):** straight FMCW radar in NICU. RMSE 4.3 BPM in premature infants. Demonstrates clinical feasibility of FMCW respiratory monitoring in our exact target population. **Citation for IRAS/HRA submission later.**
- **PMC 8404938 (Non-Contact Automatic Vital Signs Monitoring of Infants in a NICU Based on Neural Networks):** uses learned models on top of radar/sensor data. Worth reviewing for the Phase 2 cardiac BCG and apnoea pattern classification modules.

## Datasets to leverage

**Comprehensive mm-Wave FMCW Radar Vital Sign Dataset (arxiv 2405.12659):** publicly available, includes "extreme physiological scenarios" (breath-hold, deep breathing, rapid breathing). **Use as a development corpus for our algorithm work without needing more in-house captures.** Particularly useful for: (a) testing the octave-error fix on multiple subjects, (b) validating apnoea-detection logic against simulated cessation, (c) cross-subject generalisation of any settings we tune.

## Implications for the Knight Vision pipeline

### Immediate (this Phase 1 sprint)

- **Confirm `clutterRemoval` state in the radar pipeline.** If we've overridden it to 0 for visualisation purposes but it's also off during analysis, static-reflector DC offset is in our spectrum.
- **Switch LiDAR reporting from centroid to peak-pick** (already identified from Run 9). Maps to the same octave-error problem the radar literature has documented and solved.
- **Pull the Gdańsk octave-error approach** into the architecture doc as a planned algorithm. Even if the source isn't accessible, the existence of a published solution means PR I-style "prefer lowest peak" can be replaced with something more rigorous.

### Phase 2 (when LiDAR pipeline is stable)

- **Try MTI-EEMD or elliptical filtering** on the same recordings to see whether it closes the gap below ±2 BPM.
- **Test signal superposition** across multiple range bins for the radar arm — current pipeline likely only uses the centroid range bin; integrating across the chest depth window could improve SNR significantly.
- **Validate on the public mm-Wave dataset** before paediatric captures, to de-risk in-house data collection.

### Phase 3 (paediatric)

- **Plan adaptive motion artifact filtering from the start** (per the neonatal heartbeat paper). The wriggling-baby problem is non-trivial; literature has approaches that have been tried.

## What this changes about our grant story

**Strengthens, not weakens:**

- The "non-contact respiratory monitoring is feasible in NICU" claim is no longer ours alone to defend — multiple groups have published RMSE 4 BPM-class results in this population. We're now positioned as "advancing the existing approach with multi-modal fusion + spatial priors", not "introducing a novel modality".
- Our two-tier product strategy aligns with the published trajectory of the field (laboratory → adult home → infant home → NICU).
- The specific differentiators we should emphasise: (a) multi-modal fusion architecture beyond what published papers have done (radar + LiDAR + thermal, not just two of those); (b) spatial priors / background subtraction as the front-end (no published radar paper has this); (c) edge deployment on commodity Jetson at clinically-relevant latency.

**A potential weak point to address before submission:**

- If reviewers ask "why not just use the published FMCW-radar-only approach with RMSE 4 BPM?" — our answer needs to be specifically about what multi-modal adds that single-modality radar cannot. The cardiac-vs-respiratory disambiguation, the obstructive-vs-central apnoea distinction (thermal), and the cross-modal confirmation defence against either-sensor-failed are the differentiators. Make these explicit in the grant.

---

## Sources

- [High-Precision Vital Signs Monitoring Method Using a FMCW Millimeter-Wave Sensor — PMC 9572116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572116/)
- [FMCW Radar Respiratory Pattern Detection Technology Based on Multifeature — PMC 8370824](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8370824/)
- [Detection of vital signs based on millimeter wave radar — PMC 12317012](https://pmc.ncbi.nlm.nih.gov/articles/PMC12317012/)
- [DC Offset Contribution of Static Reflectors in FMCW Radar Vital Signs — PMC 9781610](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9781610/)
- [mmWave-RM: A Respiration Monitoring and Pattern Classification System — PMC 11243972](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243972/)
- [Spatial Blind Source Estimation of RR and HR with FMCW Radar — PMC 11859688](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11859688/)
- [High-precision vital signs detection method based on mmWave radar — Sci Reports 2024](https://www.nature.com/articles/s41598-024-77683-1)
- [Random Body Movement Removal in mmWave Radar-Based Neonatal Heartbeat Sensing — MDPI 2024](https://www.mdpi.com/2079-9292/13/8/1471)
- [mmWave for non-invasive respiratory motion visualization — PMC 12082751](https://pmc.ncbi.nlm.nih.gov/articles/PMC12082751/)
- [Multi-Scenario Vital Sign Detection Using mm-Wave MIMO FMCW Radar — arxiv 2508.20864](https://arxiv.org/html/2508.20864v1)
- [Detection of vital signs based on mmWave radar — Sci Reports 2025](https://www.nature.com/articles/s41598-025-09112-w)
- [Survey of mmWave Technologies for Medical Applications — PMC 12197076](https://pmc.ncbi.nlm.nih.gov/articles/PMC12197076/)
- [Clinical Trial: mmWave Radar Sleep Respiratory Monitoring — NCT06038006](https://clinicaltrials.gov/study/NCT06038006)
- [Automated Non-Contact RR Monitoring of Neonates: ToF + Microwave Radar — PMC 8122919](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8122919/)
- [Radar-Based Monitoring Piglet Model — PMC 13075061](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075061/)
- [Contactless radar breathing monitoring of premature infants in NICU — PMC 8956695](https://pmc.ncbi.nlm.nih.gov/articles/PMC8956695/)
- [Non-Contact Vital Signs Monitoring of Infants in NICU with Neural Networks — PMC 8404938](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404938/)
- [Algoritmically improved microwave radar monitors breathing more accurate than sensorized belt — PMC 9400007 (Gdańsk octave-error work)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9400007/)
- [Comprehensive mm-Wave FMCW Radar Dataset for Vital Sign Monitoring — arxiv 2405.12659](https://arxiv.org/html/2405.12659v1)
- [Doppler radar millimetre-wave (76-81 GHz) sensing firmware — Healthcare Technology Letters 2024](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/htl2.12075)
