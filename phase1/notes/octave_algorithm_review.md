# Octave-error algorithm — PMC 9400007 synopsis

*Task #78 (2026-05-19). Synopsis of Czyżewski et al., "Algorithmically improved microwave radar monitors breathing more accurately than a sensorized belt", Scientific Reports 12:14412 (2022).*

---

## TL;DR

The Gdańsk algorithm **is reimplementable from the paper alone for our pipeline**. The proprietary part is TI's VitalSigns demodulation (raw radar phase → respiratory waveform); we already have an equivalent demodulated waveform from LiDAR (`centroid_z`), so we skip that step entirely. The 3-stage octave-correction logic on top of the waveform is fully specified in pseudocode + flowcharts. Our **20260517_071225** capture (peak-pick locked onto 26.6 BPM vs Polar 12.2 — see task #77 re-score) is a canonical test input for any implementation.

The algorithm works in the **time domain on the autocorrelation function**, not the FFT spectrum we currently use. This is a structural switch from peak-pick / centroid, not a tweak.

---

## The problem (matches our 071225 failure exactly)

Fundamental-frequency estimation from a quasi-periodic respiratory waveform can lock onto the **2× harmonic** rather than the fundamental: subject breathes at 12 BPM, detector reports 24 BPM, reproducibly. They name this **octave error**. Cause in their pipeline: the second-largest autocorrelation peak — which they use as the fundamental-period estimator — is sometimes the *half-period* peak instead, biasing the estimate by exactly 2×.

In our pipeline the failure mode is the same but in the FFT spectrum: harmonic energy at 2×f₀ exceeds the fundamental, peak-pick lands on the harmonic. Both manifestations stem from "the strongest spectral/autocorrelation feature isn't necessarily the fundamental".

The radar literature treats this as a *named, structural* failure mode of FMCW vital-signs estimation. We're not alone in hitting it.

---

## Algorithm — 3 stages on an autocorrelation trajectory

**Pre-conditions.** A demodulated respiratory waveform sampled at a uniform rate. They resample to 100 Hz; the absolute rate doesn't matter as long as it's consistent. Frame = 4,096 samples (≈ 41 s at 100 Hz), overlap 0.85 (≈ 6.1 s step). Per-frame autocorrelation is the core signal.

### Stage 1 — Naïve estimator + heuristic voting

- For each frame: compute autocorrelation of the demodulated waveform.
- Find all local maxima (differentiate, sign-change).
- **Pick the second-largest maximum by Euclidean distance from the central peak (lag=0 maximum).** This is the candidate fundamental-period peak.
- Convert lag-index → frequency.
- **Voting check:** if the current frame's estimate is **≥ 1.8×** the value seen in **3 of the preceding 5** frames, flag as octave-error and replace with the closest preceding non-flagged estimate.

This catches the obvious 2× jumps frame-to-frame but doesn't help if the algorithm starts in an octave-error state.

### Stage 2 — K-means outlier elimination across the trajectory

- Collect the per-frame second-maximum-index trajectory (the time-series of candidate fundamental lags).
- Build a 4-feature vector per frame: (peak index, distance to left neighbour, distance to right neighbour, estimated fundamental frequency).
- Run **k-means with k=2** (sklearn default). Assume the smaller cluster = outliers.
- Reassign each outlier leftward to the **nearest valid peak from the preceding frame**.
- Iterate until every outlier's predecessor is non-outlier and nearest-neighbour-coherent.

This globally cleans up the trajectory under the assumption that "most frames are right, some are wrong" — fragile if octave-error dominates a recording.

### Stage 3 — Sub-path splitting + greedy merging

- Split the trajectory at every frame where the chosen maximum is NOT the nearest neighbour of the previous frame's chosen maximum.
- For each pair of adjacent sub-segments (P_n, P_n+1), compute an alternative path that hops nearest-neighbour through both, keep the **shorter** of (original concatenation, alternative).
- Greedy left-to-right merge until no further shortening.

The output is a smooth-in-lag-space trajectory of fundamental-period estimates. Convert to BPM as the final step.

---

## Validation in the paper

- **31 healthy adults**, paired with a TN1132/ST respiratory belt + ADInstruments PowerLab at 1 kHz reference.
- Three conditions: 10 min free breathing; 5 min paced @ 12 BPM; 5 min paced @ 15 BPM.
- **Paced 12:** radar median 11.79 BPM, belt 11.92 BPM, Δ = 0.13 (not significant).
- **Paced 15:** radar 14.71, belt 14.85, Δ = 0.14 (statistically significant, practically negligible).
- The naïve Euclidean stage alone matched the medians but had higher variance; the voting/k-means/merging cut variance.
- DTW distance between paired radar+belt traces on the *same person* (43.99) was an order of magnitude smaller than between different people on the same device — the residual variance is real person-to-person breathing variability, not sensor noise.

Their hardware: TI IWR6843AOP (same family as our radar), 60-64 GHz, raw frame rate 30 Hz, Pi 4 host, on top of TI VitalSigns firmware. The published reference RR accuracy is **~0.13 BPM bias at adult rest** — better than our current peak-pick (~|Δ| median 1.16 across the four clean 05-17 runs).

---

## Reimplementation feasibility for Knight Vision

**YES, reimplementable from the paper alone for our pipeline.** Three reasons:

1. **The proprietary part is upstream of where we'd plug in.** The closed-source TI VitalSigns firmware converts raw FMCW phase/amplitude → demodulated respiratory waveform. We don't use TI VitalSigns; we have our own demodulation chain producing `centroid_z`. The Gdańsk algorithm operates *on the demodulated waveform*, downstream of where the IP boundary sits.
2. **All three stages have pseudocode or flowcharts in the paper.** Parameters (frame length 4,096, overlap 0.85, voting threshold 1.8×, voting window 5 frames, k=2 clusters) are explicit.
3. **Only sklearn + scipy** required — already in `phase1/requirements.txt`. K-means init / random seed not specified, but `KMeans(n_clusters=2, n_init=10)` is a defensible default.

**Caveats / gaps:**

- Voting-classifier threshold (1.8×) is empirically chosen, not justified. May need tuning for our sample rate / frame length.
- They don't say *exactly* whether the 3-stage algorithm runs on the radar's DSP or in the Python host. Context (sklearn, k-means, post-hoc trajectory analysis) strongly implies host-side. Either way, host-side is what we'd implement.
- Frame length of 41 s is much longer than our 20 s sliding-window FFT. For sub-30-second apnoea responsiveness we'd need to shrink the frame (probably losing some accuracy) and re-validate the voting/k-means parameters. **Frame length is the load-bearing parameter to re-tune.**
- The algorithm operates on a *single* (autocorrelation-trajectory) modality. Our multi-modal architecture eventually gates RR estimates on radar + LiDAR agreement — Gdańsk's output is just one of several inputs to that gate, not the whole answer.

**NDA-only parts we'd skip:** the TI firmware-integrated version (mentioned in our radar lit review as "integrated into TI's reference firmware"). That's a separate artifact from the published algorithm — same authors, same idea, but implementation details under NDA. Re-implementing the published version is the path.

---

## Recommended next step

1. **Implement on `centroid_z.npy` first**, before touching the radar arm. The LiDAR side is more stable for offline testing and the canonical 071225 test case lives there.
2. **Validate on 20260517_071225** as the primary test case — known octave-error failure (peak-pick 26.6 BPM, Polar 12.2). Success = the corrected estimator returns ≤ 14 BPM with HIGH or MEDIUM confidence.
3. **Run across the other five 05-17 captures** as a regression: success = doesn't break the four clean-front-end runs (mean peak-pick Δ -0.99 BPM today; should not degrade by more than ~0.5 BPM).
4. **Co-locate with peak-pick / centroid in `respiratory.py`** as a third `rr_method = "gdansk-autocorr"` option, alongside the existing two. Same interface, returns the same dict — keeps the dual-method viewer infrastructure (PR Z) working.
5. **Decision after step 3:** if it salvages 071225 without regressing the clean runs, promote to a real PR. If it regresses the clean runs by more than peak-pick's bias, queue behind task #62 (HR notch on centroid) — the notch fixes the algorithmic bias on the runs where Gdańsk doesn't help.

Phase 2 candidate, not Phase 1 blocker. Realistic implementation: **150-200 lines**, plus tests against the 6 paired captures.

---

## Reference

Czyżewski A, Kostek B, Kurowski A, Narkiewicz K, Graff B, Odya P, Śmiałkowski T, Sroczyński A. *Algorithmically improved microwave radar monitors breathing more accurate than a sensorized belt.* Scientific Reports 12, 14412 (2022). DOI: [10.1038/s41598-022-18808-2](https://doi.org/10.1038/s41598-022-18808-2). PMC: [PMC9400007](https://pmc.ncbi.nlm.nih.gov/articles/PMC9400007/).
