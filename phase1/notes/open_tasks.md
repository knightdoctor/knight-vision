# Open tasks — carried from 2026-06-23

Two items carried forward from the Week-2 B1 milestone (commit
`3d19498`). Both block any amplitude-based simulation work.

---

## 1. PCA auto-detect breathing axis (pipeline fix, NOT sim) — PRIORITY

**Prerequisite for all amplitude work.** Cannot meaningfully run the
Optuna geometry sweep (#4), voxel-downsample tuning (#3), or
cessation-arm validation until this lands.

**Symptom.** Three axes (shared-Z, shared-Y, Cam-A depth) all give
different amplitude recovery ratios on B1, none faithful to the
underlying chest motion. shared-Z is geometry-orthogonal in the B1
layout (ideal std = 0). shared-Y and Cam-A depth are noise-dominated
(ratios 1.46× to 43× across the three texture variants — always >> 1).
The smoke test currently hardcodes shared-Z, which was wrong for the
side-on Week-1 geometry and is even more wrong for any pitched-camera
configuration.

**Right path.** PCA on the per-frame chest-cluster centroid to detect
the dominant oscillation axis automatically. Project the cz signal
onto that axis. Geometry-agnostic — works regardless of cam mounting
angle, baseline orientation, or scene layout.

**Why this is a pipeline fix, not a sim fix.** The Phase-1 LiDAR
pipeline ingests sensor-depth and runs the same shared-Z assumption
in `extract_rr_from_signal`. Fixing in the sim alone (by manually
choosing shared-Y for B1) papers over the underlying issue and won't
generalise to real hardware mounting variations.

---

## 2. Cessation-arm validation gate — BLOCKED by #1

**Status.** Cannot be honestly tested on synthetic data while the
centroid signal is noise-dominated.

**Numbers from B1 (2026-06-23):**
- shared-Y noise inflation: 1.46× (control) → 4.04× (projector)
- Cam-A depth noise inflation: 15× → 43×
- All ratios >> 1: measurement bigger than the forward-modelled ideal
  centroid signal in every case

**Implication for the 0.12 mm absolute threshold.** The threshold was
derived against a real sensor signal of unknown axis-convention. Until
#1 lands and the synthetic signal is faithful to the same axis, the
0.12 mm gate cannot be evaluated on synthetic data without risking
either false-positives (sim noise floor sits above 0.12 mm and triggers
cessation during normal breathing) or false-negatives (sim signal is
amplitude-attenuated and never crosses the threshold).

**Re-evaluate after #1 lands.**

---

## DO NOT run yet

- **#3 Voxel-downsample tuning.** Picks a voxel size based on its
  effect on the cz signal. With the current noise-dominated cz, any
  pick optimises against noise, not signal.
- **#4 Optuna geometry sweep.** Objective `0.5·RR-MAE + 0.3·SNR + 0.2·
  settled-window-fraction`. The SNR and settled-window terms ride on
  the same noise-dominated signal. Sweeping now would optimise for
  whatever geometry happens to produce the largest noise.

Both wait until #1 closes.

---

## What CAN proceed in parallel

- Pipeline-side PCA-axis implementation (this is the #1 work).
- Scene-quality work that doesn't depend on amplitude: e.g. swapping
  the capsule torso for a Sketchfab infant mesh (per Week-1 results
  doc §"Known issues" #5) — improves visual realism without
  interacting with the amplitude question.
- Hardware procurement / Jetson bring-up for real-sensor validation.

---

## Reference

- Milestone commit: `3d19498` (B1 L-overhang + matrix-bug fix, 2026-06-23)
- Underlying-bug walk-through: `phase1/notes/dev_log_2026-06-23.md`
- Carried-items history: `phase1/notes/dev_log_2026-06-22.md`
