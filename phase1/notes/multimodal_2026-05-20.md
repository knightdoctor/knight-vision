# Multi-modal validation — 2026-05-20 session

**Goal:** build N from 4 toward 10 strict M1-grade paired captures + execute task #64 (HR-variation validation for the cardiac BCG hypothesis).

**Hardware state change:** IWR6843AOPEVM radar physically mounted on top of the Femto Bolt; they now move together with a fixed relative offset. Position-variability between modalities removed from this session forward.

**Pipeline config (all runs unless noted):** post-PR Z default; `--rr-method peak-pick`; fresh background via `bootstrap.sh --recapture-bg`; `--save-raw` (raw depth) + `--record-radar` (radar sidecar → `radar.npz` in each run dir). LiDAR analysis band 10-30 BPM. Subject lock + chest-band (PR S/T/X/Y) + settled-median (PR M) all default.

## Run plan

| Run | Protocol | Duration | Purpose |
|---|---|---|---|
| α | Sitting in chair, normal breathing | 60 s | Baseline under fixed-offset rig |
| β | Same chair, same posture — **immediately after α** (≤ 5 min) | 60 s | Inter-run reproducibility |
| γ | Lying on sofa, normal breathing | 60-90 s | Pose-sensitivity (geometry change, RR unchanged) |
| δ | **Task #64:** 30 s moderate exercise → sit → record while HR drops 100 → 70 | 120-180 s | HR/2 cardiac-BCG validation |

For each run Polar bridge is the GT; manual `gt.csv` markers as needed (Run δ: `hr=high` at start, `hr=normal` when subject feels recovered).

## Runbook (Jetson)

The session is **one pipeline launch** for all four runs. Recordings are triggered from the viewer; each REC press writes its own `phase1/runs/<ts>/`. `bootstrap.sh` now bakes M1-grade defaults (`--viewer --rr-method peak-pick --save-raw --record-radar --duration 1800`).

```bash
# Terminal 1 — Polar bridge for the whole session:
phase1/run.sh phase1/polar_bridge.py --duration 1800

# Terminal 2 — session pipeline (background already cached this evening;
# add --recapture-bg only if room state changes):
phase1/bootstrap.sh
```

Viewer at **http://phil-desktop.local:5005** (or `http://192.168.1.90:5005`). Inside the viewer: `R` / REC button = start recording, `S` / STOP = end recording. Per-run protocols:

- **α** seated chair, normal breathing, ~60 s. R → 60 s → S.
- **β** same chair, same posture, ≤ 5 min after α. R → 60 s → S.
- **γ** lying on sofa, ~90 s. R → 90 s → S.
- **δ** (task #64) — moderate exercise OFF-camera (~30 s jumping jacks / stairs) → sit at chair → R immediately → 120-180 s while HR drops → S when HR feels recovered. Manual gt.csv markers: `hr=high` at recording start, `hr=normal` when recovered.

After each REC/STOP, append the per-run block:
```bash
phase1/run.sh phase1/per_run_log.py phase1/runs/<run_id> \
    --label "Run α" --append phase1/notes/multimodal_2026-05-20.md
```

After δ, also run the HR/2 retrospective:
```bash
phase1/run.sh phase1/hr2_tracking.py phase1/runs/<delta_run_id> \
    --out phase1/notes/hr2_tracking_<delta_run_id>.md
```

## Per-run logs

*(Appended by `phase1/per_run_log.py` after each capture lands.)*

<!-- per-run blocks land below this line -->

## Session-level questions (filled at end)

- α-vs-β Δ: ≤ 1 BPM ⇒ rig stable; > 1 BPM ⇒ residual run-level variability worth tracking.
- γ vs α: pose change should not move RR more than breath-to-breath. Method Δ (centroid − peak-pick) may shift if cardiac BCG amplitude changes with posture.
- δ HR/2 tracking: confirmed if `r²` ≥ 0.7 and median |peak − HR/2| ≤ 3 BPM in `phase1/notes/hr2_tracking_<run_id>.md`.

## Files added for this session

- `phase1/per_run_log.py` — per-run multimodal block emitter
- `phase1/hr2_tracking.py` — task #64 HR/2 cardiac-tracking analysis
- `phase1/notes/multimodal_2026-05-20.md` — this file
