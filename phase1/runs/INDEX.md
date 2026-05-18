# phase1/runs — INDEX

One row per significant capture. Skip demo-mode entries unless they're
diagnostic. **Tag** column: `[M1]` = paired vs ground truth, `[diag]` =
debugging/exploration, `[bad-bg]` = run captured with the synthetic-
contaminated background (results not usable).

## Convention

`run-dir | when | tag | subject / conditions | GT source | LiDAR | GT | Δ | note`

LiDAR/GT values are the run's *final* estimate (settled-median where
available). Δ = `|LiDAR − GT|` for paired runs. SNR/conf in note column
when relevant.

---

## Yesterday — 2026-05-15 (Garmin-GT era, all retracted)

All yesterday's "paired" runs are retracted: pipeline ran at varying
fps (2-8) but FFT was hardcoded to 10 fps → BPM systematically biased.
PR B (measured-fps fix) deployed today. See `notes/dev_log_2026-05-16.md`.

| run | when | tag | conditions | GT | LiDAR | GT | Δ | note |
|---|---|---|---|---|---|---|---|---|
| `20260515_140302` | 14:03 | [diag] | early bring-up, --duration 10 | none | 58.0 | – | – | fps actual 2.4; no GT |
| `20260515_141350` | 14:13 | [diag] | bring-up | none | 8.6 | – | – | fps actual 2.8 |
| `20260515_142311` | 14:23 | [diag] | sway-only test (no subject) | none | 6.0 | – | – | fps actual 7.6; bogus 17 SNR — was the 6-BPM sway artifact, hint of today's chest-cz issue |
| `20260515_212728` | 21:27 | [M1-RETRACTED] | Phil on sofa, lit | Garmin manual ~10 | 9.96 | 10 | (coincidence) | fps actual 7.6; true BPM ≈ 7.6 with correct fps |
| `20260515_214726` | 21:47 | [M1-RETRACTED] | Phil on sofa, dark | Garmin manual ~12 | 12.16 | 12 | (coincidence) | fps actual 3.95; true BPM ≈ 4.8 — Garmin "match" was coincidence |

## This morning — 2026-05-16 (chasing the algorithm bugs)

| run | when | tag | conditions | GT | LiDAR | GT | Δ | note |
|---|---|---|---|---|---|---|---|---|
| `20260516_081815` | 08:18 | [M1-RETRACTED] | Phil on sofa, fresh bg | Garmin guesstimate ~10 | 10.1 | ~10 | – | "**Run #3**". Was claimed as first paired but harmonic_diagnostic.py later proved the 10.1 was algorithm grabbing a narrow band edge — no real 10 BPM peak. Useful diagnostic data; not M1. |
| `20260516_082351` | 08:23 | [diag] | **two persons on sofa** | none | 29.9 | – | – | DBSCAN merged subjects → BPM ≈ sum of two breath rates. Phase 2 multi-subject task spawned. |
| `20260516_085752` | 08:57 | [diag] | preview-mode test, partial | none | 8.6 | – | – | First test of preview/record split + record-toggle (PR1) |
| `20260516_090722` | 09:07 | [diag] | GT endpoint smoke test | curl | 11.3 | – | – | Verified POST /gt + gt.csv write path |
| `20260516_101855` | 10:18 | [diag] | layout test, kid in frame | none | 0.0 | – | – | "No breathing detected" state (PR D) tested. Kid running through frame confused subject selection. |

## Polar-GT era — 2026-05-16 (real ground truth)

Polar H10 paired with Mac CoreBluetooth bridge from 16:49 onwards.
Earlier Polar runs used Jetson bleak daemon (stall issues, retracted).

| run | when | tag | conditions | GT | LiDAR | GT | Δ | note |
|---|---|---|---|---|---|---|---|---|
| `20260516_110954` | 11:09 | [bad-bg] | first Polar pairing attempt | Polar (Jetson daemon) | 6.04 | – | – | bg contaminated by demo at 08:43; resid ≈ frame pts; LiDAR locked on a NaN-mean-fill artifact reading "HIGH SETTLED". Demonstrated the danger of high SNR on bad input. |
| `20260516_160507` | 16:05 | [bad-bg] | post-Polar-fix attempt | Polar (Jetson daemon) | 6.06 | 7.27 | – | Same bad-bg problem. The 6 vs 7 "agreement" was both wrong. |
| `20260516_161947` | 16:19 | [bad-RR-algo] | **post-bg-recapture**, Polar (Jetson daemon) | Polar naive-RSA | 13.35 | 7.27 | – | bg now clean (resid 2.5k of 14k). Pipeline gave correct 13 BPM. Polar's naive RSA-FFT daemon stuck on 6-7 BPM (LF baroreflex peak, not respiratory). Demonstrated Polar algorithm needs improvement. |
| `20260516_164903` | 16:49 | [diag] | first Mac Polar bridge + HF-centroid GT | Polar (Mac, HF-centroid) | 6.11 | 12-14 | ~7 | Polar GT now solid (matches Phil's stated ~15 BPM breathing). LiDAR pipeline STILL locked on 6 BPM artifact — exposed it as a real algorithm bug, not a GT problem. PR I (HF-centroid on LiDAR side) implemented. |
| `20260516_165647` | 16:56 | **[M1]** | **first valid paired**, post PR I | Polar (Mac, HF-centroid) | **14.5** (last win) / 16.0 (settled) | **14.1** | **0.4 / 1.9** | Phil seated closer to sensor, normal breathing. First time LiDAR and Polar both report physiological RR in agreement. M1 #1. |
| `20260516_170653` | 17:06 | **[M1]** | **Run 5: breath-hold protocol** | Polar (Mac) + manual phase markers | 14.6 (settled over recording) | 14.1 (steady-state pre + post) | 0.5 | 30s normal → 56s breath hold (manual rr=0 at start, rr=15 at end) → 30s normal. Apnoea detected by **two signatures**: LiDAR BPM destabilises UP to 22-24 (centroid drifts when respiratory peak vanishes) + Polar HR drops 77→64 (vagal bradycardia). ~20s detection lag → PR J. See `protocols/breath_hold_validation.md`. |
| `20260518_213754` | 21:37 | **[M1]** | **post-PR-T/V/X**: chest band widened (0.30, 0.75), sticky tracking + size-dominance, side-elevation viz, panels tightened, monitoring volume Z=1.4m | Polar (Mac, HF-centroid, n=42) | **17.0** peak / 19.4 centroid (SNR 6.10) | **14.22** median (11.6-16.1) | **+2.8** peak / +5.2 centroid | 81s @ 9.2 fps, 100% valid frames, ptp chest displacement 28.6mm. First paired run with chest+abdomen band. Peak-pick is the headline agreement (within 2.8 BPM of Polar) — best within-subject match this session. Centroid bias persists. Full analysis in `notes/multimodal_2026-05-18.md` evening-session section. |

---

## Headline counts

- **Valid M1 paired runs: 3** (Run 4 = `20260516_165647`, Run 5 = `20260516_170653`, Run 6 = `20260518_213754`)
- **Retracted: 5** (all due to fps mismatch or bad-bg)
- **Diagnostic-only: 7** (informative but no GT)

Target: N ≥ 10 valid M1 readings by 2026-05-31.
