# Next session — pickup

## First task: PR J — sliding-window FFT

Pipeline currently FFTs the cumulative centroid_z series (all frames
since the recording started). Per-window estimates therefore drift
slowly: a real change in breathing pattern (apnoea, rate shift) takes
~20 s before the new data dilutes the old enough for the centroid to
move. Run 5 (20260516_170653, breath-hold) showed this clearly — 56 s
apnoea visible only as a delayed upward drift.

### Blocker — window size, decide before coding

Three viable options. Each has a real trade-off. Don't punt.

**20 s window with 75 % overlap (starting recommendation)**
- New FFT every 5 s, each over the most recent 20 s of cz
- Bin spacing at 10 fps: ~3 BPM (acceptable for 10–30 BPM band)
- Apnoea visible within ~10 s of onset (half-window in)
- Recovers in ~10 s after breath restart
- Reasonable compromise; matches HRV literature defaults

**30 s window**
- Safer — more samples, better SNR, finer bin spacing (~2 BPM)
- But still 15–20 s detection lag — barely improves on the cumulative
  approach for apnoea. Not justified given the goal.

**15 s window**
- Tightest, ~5 s detection lag
- But bin spacing at 10 fps is ~4 BPM. With a true RR of 13 BPM,
  nearest bins are 12 and 16 BPM — frequency resolution worse than
  the inter-subject RR variation we're trying to measure.
- Workaround would be zero-pad more aggressively, but that doesn't
  add real information and risks false centroid stability.

**Recommendation: 20 s / 75 % overlap.** Confirm by reasoning through
whether 5 s update cadence vs Polar's 2 s feels live enough in the
viewer, AND whether 3 BPM bin spacing is OK for the M1 paired-Δ
target (we've been getting Δ ≈ 0.5–2 BPM, the bin spacing would
become the floor).

Document the choice in the dev log when you ship it.

### Implementation notes

- In `phase1/pipeline.py`, the cumulative FFT is in `run_live()` at the
  `if (i + 1) % compute_rr_every == 0` block. Change `sig = np.array
  (self._centroid_z, dtype=float)` to take only the last
  `int(window_seconds * actual_fps)` samples.
- Set `compute_rr_every` from the overlap rule: `int(window_seconds *
  (1 − overlap) * actual_fps)`.
- Both window_seconds and overlap belong in `config.py` (new fields,
  defaulted to whatever you decide).
- Each window now has a definite time anchor — `meta.json`'s
  `rr_windows` should record `t_centre` for each entry so future
  apnoea analysis can align with manual phase markers.
- Settled-median still applies on top; reconsider whether trailing-only
  is still the right rule once windows are sharper (it may become
  unnecessary — the early sway transient should drop out of the window
  within 30 s).

## Carry-over items

- **PR D** (no-signal indicator) — shipped, but worth verifying it
  fires correctly during apnoea once PR J makes the SNR drop sharper.
- **PR G** (longest-settled-subsequence) — was deferred. With PR J in,
  per-window stability is real-time, so the "settled trailing window"
  rule may not need replacing. Revisit after a couple of PR J test runs.
- **PR5** (band-confidence weighting) — Phase 2. Less urgent now that
  HF-centroid + tight band 10–30 BPM gives credible results.
- **Multi-subject tracking** — Phase 2. See `20260516_082351` run for
  the two-person residual signature.
- **Confidence-threshold recalibration** — need N ≥ 5 paired runs
  against Polar before tuning empirically; don't guess.
- **`~/knight-vision-mac/`** — the Mac Polar bridge venv + script live
  here, not in the Jetson repo. If you switch Macs or rebuild, this is
  the recreation path: `python3 -m venv venv && ./venv/bin/pip install
  bleak numpy`, then copy `polar_bridge.py`. Not currently version-
  controlled — decide whether to make a sibling repo or pull into
  knight-vision.

## State of working tree

Check `git status` at session open. Today's tree was clean after the
end-of-day commit (see commit message for the run-down).
