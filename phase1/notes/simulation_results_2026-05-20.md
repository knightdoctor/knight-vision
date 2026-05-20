# MVP simulation — Week 1 smoke test results

*Generated 2026-05-20. Companion to `mvp_simulation_spec_2026-05-19.md` (iCloud) and the `mvp-simulation` feature branch.*

## Headline

**PASS.** Synthetic 15 BPM breathing rendered through Blender stereo IR, recovered through OpenCV StereoSGBM depth + chest centroid + the existing `extract_rr_from_signal` pipeline, lands at **15.02 BPM** (Δ +0.016, well inside the ±2 BPM tolerance). The Phase 1 algorithm stack ports to passive stereo IR depth without modification.

```
✅ smoke test recovered 15.02 BPM (gt 15.0, Δ +0.02) · SNR 4.30 · MEDIUM · cz_std 0.069 mm
```

## Config that passed

- Scene: capsule torso (30 cm length, 8 cm radius) + sphere head + cot frame + IR LED + ambient fill.
- Cameras: vertical-baseline 200 mm stereo, look-at chest from Y = −0.40 m (40 cm working distance), both pointed at chest centre (Cam A 100 mm above chest Z, Cam B 100 mm below). **Not the MVP L-overhang convergent + pitched geometry** — that's deferred to Week 2 because it needs full `cv2.stereoRectify` and the smoke goal was algorithm-port validation, not geometry validation.
- Breathing: 15 BPM, ±4 mm sinusoidal chest excursion, 20 s duration, 10 fps render.
- Renderer: Eevee (Blender 5.1.2).
- Stereo: `cv2.StereoSGBM_create` with `numDisparities=640` and `blockSize=9`, images CW-rotated before SGBM (vertical baseline → horizontal disparity in rotated frame), input order swapped (B-rot as left, A-rot as right) for positive disparity sign.
- Voxel downsample: **disabled** (`voxel_size_m=0`). See "Known issues" below.
- Pipeline: `extract_rr_from_signal` with `rr_method="peak-pick"`, default KVConfig 10–30 BPM band.

## Validation chain that landed

```
scene_build.py     → scene.blend + intrinsics.json
render.py          → 200 stereo pairs (L/R PNG) + ground_truth.csv
depth.py           → 200 point clouds (Nx3 shared frame, X right Y up Z forward)
smoke_test.py      → centroid_z series → extract_rr_from_signal
                   → assert |recovered − 15| ≤ 2
```

End-to-end runtime on the home Mac: **~5 minutes** (Eevee render ~3 min, depth + SGBM ~1.5 min, pipeline ~1 s).

## Known issues to track for Week 2

### 1. Depth scale error
Recovered mean depth ~0.95 m, true optical distance from Cam A to chest centre is ~0.41 m — a ~2.3× over-estimate. The FFT still finds the right frequency because the *temporal pattern* of the motion is preserved, but the absolute depth is wrong. Likely causes:
- Pitch / look-at geometry making the two cameras non-parallel; `cv2.stereoRectify` would handle this properly.
- The 90° rotation + swap heuristic is correct for sign but the disparity-to-depth conversion `Z = fx·B/d` assumes parallel cameras with a known baseline in the *rectified* frame; ours is convergent.

Fix in Week 2: implement full stereo rectification with `cv2.stereoRectify` from explicit R, T computed from the camera world poses. Re-run smoke and confirm depth lands within 5% of true.

### 2. Motion attenuation
True chest excursion was 8 mm peak-to-peak; the cz_std the pipeline saw was **0.07 mm** — ~100× attenuation. Despite the attenuation, the *temporal frequency* of the motion is preserved, so peak-pick locks on the right BPM. The attenuation is from two sources:
- (a) Stereo block matching averages a 9-pixel window — finer chest motion gets smoothed.
- (b) Without voxel downsampling, the centroid-Z over ~34k chest-box points integrates a lot of background that doesn't move.

Fix in Week 2: tighten the chest box (currently 24×24×40 cm; the actual torso is ~8×30×8 cm). Investigate whether sub-pixel disparity interpolation (`StereoSGBM`'s `disp12MaxDiff` etc.) helps recover the true magnitude.

### 3. Voxel downsampling needs to be 0 for smoke
At the lidar_driver default of 3 cm voxels, the 8 mm chest motion gets quantised into the same voxel cell across frames → the chest centroid is *constant* → the FFT octave-errors onto 2× the true rate (recovered 29.9 BPM on first attempt). Smoke test disables voxel downsampling. This will need real thought for L-overhang work where the lidar_driver point density matters for the downstream pipeline.

Fix in Week 2: figure out the right downsample resolution for stereo depth (likely 5–10 mm rather than 30 mm) and re-run smoke at lidar-parity point density.

### 4. Camera geometry is not the MVP L-overhang
This smoke used parallel vertical-baseline cameras looking straight at the chest. The MVP spec calls for L-overhang convergent cameras with Cam A pitched 30° down, Cam B 10° down, both off-axis. That's a strictly harder stereo problem (non-parallel optical axes need full `cv2.stereoRectify`) and isn't load-bearing for the algorithm-port question this smoke was built to answer.

Fix in Week 2: implement the L-overhang geometry + rectification path, then re-run the smoke. Only then is `optimize.py` (Phase 1 geometry sweep) on solid ground.

### 5. Infant model is a primitive capsule
The Week 1 capsule torso + sphere head is enough to validate that pipeline integration works. It's not realistic enough for the Week 2/3 robustness sweep (clothing colour, blanket coverage, skin tone — none of these are modelled). Swap in a rigged commercial mesh or Sketchfab asset before the robustness sweep.

## What this validates for the project

- **The Phase 1 algorithm stack ports to passive-stereo depth.** The `extract_rr_from_signal` peak-pick + 10–30 BPM band + sliding window setup recovers the right respiratory rate from depth data whose physical characteristics (vertical baseline, 200 mm separation, 40 cm working distance) match the MVP class of design — *without any algorithmic adaptation*.
- **The mvp_sensor_stack_architecture pivot (stereo IR rather than single LiDAR) is plausible from an algorithm standpoint.** If the stereo depth pipeline can recover synthetic 15 BPM at Δ < 0.1 BPM, the algorithm holds up. The remaining work is geometry optimisation (Week 2) and robustness sweep (Week 3).
- **Task #78 (Gdańsk octave-error algorithm) becomes more important, not less.** The voxel-downsampled smoke trial hit the same octave-error failure mode the spec calls out — synthetic data reproduces the issue. Implementing the Gdańsk algorithm validates twice over (real radar + synthetic stereo).

## Branch / commit state

- `mvp-simulation` branch — Week 1 deliverables (scene_build.py, render.py, depth.py, smoke_test.py, run_smoke.sh, README.md, this results note).
- Merge to `main` per spec ("Once it does, merge").
- Push to origin deferred per the standing tonight rule unless explicitly approved.

## Next actions (Week 2)

1. **Full stereo rectification** for arbitrary convergent + pitched cameras using `cv2.stereoRectify` + `initUndistortRectifyMap`. Re-validate scale recovery to within 5%.
2. **L-overhang geometry** in `scene_build.py` matching the MVP spec (Cam A 30° down, Cam B 10° down). Re-run smoke against the geometry the production hardware will use.
3. **Voxel-downsample tuning** — find the right resolution that preserves chest motion *and* matches the lidar_driver point density downstream.
4. **`optimize.py`** — Bayesian (Optuna) sweep over 6-parameter geometry, 500 trials, ~1.5 h wall-clock on the Mac.
