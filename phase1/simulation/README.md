# MVP geometry simulation — phase1/simulation

Companion to `mvp_simulation_spec_2026-05-19.md` (iCloud Knight Vision folder).

**Scope (Week 1):** prove the existing `phase1` algorithm stack ports to passive stereo IR by running synthetic 15-BPM breathing through the full pipeline and checking the recovered RR matches ±2 BPM.

**Out of scope (Week 2/3):** Optuna geometry sweep, robustness condition sweep, rigged commercial infant mesh, thermal modality, IR-bandpass simulation. Those land once the smoke test passes.

## Architecture

```
scene_build.py   →   scene.blend      (parameterised infant + cot + 2 cameras + IR LED)
render.py        →   frames/L/*.png   (left camera)
                     frames/R/*.png   (right camera)
                     intrinsics.json
                     ground_truth.csv (breathing phase per frame)
depth.py         →   frames/depth/*.npy or point cloud Nx3 array
                     (output format matches lidar_driver.py — shared frame
                      X right, Y up, Z forward, metres)
smoke_test.py    →   orchestrates the above, runs the synthetic point clouds
                     through the existing extract_rr_from_signal, asserts
                     recovered BPM within 15 ± 2 BPM.
```

## Toolchain

Blender 5.1.2 — uses the bundled Python (3.13.9) which has bpy + numpy + cv2 baked in. Invoke any bpy script via:

```
/Applications/Blender.app/Contents/MacOS/Blender --background --python <script.py> -- <script-args>
```

`depth.py` and `smoke_test.py` don't need bpy and run under any Python that has numpy + scipy + opencv-python.

## Run order (Week 1 smoke test)

```bash
cd ~/knight/knight-vision
phase1/simulation/run_smoke.sh    # wraps the four steps below
```

Or manually:

```bash
# 1. Build scene (idempotent — regenerates scene.blend from scratch)
/Applications/Blender.app/Contents/MacOS/Blender --background --python \
    phase1/simulation/scene_build.py -- \
    --out phase1/simulation/scene.blend

# 2. Render 30 s of synthetic 15-BPM breathing at 15 fps (450 frames per camera)
/Applications/Blender.app/Contents/MacOS/Blender --background --python \
    phase1/simulation/render.py -- \
    --scene phase1/simulation/scene.blend \
    --rr-bpm 15 --amplitude-mm 5 --duration 30 --fps 15 \
    --out phase1/simulation/frames

# 3. Compute depth maps (and point clouds) from rendered stereo pairs
.venv-local/bin/python phase1/simulation/depth.py \
    --frames phase1/simulation/frames \
    --out phase1/simulation/depth_out

# 4. Smoke test — feed depth-derived centroid_z into extract_rr_from_signal
.venv-local/bin/python phase1/simulation/smoke_test.py \
    --depth-dir phase1/simulation/depth_out \
    --ground-truth-bpm 15 --tolerance 2
```

## Status

- [x] `scene_build.py` — parametric scene (cot + capsule torso + sphere head + 2 cameras + IR LED)
- [x] `render.py` — bpy-driven stereo IR render at parameterised geometry + breathing
- [x] `depth.py` — OpenCV StereoSGBM disparity → depth → point cloud, shared frame
- [x] `smoke_test.py` — orchestrator + ±2 BPM assertion
- [ ] **Smoke test PASS** — pending first end-to-end run

This branch (`mvp-simulation`) merges to `main` once the smoke test passes.

## Caveats

- **Infant model is geometric primitives** (capsule torso + sphere head). Sufficient for the smoke test's "does the pipeline see breathing motion at the right frequency"; not sufficient for realistic IR scattering or skin reflectance. Upgrade to rigged commercial mesh in Week 2/3 if Week 1 passes.
- **Eevee renderer** (rasterisation), not Cycles (ray trace). Eevee runs ~50× faster — necessary for a sub-10-minute smoke test. Cycles is the right choice once we're optimising for realism, not pipeline integration.
- **No BLAINDER add-on.** Per Phil's 2026-05-20 note, vanilla Blender + Cycles + OpenCV is sufficient for stereo IR + depth recovery.
