"""Render N frames of synthetic IR stereo through the scene.blend cameras.

Animates `infant_torso.chest_expansion` shape-key value as a sinusoid at
``--rr-bpm`` BPM. For each frame, renders Cam_A → frames/L/{i:05d}.png
and Cam_B → frames/R/{i:05d}.png. Writes ground_truth.csv with the
shape-key value and elapsed time per frame.

Invocation:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python phase1/simulation/render.py -- \
        --scene phase1/simulation/scene.blend \
        --rr-bpm 15 --amplitude-mm 5 \
        --duration 30 --fps 15 \
        --out phase1/simulation/frames
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import bpy


def cli_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--rr-bpm", type=float, default=15.0)
    ap.add_argument("--amplitude-mm", type=float, default=5.0,
                    help="Peak chest excursion in mm. Scales the shape-key "
                         "value: shape key 1.0 = scene's built-in excursion; "
                         "values >1 over-drive the existing shape.")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Total rendered duration in seconds")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Frames per second to render")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory (creates L/, R/, ground_truth.csv)")
    return ap.parse_args(argv)


def main():
    args = cli_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.scene.resolve()))

    scene = bpy.context.scene
    scene.render.fps = int(round(args.fps))
    n_frames = int(round(args.duration * args.fps))

    torso = bpy.data.objects.get("infant_torso")
    if torso is None or torso.data.shape_keys is None:
        sys.exit("ERROR: scene has no infant_torso with shape keys")
    sk = torso.data.shape_keys.key_blocks.get("chest_expansion")
    if sk is None:
        sys.exit("ERROR: chest_expansion shape key not found on infant_torso")

    cam_a = bpy.data.objects.get("Cam_A")
    cam_b = bpy.data.objects.get("Cam_B")
    if cam_a is None or cam_b is None:
        sys.exit("ERROR: Cam_A and Cam_B must exist in the scene")

    out = args.out.resolve()
    (out / "L").mkdir(parents=True, exist_ok=True)
    (out / "R").mkdir(parents=True, exist_ok=True)
    gt_path = out / "ground_truth.csv"

    # Scale factor: scene.blend was built with chest_excursion_mm = 5 by default
    # (DEFAULTS["chest_excursion_mm"] in scene_build.py). The shape key value
    # maps 1.0 → that built-in excursion. To get `args.amplitude_mm`, set
    # max shape-key value to (args.amplitude_mm / 5.0).
    scene_default_mm = 5.0
    sk_max = args.amplitude_mm / scene_default_mm

    with gt_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "shape_key_value", "expected_amplitude_mm"])

        omega = 2.0 * math.pi * (args.rr_bpm / 60.0)
        for i in range(n_frames):
            t = i / args.fps
            # Half-amplitude sine centred on 0.5×max so the chest exhales
            # below baseline as well as inhales above. (sin range [-1,1] →
            # shape-key range [0, sk_max].)
            sk_val = (sk_max / 2.0) * (1.0 + math.sin(omega * t))
            sk.value = sk_val
            scene.frame_set(i + 1)
            sk.value = sk_val  # ensure post-frame-set application
            w.writerow([i, f"{t:.4f}", f"{sk_val:.6f}",
                        f"{args.amplitude_mm * 0.5 * (1.0 + math.sin(omega * t)):.4f}"])

            # Left
            scene.camera = cam_a
            scene.render.filepath = str(out / "L" / f"{i:05d}.png")
            bpy.ops.render.render(write_still=True)

            # Right
            scene.camera = cam_b
            scene.render.filepath = str(out / "R" / f"{i:05d}.png")
            bpy.ops.render.render(write_still=True)

            if (i + 1) % 30 == 0 or i == n_frames - 1:
                print(f"[render] frame {i+1}/{n_frames} (t={t:.2f}s, "
                      f"sk={sk_val:.3f})")

    print(f"[render] done. {n_frames} frames × 2 cams in {out}")
    print(f"[render] ground truth → {gt_path}")


if __name__ == "__main__":
    main()
