"""OpenCV stereo block matching → depth maps + Knight-Vision-frame point clouds.

Pure numpy + cv2 — no bpy. Iterates over rendered stereo pairs in
``frames/L`` and ``frames/R``, runs StereoSGBM to recover per-pixel
disparity, converts disparity → depth using the camera intrinsics +
baseline saved by scene_build.py, projects depth to a 3D point cloud,
and rotates from Blender's (X right, Y forward, Z up) into the shared
Knight Vision frame (X right, Y up, Z forward) — the same frame
``phase1/drivers/lidar_driver.py`` produces.

Output (one per frame):
    depth_out/depth/{i:05d}.npy   — depth map (H × W, float32, metres)
    depth_out/cloud/{i:05d}.npy   — point cloud (N × 3, float32, shared frame)

Invocation:
    .venv-local/bin/python phase1/simulation/depth.py \
        --frames phase1/simulation/frames \
        --out    phase1/simulation/depth_out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(frames_dir: Path) -> dict:
    """intrinsics.json lives next to scene.blend; frames_dir is sibling."""
    candidates = [
        frames_dir / "intrinsics.json",
        frames_dir.parent / "intrinsics.json",
        frames_dir.parent / "simulation" / "intrinsics.json",
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    raise FileNotFoundError(
        f"intrinsics.json not found near {frames_dir}; expected one of "
        + ", ".join(str(c) for c in candidates))


def make_K(intr: dict) -> np.ndarray:
    """Pinhole intrinsics K (3×3) from horizontal FOV + resolution."""
    w, h = intr["resolution"]
    fov = math.radians(intr["horiz_fov_deg"])
    fx = (w / 2.0) / math.tan(fov / 2.0)
    fy = fx        # square pixels — matches Blender default sensor
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def make_stereo_matcher(num_disp: int = 128, block: int = 9):
    """Block size must be odd. num_disp must be divisible by 16."""
    num_disp = int(num_disp)
    num_disp = ((num_disp + 15) // 16) * 16
    block = block if block % 2 == 1 else block + 1
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8  * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def disparity_to_depth(disparity: np.ndarray, fx: float,
                       baseline_m: float) -> np.ndarray:
    """Z = f * B / d, with d in pixel units. Invalid (disp ≤ 0) → NaN."""
    disp = disparity.astype(np.float32) / 16.0  # SGBM returns int16 * 16
    depth = np.full_like(disp, np.nan, dtype=np.float32)
    valid = disp > 0.0
    depth[valid] = (fx * baseline_m) / disp[valid]
    return depth


def depth_to_cloud_shared(depth: np.ndarray, K: np.ndarray,
                          cam_loc_world_blender: tuple,
                          chest_centre_world_blender: tuple,
                          voxel_size_m: float = 0.03) -> np.ndarray:
    """Reproject depth pixels to 3D points in the SHARED Knight Vision frame.

    Blender Cam_A points from above-and-near-toe toward the chest; depth
    values from StereoSGBM are along the cam's optical axis. We:
      1. Back-project pixel (u,v,depth) → camera-frame XYZ
      2. Rotate camera-frame → world (Blender) — approximated by aligning
         the camera optical axis with the (chest - cam) vector
      3. Permute Blender (x, y, z) → shared (x, z, y)
      4. Voxel-downsample to ~lidar_driver point count

    The intrinsics + extrinsics from a pinhole model + a known fixed
    rig like this are exact, but we approximate by treating the optical
    axis as the cam→chest vector — same approximation
    fused_overlay_view.py uses for the radar → camera projection on the
    Jetson, per jetson-setup-notes.md.
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    x_cam = (us[valid] - cx) * z[valid] / fx
    y_cam = (vs[valid] - cy) * z[valid] / fy
    z_cam = z[valid]
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)

    # Camera optical axis: from cam_loc toward chest_centre (Blender frame).
    cam = np.array(cam_loc_world_blender, dtype=np.float32)
    chest = np.array(chest_centre_world_blender, dtype=np.float32)
    forward = chest - cam
    forward /= (np.linalg.norm(forward) + 1e-9)
    # Up vector — keep Blender's +Z up by default; orthogonalise.
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)

    # Camera-frame axes (OpenCV convention): X right, Y down, Z forward.
    # Map: cam_x → right, cam_y → -up, cam_z → forward.
    R_cam_to_world = np.stack([right, -up, forward], axis=1)
    pts_world = pts_cam @ R_cam_to_world.T + cam[None, :]

    # Permute Blender (x_world, y_world, z_world) → shared (x, z, y)
    # because Blender Y = forward but shared Y = up.
    shared = np.stack([pts_world[:, 0], pts_world[:, 2], pts_world[:, 1]],
                      axis=1)

    # Voxel downsample so the point count matches the lidar_driver's
    # ~5-10k post-voxel count (3 cm voxels by default).
    if voxel_size_m > 0 and shared.shape[0] > 0:
        keys = np.floor(shared / voxel_size_m).astype(np.int64)
        # Hash keys; unique returns canonical representatives.
        _, idx = np.unique(keys, axis=0, return_index=True)
        shared = shared[idx]

    return shared.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True,
                    help="Directory containing L/ R/ and ground_truth.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (creates depth/ and cloud/)")
    ap.add_argument("--block-size", type=int, default=9)
    ap.add_argument("--num-disparities", type=int, default=128)
    ap.add_argument("--voxel-size-m", type=float, default=0.03)
    args = ap.parse_args()

    frames = args.frames.resolve()
    if not (frames / "L").exists() or not (frames / "R").exists():
        sys.exit(f"ERROR: {frames}/L and {frames}/R must exist")
    intr = load_intrinsics(frames)
    K = make_K(intr)
    fx = K[0, 0]
    baseline_m = float(intr["baseline_m"])
    cam_a_loc = tuple(intr["cam_a"]["location"])
    chest_centre = tuple(intr["chest_centre_world"])

    out = args.out.resolve()
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "cloud").mkdir(parents=True, exist_ok=True)

    matcher = make_stereo_matcher(args.num_disparities, args.block_size)

    L_files = sorted((frames / "L").glob("*.png"))
    if not L_files:
        sys.exit("ERROR: no PNG frames found in L/")
    n = len(L_files)
    print(f"[depth] {n} stereo pairs · baseline {baseline_m:.3f} m · "
          f"fx {fx:.1f} px · K[2,2] {K[2,2]} px")

    for i, L_path in enumerate(L_files):
        R_path = frames / "R" / L_path.name
        if not R_path.exists():
            print(f"[depth] warn: no R pair for {L_path.name}, skipping")
            continue
        L = cv2.imread(str(L_path), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(R_path), cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            print(f"[depth] warn: failed to read pair {L_path.name}")
            continue
        disp = matcher.compute(L, R)
        depth = disparity_to_depth(disp, fx, baseline_m)
        cloud = depth_to_cloud_shared(depth, K, cam_a_loc, chest_centre,
                                      voxel_size_m=args.voxel_size_m)

        np.save(out / "depth" / f"{L_path.stem}.npy", depth)
        np.save(out / "cloud" / f"{L_path.stem}.npy", cloud)

        if (i + 1) % 30 == 0 or i == n - 1:
            valid = np.isfinite(depth) & (depth > 0)
            mean_z = float(np.nanmean(depth[valid])) if valid.any() else float("nan")
            print(f"[depth] {i+1}/{n} · valid pixels {int(valid.sum())} · "
                  f"mean depth {mean_z:.3f} m · cloud pts {cloud.shape[0]}")

    print(f"[depth] done → {out}")


if __name__ == "__main__":
    main()
