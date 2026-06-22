"""OpenCV stereo block matching → depth maps + Knight-Vision-frame point clouds.

Pure numpy + cv2 — no bpy. Iterates over rendered stereo pairs in
``frames/L`` and ``frames/R``, runs a full cv2.stereoRectify pipeline,
then SGBM on the rectified pair, recovers 3D points via
``cv2.reprojectImageTo3D``, and rotates from Blender's
(X right, Y forward, Z up) into the shared Knight Vision frame
(X right, Y up, Z forward) — the same frame
``phase1/drivers/lidar_driver.py`` produces.

Week-2 #1 (2026-06-21): replaces the rotate-90-and-swap heuristic. The
Cam_A / Cam_B world rotations are derived analytically from the camera
locations + ``chest_centre_world`` by reproducing Blender's
``Vector(target - loc).to_track_quat('-Z', 'Y')`` rule, then converted
to the OpenCV camera convention (X right, Y down, Z forward).

Because Cam_A / Cam_B have a *vertical* baseline (Cam_A above Cam_B in
world), the inputs are pre-rotated 90° CW before rectification — that
swings the baseline horizontal in image space so the SGBM rectified
output has horizontal epipolar lines (which SGBM requires). The
camera-to-world transforms are adjusted to track this pre-rotation, so
reprojected 3D points end up in the correct world coordinates.

Output (one per frame):
    depth_out/depth/{i:05d}.npy   — depth map (H × W, float32, metres)
                                    in the rectified frame
    depth_out/cloud/{i:05d}.npy   — point cloud (N × 3, float32,
                                    shared Knight Vision frame)

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


# ── camera-pose maths ────────────────────────────────────────────────────────

# Blender camera convention: X right, Y up, Z back (cam looks along -Z).
# OpenCV camera convention:  X right, Y down, Z forward.
# Conversion: pt_opencv = diag(1, -1, -1) @ pt_blender, equivalently
# R_cv_to_world = R_blender_to_world @ diag(1, -1, -1).
_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# 90°-CW image rotation = 90°-CCW camera-frame rotation around +Z.
# A point at (x, y, z) in the OLD camera frame is at (y, -x, z) in the
# NEW (post-image-CW-rotation) camera frame. So P_new = R_OLD_TO_NEW @ P_old
# with R_OLD_TO_NEW = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] (which is R_z(-90°)).
_R_OLD_TO_NEW_CWROT = np.array([[0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
_R_NEW_TO_OLD_CWROT = _R_OLD_TO_NEW_CWROT.T


def _lookat_rotation_blender(loc, target):
    """Reproduce mathutils.Vector(target - loc).to_track_quat('-Z', 'Y').

    Returns 3×3 R such that ``pt_world = R @ pt_in_blender_camera_frame``.
    The rule: -Z (local) aligns with ``target - loc``; +Y (local) lies
    closest to Blender's world up (+Z) in the plane perpendicular to the
    forward direction.
    """
    loc = np.asarray(loc, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    direction = target - loc
    n = np.linalg.norm(direction)
    if n < 1e-9:
        raise ValueError(f"camera at {loc} cannot look at itself")
    direction /= n
    z_cam = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    y_cam = world_up - np.dot(world_up, z_cam) * z_cam
    y_norm = np.linalg.norm(y_cam)
    if y_norm < 1e-6:
        y_cam = np.array([0.0, 1.0, 0.0])
    else:
        y_cam /= y_norm
    x_cam = np.cross(y_cam, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    return np.column_stack([x_cam, y_cam, z_cam])


def camera_world_pose_opencv(loc, target):
    """Return (R_cv_to_world, T_world) for a Blender-look-at camera in the
    OpenCV camera convention (X right, Y down, Z forward)."""
    R_blender = _lookat_rotation_blender(loc, target)
    R_cv = R_blender @ _BLENDER_TO_OPENCV
    T_world = np.asarray(loc, dtype=np.float64)
    return R_cv, T_world


def apply_cw_pre_rotation(R_cv_to_world: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation after the camera's image is rotated 90° CW.
    Equivalent to rotating the camera 90° CCW around its optical (Z) axis."""
    return R_cv_to_world @ _R_NEW_TO_OLD_CWROT


def relative_pose(R_a_world, T_a_world, R_b_world, T_b_world):
    """Return (R, T) describing the transform that takes points from Cam A
    into Cam B's frame: ``P_in_B = R @ P_in_A + T``. cv2.stereoRectify
    consumes exactly this convention as its (R, T) arguments."""
    R = R_b_world.T @ R_a_world
    T = R_b_world.T @ (T_a_world - T_b_world)
    return R, T


# ── intrinsics + matcher ─────────────────────────────────────────────────────

def load_intrinsics(frames_dir: Path) -> dict:
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


def rotate_K_cw(K: np.ndarray, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """K' for an image rotated 90° CW. New image dims are (h, w).

    Old pixel (x, y) maps to new pixel (h - 1 - y, x). So a 3D ray that
    projected to (cx_old, cy_old) in the old image projects to
    (h - 1 - cy_old, cx_old) in the new image — that's the new principal
    point.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    K_new = np.array([[fy, 0.0, (h - 1) - cy],
                      [0.0, fx, cx],
                      [0.0, 0.0, 1.0]])
    return K_new, h, w


def make_stereo_matcher(num_disp: int = 128, block: int = 9):
    """Block size must be odd. num_disp must be divisible by 16."""
    num_disp = int(num_disp)
    num_disp = ((num_disp + 15) // 16) * 16
    block = block if block % 2 == 1 else block + 1
    return cv2.StereoSGBM_create(
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


# ── rectification ────────────────────────────────────────────────────────────

def setup_rectification(K: np.ndarray, image_size: tuple[int, int],
                        R_a_to_b: np.ndarray, t_a_to_b: np.ndarray):
    """Run cv2.stereoRectify + initUndistortRectifyMap for both cameras.

    Returns a dict with R1, R2, P1, P2, Q, remap LUTs, and the rectified
    baseline magnitude (extracted from P2 for sanity printing).
    """
    dist_zero = np.zeros((5,), dtype=np.float64)
    R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=dist_zero,
        cameraMatrix2=K, distCoeffs2=dist_zero,
        imageSize=image_size,
        R=R_a_to_b, T=t_a_to_b,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1,
    )
    map_a_x, map_a_y = cv2.initUndistortRectifyMap(
        K, dist_zero, R1, P1, image_size, cv2.CV_32FC1)
    map_b_x, map_b_y = cv2.initUndistortRectifyMap(
        K, dist_zero, R2, P2, image_size, cv2.CV_32FC1)
    # Horizontal-stereo rectification: P2[0, 3] = -fx_new * baseline.
    # Vertical-stereo rectification: P2[1, 3] = -fy_new * baseline.
    bx = abs(P2[0, 3] / P2[0, 0]) if abs(P2[0, 0]) > 0 else 0.0
    by = abs(P2[1, 3] / P2[1, 1]) if abs(P2[1, 1]) > 0 else 0.0
    baseline_m = float(max(bx, by))
    axis = "horizontal" if bx >= by else "vertical"
    return {
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "map_a_x": map_a_x, "map_a_y": map_a_y,
        "map_b_x": map_b_x, "map_b_y": map_b_y,
        "baseline_m": baseline_m,
        "rect_axis": axis,
    }


def blender_world_to_shared(pts_world: np.ndarray) -> np.ndarray:
    """Blender (X right, Y forward, Z up) → shared (X right, Y up, Z forward).
    Permutation: shared = (x, z, y) of Blender world."""
    return np.stack([pts_world[:, 0],
                     pts_world[:, 2],
                     pts_world[:, 1]], axis=1)


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m <= 0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True,
                    help="Directory containing L/ R/ and ground_truth.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (creates depth/ and cloud/)")
    ap.add_argument("--block-size", type=int, default=9)
    # Size from the measured scene min depth, not assumption.
    # With rectified fx≈1108 px and baseline 0.20 m, Z_min vs num_disp:
    #   640  → Z_min 0.347 m   (FAILED — closest valid pixel saturated at 639/640)
    #   784  → Z_min 0.283 m   (current default; verify min depth ≥ this)
    # SGBM has a hard width constraint: image_width − (minDisp + numDisp)
    # > block_size/2. The CW pre-rotation puts width at 800 px, capping
    # num_disp at 784 here. If the scene's true min depth is < 0.28 m,
    # raise minDisparity to shift the search window upward.
    ap.add_argument("--num-disparities", type=int, default=784)
    ap.add_argument("--voxel-size-m", type=float, default=0.03)
    ap.add_argument("--max-depth-m", type=float, default=2.0,
                    help="Reject reprojected pixels beyond this depth in "
                         "the rectified-Cam-A frame (kills sky / wall pixels)")
    args = ap.parse_args()

    frames = args.frames.resolve()
    if not (frames / "L").exists() or not (frames / "R").exists():
        sys.exit(f"ERROR: {frames}/L and {frames}/R must exist")
    intr = load_intrinsics(frames)
    K_orig = make_K(intr)
    w_orig, h_orig = intr["resolution"]

    cam_a_loc = tuple(intr["cam_a"]["location"])
    cam_b_loc = tuple(intr["cam_b"]["location"])
    chest_centre = tuple(intr["chest_centre_world"])

    # Original OpenCV-convention world poses (look-at chest).
    R_a_cv, T_a_world = camera_world_pose_opencv(cam_a_loc, chest_centre)
    R_b_cv, T_b_world = camera_world_pose_opencv(cam_b_loc, chest_centre)

    # Pre-rotate both cameras 90° CW (in image) so the world baseline
    # (vertical between Cam A above and Cam B below) becomes a horizontal
    # baseline in image space — required for SGBM rectified output to
    # have horizontal epipolar lines.
    R_a_pre = apply_cw_pre_rotation(R_a_cv)
    R_b_pre = apply_cw_pre_rotation(R_b_cv)
    K_rot, w_rot, h_rot = rotate_K_cw(K_orig, w_orig, h_orig)
    image_size = (int(w_rot), int(h_rot))

    R_a_to_b, t_a_to_b = relative_pose(
        R_a_pre, T_a_world, R_b_pre, T_b_world)

    rect = setup_rectification(K_rot, image_size, R_a_to_b, t_a_to_b)
    R1 = rect["R1"]
    Q = rect["Q"]
    print(f"[depth] world cam-cam {np.linalg.norm(t_a_to_b):.4f} m · "
          f"rectified baseline {rect['baseline_m']:.4f} m · "
          f"axis {rect['rect_axis']} · "
          f"fx_rect {rect['P1'][0,0]:.1f} px")
    print(f"[depth] t_a_to_b after CW pre-rotation = "
          f"({t_a_to_b[0]:+.3f}, {t_a_to_b[1]:+.3f}, {t_a_to_b[2]:+.3f}) m")

    # Combined rotation taking a point in the rectified-Cam-A frame all the
    # way to Blender world: P_world = R_combined @ P_rect + T_a_world.
    # P_rect       = R1 @ P_pre_camA       (rectification rotates pre-rot Cam A)
    # P_pre_camA   = R_pre_to_old @ P_camA = R_NEW_TO_OLD @ P_camA — but our
    # R_a_pre already absorbed that transform (R_a_pre maps the pre-rotated
    # camera frame to world). So:
    # P_world = R_a_pre @ R1.T @ P_rect + T_a_world.
    R_rect_to_world = R_a_pre @ R1.T

    out = args.out.resolve()
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "cloud").mkdir(parents=True, exist_ok=True)
    matcher = make_stereo_matcher(args.num_disparities, args.block_size)

    L_files = sorted((frames / "L").glob("*.png"))
    if not L_files:
        sys.exit("ERROR: no PNG frames found in L/")
    n = len(L_files)
    fx_rect = float(rect["P1"][0, 0])
    z_min_resolvable = fx_rect * rect["baseline_m"] / args.num_disparities
    print(f"[depth] {n} stereo pairs · CW pre-rotate + stereoRectify "
          f"path · num_disp {args.num_disparities} · block {args.block_size}")
    print(f"[depth] num_disp {args.num_disparities} ⇒ Z_min resolvable "
          f"{z_min_resolvable:.3f} m (anything closer clips at max disp)")

    # Track scene min depth + max disparity across all frames.
    scene_min_depth = float("inf")
    scene_max_disp = 0.0

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
        # Pre-rotate inputs 90° CW.
        L_pre = cv2.rotate(L, cv2.ROTATE_90_CLOCKWISE)
        R_pre = cv2.rotate(R, cv2.ROTATE_90_CLOCKWISE)
        # Rectify.
        L_rect = cv2.remap(L_pre, rect["map_a_x"], rect["map_a_y"],
                           cv2.INTER_LINEAR)
        R_rect = cv2.remap(R_pre, rect["map_b_x"], rect["map_b_y"],
                           cv2.INTER_LINEAR)
        disp_int = matcher.compute(L_rect, R_rect)        # int16, ×16
        disp = disp_int.astype(np.float32) / 16.0
        pts3d = cv2.reprojectImageTo3D(disp, Q)           # (H, W, 3), rect-A
        depth = pts3d[..., 2]
        valid = (
            np.isfinite(depth)
            & (disp > 0.0)
            & (depth > 0.0)
            & (depth < args.max_depth_m)
        )
        np.save(out / "depth" / f"{L_path.stem}.npy",
                np.where(valid, depth, np.nan).astype(np.float32))

        pts_rect_valid = pts3d[valid].reshape(-1, 3).astype(np.float64)
        if pts_rect_valid.shape[0] == 0:
            cloud_shared = np.empty((0, 3), dtype=np.float32)
        else:
            pts_world = pts_rect_valid @ R_rect_to_world.T + T_a_world[None, :]
            cloud_shared = blender_world_to_shared(pts_world).astype(
                np.float32)
            cloud_shared = voxel_downsample(cloud_shared, args.voxel_size_m)
        np.save(out / "cloud" / f"{L_path.stem}.npy", cloud_shared)

        if valid.any():
            scene_min_depth = min(scene_min_depth, float(np.nanmin(depth[valid])))
            scene_max_disp = max(scene_max_disp, float(np.nanmax(disp[valid])))
        if (i + 1) % 30 == 0 or i == n - 1:
            mean_z = float(np.nanmean(depth[valid])) if valid.any() \
                else float("nan")
            mean_d = float(np.nanmean(disp[valid])) if valid.any() \
                else float("nan")
            print(f"[depth] {i+1}/{n} · valid {int(valid.sum())} px · "
                  f"mean disp {mean_d:.2f} px · "
                  f"mean depth {mean_z:.3f} m · "
                  f"cloud pts {cloud_shared.shape[0]}")

    headroom_disp = args.num_disparities - scene_max_disp
    print(f"[depth] scene min depth: {scene_min_depth:.3f} m  "
          f"(Z_min resolvable: {z_min_resolvable:.3f} m)")
    print(f"[depth] scene max disp:  {scene_max_disp:.1f} px  "
          f"(num_disp ceiling: {args.num_disparities}; headroom "
          f"{headroom_disp:.0f} px = {headroom_disp/args.num_disparities*100:.1f}%)")
    if scene_max_disp >= args.num_disparities - 1:
        print(f"[depth] WARNING: max disp at ceiling — disparity is clipping. "
              f"Bump num_disp.")
    print(f"[depth] done → {out}")


if __name__ == "__main__":
    main()
