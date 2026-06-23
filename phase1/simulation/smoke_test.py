"""End-to-end smoke test: synthetic 15-BPM breathing → recovered RR.

Validates that the Phase 1 algorithm stack (extract_rr_from_signal) ports
to depth derived from passive stereo IR. Path:

  depth.py's per-frame point clouds
    → extract chest-region centroid_z per frame
    → feed centroid_z series into phase1.respiratory.extract_rr_from_signal
    → assert recovered RR within ±tolerance of --ground-truth-bpm.

Note: this is the *thin-adapter* smoke test specified in
mvp_simulation_spec_2026-05-19.md §"Algorithm pipeline integration".
The full Phase1Pipeline (background subtraction + DBSCAN + chest band
selection) is exercised in a follow-up step in Week 2; for the smoke
test, the synthetic scene's chest centroid is known by construction
so we can short-circuit to the FFT and validate it picks the right
frequency on synthetic stereo depth.

Exit codes:
    0 — PASS (|recovered − ground_truth| ≤ tolerance)
    1 — FAIL
    2 — error (no data / invalid input)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal

# Used for forward-modelling the ideal chest signal in
# amplitude_recovery_report. Importing from the sibling depth module is the
# cleanest way to share the camera-pose maths.
from depth import camera_world_pose_opencv_from_intr

# ── Synthetic-scene geometry — must mirror scene_build.py DEFAULTS ─────────
# (Hardcoded here rather than read from intrinsics.json so the smoke test can
# forward-model the chest deformation analytically without depending on bpy.)
TORSO_RADIUS_M       = 0.08    # capsule radius
TORSO_LENGTH_M       = 0.30    # capsule length (along Blender +Y)
CHEST_EXCURSION_MM_AT_SK1 = 5.0  # shape key value 1.0 ⇒ apex push 5 mm


def chest_centroid_z(cloud: np.ndarray, intr: dict) -> float:
    """Mean Z of points falling inside a torso-fitted box.

    Shared-frame: X right, Y up, Z forward. Chest centre in shared frame =
    (chest_world[0], chest_world[2], chest_world[1]) — Blender (x,y,z) →
    shared (x, z, y) permutation that depth.py applies.

    Box is sized to the synthetic torso (capsule of radius 8 cm × length
    30 cm) and clears the mattress (top at shared-Y = 0.10 m). Per
    simulation_results_2026-05-20.md "Known issues" #2: the Week-1 box
    (24 × 24 × 40 cm) was 3× too large in Y and pulled mattress pixels
    into the centroid, which interacted with the set-membership
    non-linearity to produce a strong 30-BPM 2nd harmonic that fooled
    peak-pick. Tightening to the torso geometry restores 15-BPM dominance
    and recovers ~70 % of the true 8-mm peak-to-peak excursion.
    """
    if cloud.shape[0] == 0:
        return float("nan")
    cx, cy_blender, cz_blender = intr["chest_centre_world"]
    chest_shared = (cx, cz_blender, cy_blender)
    half_x = 0.08    # ± torso radius (lateral)
    half_y = 0.10    # ± 10 cm around chest centre — wide enough to catch the
                     # upper torso surface that pitched-down cameras see (the
                     # B1 L-overhang variant only sees the top hemisphere of
                     # the torso, so cloud points sit at shared-Y > 0.18)
    half_z = 0.15    # ± half torso length (forward range)
    mask = (
        (np.abs(cloud[:, 0] - chest_shared[0]) <= half_x)
        & (np.abs(cloud[:, 1] - chest_shared[1]) <= half_y)
        & (np.abs(cloud[:, 2] - chest_shared[2]) <= half_z)
    )
    if not mask.any():
        return float("nan")
    return float(np.mean(cloud[mask, 2]))


def load_clouds(cloud_dir: Path) -> tuple[list[Path], list[np.ndarray]]:
    paths = sorted(cloud_dir.glob("*.npy"))
    return paths, [np.load(p) for p in paths]


# ── Amplitude recovery report ─────────────────────────────────────────────
# Reports MEASURED and forward-modelled IDEAL chest-centroid std/p2p on three
# signals so it's explicit which projection of chest motion is being recovered:
#   shared-Y   — chest-height direction (Blender Z); the direction the chest
#                physically moves in this scene.
#   shared-Z   — world forward (Blender Y); orthogonal to motion in the
#                current camera layout, so ideal std/p2p are ~0 and any
#                measurement here is amplitude-modulated noise.
#   cam-A depth — depth along Cam_A's OpenCV optical axis (sensor-centric).
#                 What the Phase-1 LiDAR pipeline measures. Sensitivity to
#                 chest motion depends on the angle between the optical axis
#                 and the breathing direction, so #2 (L-overhang, cams
#                 pitched down) should grow this number relative to here.
#
# Recovery ratio = measured_std / ideal_std. Values near 1.0 = faithful
# recovery. Very large values (>> 1) mean measurement is dominated by noise.
def _torso_surface_points(n_theta: int = 200, n_phi: int = 400) -> tuple:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    xu = np.sin(T) * np.cos(P)
    yu = np.sin(T) * np.sin(P)
    zu = np.cos(T)
    x_loc = (TORSO_RADIUS_M * xu).ravel()
    y_loc = ((TORSO_LENGTH_M / 2.0) * yu).ravel()
    z_loc = (TORSO_RADIUS_M * zu).ravel()
    w_z = np.where(z_loc > 0, z_loc / TORSO_RADIUS_M, 0.0)
    w_y = np.exp(-((y_loc) / (0.25 * TORSO_LENGTH_M)) ** 2)
    return x_loc, y_loc, z_loc, w_z, w_y


def _ideal_chest_signals_per_frame(sk_vals: np.ndarray, intr: dict,
                                   half_x: float, half_y: float,
                                   half_z: float) -> dict:
    """Forward-model the chest centroid per frame from synthetic GT.

    Returns dict with shared_y/shared_z/cam_a_depth lists of mean signal
    values across the chest box, one per frame.
    """
    x_loc, y_loc, z_loc, w_z, w_y = _torso_surface_points()
    bx, by_b, bz_b = intr["chest_centre_world"]
    chest_centre_blender = np.array([bx, by_b, bz_b])
    chest_shared = (bx, bz_b, by_b)
    R_a, T_a = camera_world_pose_opencv_from_intr(intr, "cam_a")

    excursion_m = CHEST_EXCURSION_MM_AT_SK1 / 1000.0
    sig_y, sig_z, sig_depth = [], [], []
    for sk in sk_vals:
        z_def = z_loc + excursion_m * w_z * w_y * sk
        pts_b = np.column_stack([x_loc,
                                 y_loc + chest_centre_blender[1],
                                 z_def + chest_centre_blender[2]])
        pts_s = np.column_stack([pts_b[:, 0], pts_b[:, 2], pts_b[:, 1]])
        mask = (
            (np.abs(pts_s[:, 0] - chest_shared[0]) <= half_x)
            & (np.abs(pts_s[:, 1] - chest_shared[1]) <= half_y)
            & (np.abs(pts_s[:, 2] - chest_shared[2]) <= half_z)
        )
        if not mask.any():
            sig_y.append(np.nan); sig_z.append(np.nan); sig_depth.append(np.nan)
            continue
        in_box_b = pts_b[mask]
        in_box_s = pts_s[mask]
        # Cam A depth = OpenCV Z of points expressed in Cam A's frame.
        # P_camA = R_a_cv_to_world^T @ (P_world - T_a_world)
        in_cam_a = (in_box_b - T_a) @ R_a
        sig_y.append(float(in_box_s[:, 1].mean()))
        sig_z.append(float(in_box_s[:, 2].mean()))
        sig_depth.append(float(in_cam_a[:, 2].mean()))
    return {"shared_y": np.array(sig_y),
            "shared_z": np.array(sig_z),
            "cam_a_depth": np.array(sig_depth)}


def _measured_chest_signals_per_frame(clouds, intr: dict,
                                      half_x: float, half_y: float,
                                      half_z: float) -> dict:
    bx, by_b, bz_b = intr["chest_centre_world"]
    chest_shared = (bx, bz_b, by_b)
    R_a, T_a = camera_world_pose_opencv_from_intr(intr, "cam_a")
    sig_y, sig_z, sig_depth = [], [], []
    for c in clouds:
        if c.shape[0] == 0:
            sig_y.append(np.nan); sig_z.append(np.nan); sig_depth.append(np.nan)
            continue
        mask = (
            (np.abs(c[:, 0] - chest_shared[0]) <= half_x)
            & (np.abs(c[:, 1] - chest_shared[1]) <= half_y)
            & (np.abs(c[:, 2] - chest_shared[2]) <= half_z)
        )
        if not mask.any():
            sig_y.append(np.nan); sig_z.append(np.nan); sig_depth.append(np.nan)
            continue
        in_box_s = c[mask]
        # shared (x, y, z) → blender (x, z, y)
        in_box_b = np.column_stack([in_box_s[:, 0],
                                    in_box_s[:, 2],
                                    in_box_s[:, 1]])
        in_cam_a = (in_box_b - T_a) @ R_a
        sig_y.append(float(in_box_s[:, 1].mean()))
        sig_z.append(float(in_box_s[:, 2].mean()))
        sig_depth.append(float(in_cam_a[:, 2].mean()))
    return {"shared_y": np.array(sig_y),
            "shared_z": np.array(sig_z),
            "cam_a_depth": np.array(sig_depth)}


def amplitude_recovery_report(clouds, intr: dict, gt_rows: list,
                              half_x: float = 0.08,
                              half_y: float = 0.10,
                              half_z: float = 0.15) -> dict:
    sk_vals = np.array([float(r["shape_key_value"]) for r in gt_rows])
    ideal = _ideal_chest_signals_per_frame(sk_vals, intr, half_x, half_y, half_z)
    meas = _measured_chest_signals_per_frame(clouds, intr, half_x, half_y, half_z)
    out = {}
    for axis in ("shared_y", "shared_z", "cam_a_depth"):
        i = ideal[axis]; m = meas[axis]
        i_std = float(np.nanstd(i))
        m_std = float(np.nanstd(m))
        i_p2p = float(np.nanmax(i) - np.nanmin(i))
        m_p2p = float(np.nanmax(m) - np.nanmin(m))
        # Ratio defined only when ideal std is nonzero. If ideal is ~0 the
        # axis is geometry-orthogonal to motion and the measured signal is
        # by definition not amplitude — flag it explicitly.
        ratio = (m_std / i_std) if i_std > 1e-6 else None
        out[axis] = {
            "ideal_std_mm":     round(i_std * 1000.0, 4),
            "measured_std_mm":  round(m_std * 1000.0, 4),
            "ideal_p2p_mm":     round(i_p2p * 1000.0, 4),
            "measured_p2p_mm":  round(m_p2p * 1000.0, 4),
            "recovery_ratio":   None if ratio is None else round(ratio, 3),
            "note":             (
                "axis orthogonal to motion; measurement is amplitude-"
                "modulated noise, not recovery"
                if ratio is None else None),
        }
    return out


def load_ground_truth(frames_dir: Path) -> dict:
    gt = frames_dir / "ground_truth.csv"
    if not gt.exists():
        return {}
    rows = list(csv.DictReader(gt.open()))
    return {
        "n_frames": len(rows),
        "duration_s": float(rows[-1]["t_s"]) if rows else 0.0,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", type=Path, required=True,
                    help="Output dir from depth.py (contains cloud/, depth/)")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="Renders dir from render.py (for ground_truth.csv "
                         "and intrinsics.json). Defaults to "
                         "<depth-dir>/../frames")
    ap.add_argument("--intrinsics", type=Path, default=None,
                    help="intrinsics.json path; falls back to alongside scene.blend")
    ap.add_argument("--ground-truth-bpm", type=float, default=15.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON report path")
    args = ap.parse_args()

    cloud_dir = args.depth_dir.resolve() / "cloud"
    if not cloud_dir.exists():
        sys.exit(f"ERROR: {cloud_dir} not found — run depth.py first")
    frames_dir = (args.frames_dir.resolve() if args.frames_dir
                  else (args.depth_dir.parent / "frames").resolve())

    intr_path = (args.intrinsics
                 or frames_dir / "intrinsics.json"
                 or frames_dir.parent / "intrinsics.json")
    for cand in (intr_path, frames_dir / "intrinsics.json",
                 frames_dir.parent / "intrinsics.json"):
        if cand.exists():
            intr = json.loads(cand.read_text())
            break
    else:
        sys.exit("ERROR: intrinsics.json not found")

    gt = load_ground_truth(frames_dir)
    if gt.get("n_frames", 0) < 30:
        sys.exit(f"ERROR: ground_truth.csv has {gt.get('n_frames',0)} frames, "
                 f"need ≥ 30")
    fps = gt["n_frames"] / gt["duration_s"]

    paths, clouds = load_clouds(cloud_dir)
    if len(clouds) != gt["n_frames"]:
        print(f"WARN: cloud count {len(clouds)} != gt frames {gt['n_frames']}",
              file=sys.stderr)

    cz_series = np.array([chest_centroid_z(c, intr) for c in clouds],
                         dtype=float)
    n_valid = int(np.sum(np.isfinite(cz_series)))
    if n_valid < 30:
        sys.exit(f"ERROR: only {n_valid} frames had chest-box points; "
                 "scene/depth chain not producing usable data")

    cfg = replace(KVConfig(), rr_method="peak-pick")
    r = extract_rr_from_signal(cz_series, fps, cfg)

    recovered = float(r["rr_bpm"])
    snr = float(r["snr"])
    conf = str(r["confidence"])
    delta = recovered - args.ground_truth_bpm
    passed = abs(delta) <= args.tolerance

    amp = amplitude_recovery_report(clouds, intr, gt["rows"])
    report = {
        "ground_truth_bpm":  args.ground_truth_bpm,
        "tolerance_bpm":     args.tolerance,
        "recovered_bpm":     round(recovered, 3),
        "delta_bpm":         round(delta, 3),
        "snr":               round(snr, 3),
        "confidence":        conf,
        "rr_method":         cfg.rr_method,
        "fps":               round(fps, 3),
        "n_frames":          gt["n_frames"],
        "n_chest_valid":     n_valid,
        "cz_std_mm":         round(float(np.nanstd(cz_series) * 1000.0), 3),
        "cz_span_mm":        round(
            float((np.nanmax(cz_series) - np.nanmin(cz_series)) * 1000.0), 3),
        "amplitude_recovery": amp,
        "pass":              passed,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"# wrote {args.out}", file=sys.stderr)

    headline = (
        f"{'✅' if passed else '⛔'} smoke test "
        f"recovered {recovered:.2f} BPM (gt {args.ground_truth_bpm}, "
        f"Δ {delta:+.2f}) · SNR {snr:.2f} · {conf} · "
        f"cz_std {report['cz_std_mm']} mm"
    )
    print(headline)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
