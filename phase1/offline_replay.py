"""
phase1/offline_replay.py
========================
Replay a recorded run's residuals.npz through the analyser without
re-capturing from the sensor. Useful for iterating on cluster/chest/RR
parameters offline.

    ./run.sh offline_replay.py runs/<TS>/  [--fps 8.1] [--y-lo 0.50] [--y-hi 0.85]

Writes a new sidecar:
    runs/<TS>/replay_chest_<lo>_<hi>.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from phase1.config import KVConfig
from phase1.clustering import (
    cluster_residuals, select_subject_cluster, select_chest_subset,
)
from phase1.respiratory import extract_rr_from_signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--fps", type=float, default=8.1,
                    help="Effective end-to-end sample rate (default 8.1 = "
                         "what the live run actually achieved).")
    ap.add_argument("--y-lo", type=float, default=0.50)
    ap.add_argument("--y-hi", type=float, default=0.85)
    ap.add_argument("--whole-body", action="store_true",
                    help="Skip chest selection; use whole-subject centroid (baseline).")
    ap.add_argument("--differential", action="store_true",
                    help="Use chest_z - body_z (subtracts whole-body sway).")
    args = ap.parse_args()

    cfg = KVConfig()
    res_path = args.run_dir / "residuals.npz"
    if not res_path.exists():
        sys.exit(f"residuals.npz not found in {args.run_dir}")
    data = np.load(res_path)
    keys = sorted(data.files)
    n = len(keys)
    print(f"replaying {n} frames from {res_path}")
    print(f"  fps = {args.fps}  chest-band y_frac = ({args.y_lo}, {args.y_hi})  "
          f"whole-body = {args.whole_body}")

    cz_series = []
    n_subj = n_chest = 0
    for k in keys:
        r = data[k].astype(np.float64)
        cz = float("nan")
        if r.shape[0] >= cfg.cluster_min_points:
            clusters = cluster_residuals(r, cfg)
            subject = select_subject_cluster(clusters, cfg.monitoring_volume)
            if subject is not None:
                n_subj += 1
                if args.whole_body:
                    cz = float(subject[:, 2].mean())
                else:
                    chest = select_chest_subset(
                        subject,
                        y_band_frac=(args.y_lo, args.y_hi),
                    )
                    if chest is not None:
                        n_chest += 1
                        chest_cz = float(chest[:, 2].mean())
                        if args.differential:
                            body_cz = float(subject[:, 2].mean())
                            cz = chest_cz - body_cz
                        else:
                            cz = chest_cz
                    else:
                        cz = float(subject[:, 2].mean())  # fallback
        cz_series.append(cz)

    sig = np.array(cz_series, dtype=float)
    valid = ~np.isnan(sig)
    print(f"  frames with subject: {n_subj}/{n}  ({100*n_subj/n:.0f}%)")
    if not args.whole_body:
        print(f"  frames with chest:   {n_chest}/{n_subj}  "
              f"({100*n_chest/max(1,n_subj):.0f}% of subject frames)")

    if valid.sum() < 30:
        sys.exit(f"too few valid frames ({valid.sum()}) for FFT")

    result = extract_rr_from_signal(sig, fps=args.fps, config=cfg)
    print(f"\n  RR        : {result['rr_bpm']:.2f} BPM")
    print(f"  SNR       : {result['snr']:.2f}")
    print(f"  Confidence: {result['confidence']}")

    label = "whole" if args.whole_body else f"chest_{args.y_lo:.2f}_{args.y_hi:.2f}"
    if args.differential:
        label = label + "_diff"
    out = args.run_dir / f"replay_{label}.txt"
    out.write_text(
        f"fps={args.fps}\n"
        f"chest_y_band=({args.y_lo},{args.y_hi})\n"
        f"whole_body={args.whole_body}\n"
        f"differential={args.differential}\n"
        f"rr_bpm={result['rr_bpm']:.4f}\n"
        f"snr={result['snr']:.4f}\n"
        f"confidence={result['confidence']}\n"
        f"frames_total={n}\n"
        f"frames_with_subject={n_subj}\n"
        f"frames_with_chest={n_chest}\n"
    )
    cz_out = args.run_dir / f"cz_{label}.npy"
    np.save(cz_out, sig)
    print(f"  wrote {out}")
    print(f"  wrote {cz_out}")


if __name__ == "__main__":
    main()
