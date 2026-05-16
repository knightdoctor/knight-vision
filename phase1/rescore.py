"""Rescore an existing run with the current config (corrected fps,
widened band, tightened thresholds, settled-window median).

Usage: phase1/run.sh rescore.py <run_dir> [<run_dir>...]
"""
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal, settled_median


def rescore_run(run_dir: Path, compute_rr_every: int = 50,
                band_min: float = None, band_max: float = None,
                cz_file: str = "centroid_z.npy") -> dict:
    meta_p = run_dir / "meta.json"
    cz_p   = run_dir / cz_file
    if not meta_p.exists() or not cz_p.exists():
        return {"error": f"missing meta.json or {cz_file}"}

    meta = json.loads(meta_p.read_text())
    cz   = np.load(cz_p)

    fps_corrected = meta["frames_seen"] / meta["wall_seconds"]
    cfg = KVConfig()
    if band_min is not None: cfg.rr_freq_min = band_min
    if band_max is not None: cfg.rr_freq_max = band_max

    # Recreate per-window history at the same cadence the live loop uses.
    history = []
    for i in range(compute_rr_every, len(cz) + 1, compute_rr_every):
        sig = cz[:i]
        r = extract_rr_from_signal(sig, fps_corrected, cfg)
        history.append({
            "rr_bpm": float(r["rr_bpm"]),
            "snr":    float(r["snr"]),
            "confidence": str(r["confidence"]),
        })

    settled = settled_median(history, cfg)

    # Original final (from meta) for comparison
    orig = meta.get("result_final", {})

    return {
        "run":             run_dir.name,
        "fps_original":    meta.get("fps_estimate") or meta.get("fps_configured"),
        "fps_corrected":   round(fps_corrected, 3),
        "n_windows":       len(history),
        "original_final":  {
            "rr_bpm": orig.get("rr_bpm"),
            "snr":    orig.get("snr"),
            "conf":   orig.get("confidence"),
        },
        "rescored_final":  {
            "rr_bpm":      round(settled["rr_bpm"], 2),
            "snr":         round(settled["snr"], 2),
            "conf":        settled["confidence"],
            "settled":     settled["settled"],
            "note":        settled["settle_note"],
            "window_n":    settled["window_n"],
        },
        "window_bpms": [round(h["rr_bpm"], 1) for h in history],
        "window_snrs": [round(h["snr"], 2) for h in history],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--cz-file", default="centroid_z.npy",
                    help="Which cz array to score. Try cz_chest_0.50_0.85.npy "
                         "after running offline_replay.py with chest mode.")
    ap.add_argument("--bands", default="6-40",
                    help="Comma-separated bands to test, e.g. '6-40,8-30,6-120'")
    args = ap.parse_args()

    bands = []
    for b in args.bands.split(","):
        lo_bpm, hi_bpm = b.split("-")
        bands.append((f"{lo_bpm}-{hi_bpm} BPM",
                      float(lo_bpm)/60.0, float(hi_bpm)/60.0))

    for arg in args.run_dirs:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = _HERE / "runs" / p
        for label, lo, hi in bands:
            result = rescore_run(p, band_min=lo, band_max=hi,
                                 cz_file=args.cz_file)
            print("=" * 70)
            print(f"[{label}]  cz={args.cz_file}")
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
