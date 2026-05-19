"""Retrospective peak-pick vs centroid re-score against Polar GT.

Task #77 (2026-05-19). For each --runs run dir, recompute the per-window
FFT history under both ``rr_method="peak-pick"`` and ``rr_method="centroid"``
holding the spatial selection (saved centroid_z.npy) constant, apply the
PR M settled-median to derive a final RR estimate per method, and compare
against the median Polar RR across the recording.

Isolates the algorithmic component of the LiDAR-vs-Polar bias from the
spatial (chest-band) component. Captures pre-date PR Z so the original
result_final in meta.json reflects the PR I centroid default; this script
runs both methods explicitly so the comparison is direct.

Usage
-----
    phase1/run.sh phase1/peak_pick_rescore.py \
        --runs phase1/runs/20260517_063204 ... \
        --out phase1/notes/peak_pick_rescore_2026-05-17.md
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
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import KVConfig
from phase1.respiratory import extract_rr_from_signal, settled_median


def polar_median_rr(gt_csv: Path) -> dict:
    """Median Polar RR (and HR) across the recording. Skips empty cells."""
    rrs, hrs = [], []
    with gt_csv.open() as f:
        for row in csv.DictReader(f):
            rr = row.get("rr", "").strip()
            hr = row.get("hr", "").strip()
            if rr:
                try: rrs.append(float(rr))
                except ValueError: pass
            if hr:
                try: hrs.append(float(hr))
                except ValueError: pass
    return {
        "rr_median": float(np.median(rrs)) if rrs else float("nan"),
        "rr_n":      len(rrs),
        "hr_median": float(np.median(hrs)) if hrs else float("nan"),
        "hr_n":      len(hrs),
    }


def rescore_method(cz: np.ndarray, fps: float, method: str,
                   compute_rr_every: int = 50) -> dict:
    """Re-run FFT history under a specific rr_method and settle."""
    cfg = KVConfig()
    cfg = replace(cfg, rr_method=method)

    history = []
    for i in range(compute_rr_every, len(cz) + 1, compute_rr_every):
        r = extract_rr_from_signal(cz[:i], fps, cfg)
        history.append({
            "rr_bpm":     float(r["rr_bpm"]),
            "snr":        float(r["snr"]),
            "confidence": str(r["confidence"]),
        })

    settled = settled_median(history, cfg)
    return {
        "rr_bpm":     round(settled["rr_bpm"], 2),
        "snr":        round(settled["snr"], 2),
        "confidence": settled["confidence"],
        "settled":    settled["settled"],
        "settle_note": settled["settle_note"],
        "window_n":   settled["window_n"],
        "n_windows":  len(history),
        "history_bpms": [round(h["rr_bpm"], 1) for h in history],
        "history_snrs": [round(h["snr"], 2) for h in history],
    }


def rescore_run(run_dir: Path) -> dict:
    meta_p = run_dir / "meta.json"
    cz_p   = run_dir / "centroid_z.npy"
    gt_p   = run_dir / "gt.csv"
    if not meta_p.exists() or not cz_p.exists():
        return {"run": run_dir.name, "error": "missing meta.json or centroid_z.npy"}

    meta = json.loads(meta_p.read_text())
    cz   = np.load(cz_p)

    # Use the meta-recorded effective fps (frames / wall_seconds), matching
    # rescore.py's correction. Live-pipeline meta sometimes records a stale
    # fps_configured of 10 even when the loop drained at 13-15 fps.
    fps = meta["frames_seen"] / meta["wall_seconds"]

    polar = polar_median_rr(gt_p) if gt_p.exists() else {
        "rr_median": float("nan"), "rr_n": 0,
        "hr_median": float("nan"), "hr_n": 0,
    }

    orig = meta.get("result_final", {})
    out = {
        "run":           run_dir.name,
        "mode":          meta.get("mode"),
        "wall_seconds":  meta.get("wall_seconds"),
        "frames_seen":   meta.get("frames_seen"),
        "fps_corrected": round(fps, 3),
        "polar_rr_median": round(polar["rr_median"], 2)
                           if not np.isnan(polar["rr_median"]) else None,
        "polar_rr_n":    polar["rr_n"],
        "polar_hr_median": round(polar["hr_median"], 2)
                           if not np.isnan(polar["hr_median"]) else None,
        "polar_hr_n":    polar["hr_n"],
        "original_final": {
            "rr_bpm": orig.get("rr_bpm"),
            "snr":    orig.get("snr"),
            "conf":   orig.get("confidence"),
            "settle_note": orig.get("settle_note"),
        },
        "peak_pick":    rescore_method(cz, fps, "peak-pick"),
        "centroid":     rescore_method(cz, fps, "centroid"),
    }

    if out["polar_rr_median"] is not None:
        out["delta_peak_vs_polar"] = round(
            out["peak_pick"]["rr_bpm"] - out["polar_rr_median"], 2)
        out["delta_centroid_vs_polar"] = round(
            out["centroid"]["rr_bpm"] - out["polar_rr_median"], 2)
    return out


def fmt_md(results: list[dict]) -> str:
    """Pretty markdown summary for phase1/notes/."""
    lines = [
        "# Peak-pick vs centroid re-score — 2026-05-17 paired runs",
        "",
        "*Generated 2026-05-19 by `phase1/peak_pick_rescore.py` (task #77).*",
        "",
        "Each of the six live captures from 2026-05-17 is rescored against ",
        "saved `centroid_z.npy` under both `rr_method` settings, using the ",
        "meta-corrected fps and the current `KVConfig` (band 10-30 BPM, "
        "PR M settled-median). Δ rows are *signed* (estimator − Polar median).",
        "",
        "Original `result_final` in each `meta.json` was produced under the ",
        "PR I centroid default. Centroid-rescore differs from original only ",
        "by config drift (band/threshold tightening since the capture).",
        "",
        "## Headline table",
        "",
        "| Run | dur (s) | fps | Polar RR | Peak-pick | Centroid | Δ peak | Δ centroid | Δ method |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['run']} | — | — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        p = r["peak_pick"]; c = r["centroid"]
        polar = r["polar_rr_median"]
        d_peak = r.get("delta_peak_vs_polar", "—")
        d_cent = r.get("delta_centroid_vs_polar", "—")
        d_method = round(c["rr_bpm"] - p["rr_bpm"], 2) if polar is not None else "—"
        lines.append(
            f"| {r['run']} | {r['wall_seconds']:.0f} | {r['fps_corrected']} | "
            f"{polar} | {p['rr_bpm']} ({p['confidence'][0]}) | "
            f"{c['rr_bpm']} ({c['confidence'][0]}) | "
            f"{d_peak} | {d_cent} | {d_method} |"
        )

    # Aggregate stats
    rows = [r for r in results if "delta_peak_vs_polar" in r]
    if rows:
        peak_deltas = np.array([r["delta_peak_vs_polar"] for r in rows])
        cent_deltas = np.array([r["delta_centroid_vs_polar"] for r in rows])
        method_deltas = np.array(
            [r["centroid"]["rr_bpm"] - r["peak_pick"]["rr_bpm"] for r in rows])
        lines += [
            "",
            "## Aggregate",
            "",
            f"- n = {len(rows)} paired live captures",
            f"- **Peak-pick vs Polar:** mean Δ = {peak_deltas.mean():+.2f} BPM, "
            f"|Δ| median = {np.median(np.abs(peak_deltas)):.2f}, "
            f"max |Δ| = {np.max(np.abs(peak_deltas)):.2f}",
            f"- **Centroid vs Polar:** mean Δ = {cent_deltas.mean():+.2f} BPM, "
            f"|Δ| median = {np.median(np.abs(cent_deltas)):.2f}, "
            f"max |Δ| = {np.max(np.abs(cent_deltas)):.2f}",
            f"- **Method split (centroid − peak-pick):** mean = "
            f"{method_deltas.mean():+.2f} BPM, "
            f"|Δ| median = {np.median(np.abs(method_deltas)):.2f}, "
            f"max |Δ| = {np.max(np.abs(method_deltas)):.2f}",
        ]
        # Sign agreement with Run 9 finding (+2.4 BPM centroid > peak-pick)
        n_centroid_higher = int((method_deltas > 0).sum())
        lines.append(
            f"- Centroid > peak-pick in {n_centroid_higher}/{len(rows)} runs "
            f"(Run 9 finding generalises? — see narrative below)"
        )

    lines += ["", "## Per-run detail", ""]
    for r in results:
        lines.append(f"### {r['run']}")
        lines.append("")
        if "error" in r:
            lines.append(f"ERROR: {r['error']}")
            lines.append("")
            continue
        lines += [
            f"- mode: `{r['mode']}` · {r['wall_seconds']:.1f}s · {r['frames_seen']} frames · "
            f"corrected fps {r['fps_corrected']}",
            f"- Polar RR median: **{r['polar_rr_median']}** BPM "
            f"(n={r['polar_rr_n']}); HR median: {r['polar_hr_median']} BPM "
            f"(n={r['polar_hr_n']})",
            f"- Original `meta.json` final (centroid-era): "
            f"{r['original_final']['rr_bpm']:.2f} BPM, "
            f"SNR {r['original_final']['snr']:.2f}, "
            f"{r['original_final']['conf']}",
            "",
            "| Method | RR (BPM) | SNR | Conf | Settled | Note |",
            "|---|---:|---:|:---:|:---:|---|",
            f"| peak-pick | **{r['peak_pick']['rr_bpm']}** | "
            f"{r['peak_pick']['snr']} | {r['peak_pick']['confidence']} | "
            f"{'✓' if r['peak_pick']['settled'] else '✗'} | "
            f"{r['peak_pick']['settle_note']} |",
            f"| centroid | {r['centroid']['rr_bpm']} | "
            f"{r['centroid']['snr']} | {r['centroid']['confidence']} | "
            f"{'✓' if r['centroid']['settled'] else '✗'} | "
            f"{r['centroid']['settle_note']} |",
            "",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs (paths or names under phase1/runs/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional markdown output path")
    ap.add_argument("--json", type=Path, default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    results = []
    for arg in args.runs:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = _HERE / "runs" / arg
        results.append(rescore_run(p))

    md = fmt_md(results)
    print(md)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"# wrote markdown → {args.out}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"# wrote json → {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
