"""MTI-EEMD prototype on saved centroid_z (task #80).

For each retrospective run, run EEMD on the LiDAR centroid_z signal,
identify the IMF whose dominant peak falls in the RR band with the
greatest in-band power, derive an RR estimate from that IMF, and compare
against (a) the Polar median RR and (b) the peak-pick FFT result we
already computed in phase1/notes/peak_pick_rescore_2026-05-17.{json,md}.

Note on "MTI": in the radar literature, MTI (Moving Target Indicator)
is a high-pass step on slow-time radar samples — radar-specific. The
LiDAR centroid_z pipeline already runs a linear detrend (the equivalent
DC + drift removal) before any spectral analysis. We do the same here
before invoking EEMD, so this is "(LiDAR-equivalent) MTI + EEMD"; the
substantive bit is the EEMD decomposition.

Decision rule for IMF selection: two pickers are computed and the
output reports both side-by-side, because the picker is the load-bearing
design choice — EEMD itself extracts the modes correctly, but selecting
*which* IMF is the respiratory one is non-trivial when several IMFs
straddle the band.

* ``in_band_fraction`` (default, principled): IMF whose fraction of total
  energy inside [10, 30] BPM is maximal, subject to a minimum 0.5 gate.
  This is the published-literature standard — a respiratory IMF should
  be a band-pass mode whose energy is mostly inside the band.
* ``in_band_power`` (naive): IMF with maximal absolute in-band power.
  Fails when a higher-amplitude, higher-frequency IMF leaks a tail into
  the band — its absolute in-band power can exceed a clean respiratory
  IMF's, even though the latter is the "respiratory" one.

Each IMF's RR estimate is the FFT peak of that IMF within [10, 30] BPM.

Requires PyEMD (EMD-signal package). Install in a local venv:
    python3 -m venv .venv-local
    .venv-local/bin/pip install numpy scipy EMD-signal
    .venv-local/bin/python phase1/mti_eemd_prototype.py --runs ...

Usage
-----
    .venv-local/bin/python phase1/mti_eemd_prototype.py \
        --runs 20260517_063204 20260517_064107 ... \
        --out  phase1/notes/mti_eemd_prototype_2026-05-17.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import detrend

# PyEMD is optional — import inside main() so the script imports cleanly
# even when the venv isn't activated. EEMD is required to run anything.

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Knight Vision RR band — see phase1/config.py KVConfig.rr_freq_min/max.
_RR_BAND_BPM = (10.0, 30.0)

# Default EEMD knobs — kept consistent across runs for cross-run
# comparison. Trials=50 is the speed/noise compromise (default 100 is
# slow at ~2-2.5k samples; 50 still gives the literature-standard
# "white noise dithering" benefit).
_EEMD_TRIALS = 50
_EEMD_NOISE_WIDTH = 0.05
_EEMD_MAX_IMF = 8


def polar_median_rr(gt_csv: Path) -> dict:
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
    }


def preprocess(cz: np.ndarray) -> np.ndarray:
    """NaN-interp + linear detrend (matches phase1.respiratory.extract_rr_from_signal)."""
    sig = cz.astype(float).copy()
    nan_mask = np.isnan(sig)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        idx = np.arange(len(sig))
        sig[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], sig[~nan_mask])
    return detrend(sig, type="linear")


def fft_peak_bpm(signal: np.ndarray, fps: float,
                 band_bpm: tuple = _RR_BAND_BPM) -> dict:
    """Hanning-windowed, zero-padded FFT; return in-band peak BPM + power stats."""
    if len(signal) < 32:
        return {"peak_bpm": float("nan"), "in_band_power": 0.0,
                "total_power": 0.0, "in_band_frac": 0.0}
    win = signal * np.hanning(len(signal))
    n_fft = int(2 ** np.ceil(np.log2(len(signal) * 16)))
    spec = np.abs(np.fft.rfft(win, n=n_fft)) ** 2
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    band = (freq * 60.0 >= band_bpm[0]) & (freq * 60.0 <= band_bpm[1])
    if not band.any():
        return {"peak_bpm": float("nan"), "in_band_power": 0.0,
                "total_power": float(spec.sum()), "in_band_frac": 0.0}
    in_band_power = float(spec[band].sum())
    total = float(spec.sum())
    peak_idx = int(np.argmax(spec[band]))
    return {
        "peak_bpm":      float(freq[band][peak_idx] * 60.0),
        "in_band_power": in_band_power,
        "total_power":   total,
        "in_band_frac":  in_band_power / total if total > 0 else 0.0,
        "in_band_snr":   float(spec[band].max()
                               / (spec[band].mean() + 1e-12)),
    }


def eemd_decompose(signal: np.ndarray, fps: float):
    """Run EEMD; return list of IMF arrays (same length as signal)."""
    from PyEMD import EEMD
    eemd = EEMD(trials=_EEMD_TRIALS,
                noise_width=_EEMD_NOISE_WIDTH,
                separate_trends=False)
    # EEMD signature: eemd(S, T=None, max_imf=-1)
    imfs = eemd(signal, max_imf=_EEMD_MAX_IMF)
    return [np.asarray(row) for row in imfs]


_MIN_IN_BAND_FRACTION = 0.5  # gate for the principled picker


def per_imf_features(imfs: list, fps: float) -> list:
    rows = []
    for i, imf in enumerate(imfs):
        feat = fft_peak_bpm(imf, fps)
        feat["imf_idx"] = i
        feat["rms"] = float(np.sqrt(np.mean(imf ** 2)))
        rows.append(feat)
    return rows


def pick_imf(rows: list, rule: str) -> int | None:
    """Pick the respiratory IMF index under the named rule."""
    candidates = [r for r in rows if not np.isnan(r["peak_bpm"])]
    if not candidates:
        return None
    if rule == "in_band_power":
        candidates.sort(key=lambda r: (-r["in_band_power"], r["imf_idx"]))
        return candidates[0]["imf_idx"]
    if rule == "in_band_fraction":
        gated = [r for r in candidates if r["in_band_frac"]
                                          >= _MIN_IN_BAND_FRACTION]
        if not gated:
            # No IMF passes the gate — fall back to highest fraction overall.
            candidates.sort(key=lambda r: (-r["in_band_frac"], r["imf_idx"]))
            return candidates[0]["imf_idx"]
        # Among gated IMFs, prefer the one with highest in-band SNR (peakier
        # in the band = cleaner respiratory mode). Tie → lower idx.
        gated.sort(key=lambda r: (-r["in_band_snr"], r["imf_idx"]))
        return gated[0]["imf_idx"]
    raise ValueError(f"unknown pick rule {rule!r}")


def process_run(run_dir: Path, peak_pick_lookup: dict) -> dict:
    meta_p = run_dir / "meta.json"
    cz_p   = run_dir / "centroid_z.npy"
    gt_p   = run_dir / "gt.csv"
    if not meta_p.exists() or not cz_p.exists():
        return {"run": run_dir.name, "error": "missing meta.json or centroid_z.npy"}

    meta = json.loads(meta_p.read_text())
    cz   = np.load(cz_p)
    fps  = meta["frames_seen"] / meta["wall_seconds"]
    sig  = preprocess(cz)
    polar = polar_median_rr(gt_p) if gt_p.exists() else {
        "rr_median": float("nan"), "rr_n": 0, "hr_median": float("nan"),
    }

    t0 = time.time()
    imfs = eemd_decompose(sig, fps)
    eemd_seconds = time.time() - t0

    rows = per_imf_features(imfs, fps)
    picks = {
        "in_band_fraction": pick_imf(rows, "in_band_fraction"),
        "in_band_power":    pick_imf(rows, "in_band_power"),
    }

    def rr_for_pick(idx):
        if idx is None:
            return None, None
        r = next(rr for rr in rows if rr["imf_idx"] == idx)
        return (round(r["peak_bpm"], 2) if not np.isnan(r["peak_bpm"]) else None,
                round(r.get("in_band_snr", 0.0), 2))

    rr_frac, snr_frac = rr_for_pick(picks["in_band_fraction"])
    rr_pow,  snr_pow  = rr_for_pick(picks["in_band_power"])

    peak_pick = peak_pick_lookup.get(run_dir.name, {})
    polar_rr = polar["rr_median"] if not np.isnan(polar["rr_median"]) else None

    out = {
        "run":          run_dir.name,
        "wall_seconds": meta.get("wall_seconds"),
        "frames_seen":  meta.get("frames_seen"),
        "fps_corrected": round(fps, 3),
        "polar_rr_median": round(polar_rr, 2) if polar_rr is not None else None,
        "polar_hr_median": round(polar["hr_median"], 2)
                           if not np.isnan(polar["hr_median"]) else None,
        "polar_rr_n":      polar["rr_n"],
        "eemd": {
            "trials":      _EEMD_TRIALS,
            "noise_width": _EEMD_NOISE_WIDTH,
            "max_imf":     _EEMD_MAX_IMF,
            "n_imfs":      len(imfs),
            "seconds":     round(eemd_seconds, 2),
        },
        "imfs": [
            {
                "imf_idx":       r["imf_idx"],
                "peak_bpm":      round(r["peak_bpm"], 2) if not np.isnan(r["peak_bpm"]) else None,
                "in_band_frac":  round(r["in_band_frac"], 3),
                "in_band_snr":   round(r.get("in_band_snr", 0.0), 2),
                "rms":           round(r["rms"], 5),
                "picked_frac":   (r["imf_idx"] == picks["in_band_fraction"]),
                "picked_power":  (r["imf_idx"] == picks["in_band_power"]),
            }
            for r in rows
        ],
        "pick_in_band_fraction": {
            "imf_idx": picks["in_band_fraction"],
            "rr_bpm":  rr_frac,
            "in_band_snr": snr_frac,
        },
        "pick_in_band_power": {
            "imf_idx": picks["in_band_power"],
            "rr_bpm":  rr_pow,
            "in_band_snr": snr_pow,
        },
        "peak_pick_rr_bpm": peak_pick.get("rr_bpm"),
        "peak_pick_settled": peak_pick.get("settled"),
        "peak_pick_confidence": peak_pick.get("confidence"),
    }

    if polar_rr is not None:
        if rr_frac is not None:
            out["delta_frac_vs_polar"] = round(rr_frac - polar_rr, 2)
        if rr_pow is not None:
            out["delta_power_vs_polar"] = round(rr_pow - polar_rr, 2)
        if out["peak_pick_rr_bpm"] is not None:
            out["delta_peak_vs_polar"] = round(
                out["peak_pick_rr_bpm"] - polar_rr, 2)

    return out


def load_peak_pick_lookup(path: Path) -> dict:
    """Build {run_name: {rr_bpm, settled, confidence}} from task #77 JSON."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        r["run"]: {
            "rr_bpm":     r["peak_pick"]["rr_bpm"],
            "settled":    r["peak_pick"]["settled"],
            "confidence": r["peak_pick"]["confidence"],
        }
        for r in data if "peak_pick" in r
    }


def fmt_md(results: list[dict]) -> str:
    lines = [
        "# MTI-EEMD prototype on saved centroid_z — 2026-05-17 runs",
        "",
        "*Generated 2026-05-19 by `phase1/mti_eemd_prototype.py` (task #80).*",
        "",
        f"EEMD parameters held constant: trials={_EEMD_TRIALS}, "
        f"noise_width={_EEMD_NOISE_WIDTH}, max_imf={_EEMD_MAX_IMF}.",
        "",
        "Two picker rules are evaluated side-by-side. **frac** = IMF with "
        f"the highest fraction of its energy in [{_RR_BAND_BPM[0]:.0f}, "
        f"{_RR_BAND_BPM[1]:.0f}] BPM (gated at ≥ {_MIN_IN_BAND_FRACTION}, "
        "ties → higher in-band SNR). **pow** = IMF with max absolute "
        "in-band power. Both pick from the *same* EEMD output — the "
        "decomposition itself is identical; the rule is the load-bearing "
        "design choice.",
        "",
        "Peak-pick numbers are sourced from "
        "`peak_pick_rescore_2026-05-17.json` (task #77).",
        "",
        "## Headline table",
        "",
        "| Run | Polar RR | Peak-pick | EEMD-frac IMF | EEMD-frac RR | Δ frac | EEMD-pow IMF | EEMD-pow RR | Δ pow | Δ peak |",
        "|---|---:|---:|:---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['run']} | — | — | — | — | — | — | — | — | ERR: {r['error']} |")
            continue
        d_frac = r.get("delta_frac_vs_polar", "—")
        d_pow  = r.get("delta_power_vs_polar", "—")
        d_peak = r.get("delta_peak_vs_polar", "—")
        lines.append(
            f"| {r['run']} | {r['polar_rr_median']} | "
            f"{r.get('peak_pick_rr_bpm','—')} | "
            f"#{r['pick_in_band_fraction']['imf_idx']} | "
            f"{r['pick_in_band_fraction']['rr_bpm']} | {d_frac} | "
            f"#{r['pick_in_band_power']['imf_idx']} | "
            f"{r['pick_in_band_power']['rr_bpm']} | {d_pow} | {d_peak} |"
        )

    rows = [r for r in results if "delta_frac_vs_polar" in r
                                  and "delta_peak_vs_polar" in r]
    if rows:
        frac_d = np.array([r["delta_frac_vs_polar"] for r in rows])
        pow_d  = np.array([r["delta_power_vs_polar"] for r in rows
                                                     if "delta_power_vs_polar" in r])
        peak_d = np.array([r["delta_peak_vs_polar"] for r in rows])
        n_frac_better = int((np.abs(frac_d) < np.abs(peak_d)).sum())
        n_pow_better  = int((np.abs(pow_d) < np.abs(peak_d)).sum()) if pow_d.size else 0
        lines += [
            "",
            "## Aggregate",
            "",
            f"- n = {len(rows)} paired live captures",
            f"- **EEMD (frac picker) vs Polar:** mean Δ = "
            f"{frac_d.mean():+.2f} BPM, |Δ| median = "
            f"{np.median(np.abs(frac_d)):.2f}, max |Δ| = "
            f"{np.max(np.abs(frac_d)):.2f}",
            f"- **EEMD (power picker) vs Polar:** mean Δ = "
            f"{pow_d.mean():+.2f} BPM, |Δ| median = "
            f"{np.median(np.abs(pow_d)):.2f}, max |Δ| = "
            f"{np.max(np.abs(pow_d)):.2f}",
            f"- **Peak-pick vs Polar (recap):** mean Δ = "
            f"{peak_d.mean():+.2f} BPM, |Δ| median = "
            f"{np.median(np.abs(peak_d)):.2f}, max |Δ| = "
            f"{np.max(np.abs(peak_d)):.2f}",
            f"- EEMD-frac closer to Polar than peak-pick in "
            f"**{n_frac_better}/{len(rows)}** runs",
            f"- EEMD-pow closer to Polar than peak-pick in "
            f"**{n_pow_better}/{len(rows)}** runs",
        ]

    lines += ["", "## Per-run IMF detail", ""]
    for r in results:
        lines.append(f"### {r['run']}")
        lines.append("")
        if "error" in r:
            lines.append(f"ERROR: {r['error']}")
            lines.append("")
            continue
        lines += [
            f"- {r['wall_seconds']:.1f}s · {r['frames_seen']} frames · "
            f"corrected fps {r['fps_corrected']}",
            f"- Polar RR median: **{r['polar_rr_median']}** BPM "
            f"(n={r['polar_rr_n']}); HR median: {r['polar_hr_median']} BPM",
            f"- EEMD: {r['eemd']['n_imfs']} IMFs in {r['eemd']['seconds']}s",
            "",
            "| IMF | Peak BPM | in-band frac | in-band SNR | RMS | frac-pick | pow-pick |",
            "|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
        for imf in r["imfs"]:
            lines.append(
                f"| {imf['imf_idx']} | {imf['peak_bpm'] if imf['peak_bpm'] is not None else '—'} | "
                f"{imf['in_band_frac']} | {imf['in_band_snr']} | {imf['rms']} | "
                f"{'✓' if imf['picked_frac'] else ''} | "
                f"{'✓' if imf['picked_power'] else ''} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--peak-pick-json", type=Path,
                    default=_HERE / "notes" / "peak_pick_rescore_2026-05-17.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    try:
        from PyEMD import EEMD  # noqa: F401
    except ImportError as e:
        print(f"PyEMD not available: {e}\nInstall with: "
              ".venv-local/bin/pip install EMD-signal", file=sys.stderr)
        sys.exit(2)

    peak_pick_lookup = load_peak_pick_lookup(args.peak_pick_json)
    if not peak_pick_lookup:
        print(f"WARN: no peak-pick reference at {args.peak_pick_json}; "
              "comparison columns will be empty.", file=sys.stderr)

    results = []
    for arg in args.runs:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = _HERE / "runs" / arg
        print(f"[mti-eemd] processing {p.name} ...", file=sys.stderr)
        results.append(process_run(p, peak_pick_lookup))

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
