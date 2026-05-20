"""Single-command quality check for an active Knight Vision viewer session.

Fetches /diag from the live Flask viewer, grabs a topdown JPEG so Dev can
make a visual ROI assessment, optionally tails per_frame.log from the
current run directory, compares against the Run 4 baseline, and emits a
single-line headline plus a structured markdown report.

Intended invocations:

  - Manual:
      .venv-local/bin/python phase1/quality_check.py
  - With a recording run dir to also pull per_frame.log:
      .venv-local/bin/python phase1/quality_check.py \
          --run-dir phase1/runs/20260520_103430
  - Auto-fired by phase1/qc_monitor.py when a new run_dir lands or
    when SNR transitions across MEDIUM/HIGH thresholds mid-run.

Outputs:

  - Markdown report to ``--out`` (default
    ``/tmp/qc_<utc_ts>/quality_report.md``).
  - JSON sidecar alongside.
  - Topdown JPEG saved alongside.
  - Single-line headline to stdout (last line) — designed for ``tail -1``
    if you want to grab just the headline for chat.

Exit codes:

  0 — clean (no flags)
  1 — warnings (flags fired but not catastrophic)
  2 — critical (no_subject_locked or no /diag reachable)
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


VIEWER_DEFAULT = "http://192.168.1.90:5005"
JETSON_DEFAULT = "phil@192.168.1.90"


# Severity ranking — first match wins for the headline.
_CRITICAL_FLAGS = ("no_subject_locked",)
_WARN_FLAGS = (
    "cz_std_high", "cz_std_too_low", "residuals_too_high",
    "cluster_count_high", "snr_low", "rr_unstable", "fps_low",
)


def http_get(url: str, timeout: float = 4.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url: str, timeout: float = 4.0, max_bytes: int = 1_500_000) -> bytes:
    socket.setdefaulttimeout(timeout)
    with urllib.request.urlopen(url) as r:
        chunks: list[bytes] = []
        while sum(len(c) for c in chunks) < max_bytes:
            try:
                c = r.read(65536)
                if not c:
                    break
                chunks.append(c)
            except socket.timeout:
                break
        return b"".join(chunks)


def extract_last_jpeg(stream_bytes: bytes) -> bytes | None:
    """Pull the most recent complete JPEG from an MJPEG multipart stream."""
    soi = [m.start() for m in re.finditer(b"\xff\xd8", stream_bytes)]
    eoi = [m.start() for m in re.finditer(b"\xff\xd9", stream_bytes)]
    if not soi or not eoi:
        return None
    # Try last SOI first; fall back to second-to-last (last may be incomplete).
    for s in reversed(soi):
        after = [e for e in eoi if e > s]
        if after:
            return stream_bytes[s : after[0] + 2]
    return None


def ssh_tail_log(jetson: str, run_dir: str, n_lines: int = 10) -> str:
    """Tail the last N lines of per_frame.log on the Jetson via SSH."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=4",
        jetson,
        f"tail -n {n_lines} {run_dir}/per_frame.log 2>/dev/null || echo '(no per_frame.log)'",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return out.stdout.strip() or out.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"(ssh tail failed: {e})"


def find_active_run_dir(jetson: str) -> str | None:
    """Best-effort: most-recently-modified phase1/runs/<ts>/ on Jetson."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=4",
        jetson,
        "ls -dt ~/knight-vision/phase1/runs/2*/ 2>/dev/null | head -1",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        line = out.stdout.strip()
        return line.rstrip("/") if line else None
    except Exception:
        return None


def headline(diag: dict, has_subject: bool, jpeg_present: bool) -> tuple[str, int]:
    """Build the single-line chat headline + exit code."""
    flags = diag.get("flags", [])
    bits: list[str] = []

    # Severity prefix
    if "no_subject_locked" in flags:
        bits.append("⛔ no subject")
        exit_code = 2
    elif any(f in flags for f in _WARN_FLAGS):
        bits.append("⚠️")
        exit_code = 1
    else:
        bits.append("✅ QC")
        exit_code = 0

    # Quick numeric snippet
    snippet: list[str] = []
    if diag.get("cz_std_mm") is not None:
        snippet.append(f"cz_std {diag['cz_std_mm']:.1f}mm")
    if diag.get("residual_ratio") is not None:
        snippet.append(f"resid {diag['residual_ratio']:.2f}")
    if diag.get("n_clusters") is not None:
        snippet.append(f"clust {diag['n_clusters']}")
    if diag.get("rr_snr") is not None:
        snippet.append(f"SNR {diag['rr_snr']:.1f}")
    if diag.get("rr_bpm") is not None:
        snippet.append(f"RR {diag['rr_bpm']:.1f}")
    if diag.get("fps") is not None:
        snippet.append(f"fps {diag['fps']:.1f}")
    bits.append("·".join(snippet))

    # Pull leading flags into the headline
    leading = [f for f in flags if f in _CRITICAL_FLAGS] + [
        f for f in flags if f in _WARN_FLAGS
    ]
    if leading:
        bits.append("flags=" + ",".join(leading[:3]))

    if not jpeg_present:
        bits.append("(no JPEG)")

    return " · ".join(bits), exit_code


def fmt_md(diag: dict, jpeg_path: Path | None, run_dir: str | None,
           log_tail: str | None, head: str) -> str:
    base = diag.get("baseline_run4", {})
    delta = diag.get("delta_vs_run4", {})
    flags = diag.get("flags", [])
    rr_hist = diag.get("rr_history", []) or []
    recent = rr_hist[-8:]

    lines = [
        f"# Quality check — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"**Headline:** {head}",
        "",
    ]
    if jpeg_path is not None:
        lines += [
            f"**Topdown JPEG:** `{jpeg_path}`",
            "*(Dev: view this with the Read tool to make the ROI placement call — "
            "chest? shoulder? head? behind subject? outside subject?)*",
            "",
        ]
    else:
        lines += ["**Topdown JPEG:** not fetched", ""]

    lines += [
        "## /diag snapshot",
        "",
        f"- frame_no: {diag.get('frame_no')}  · fps: {diag.get('fps')}  · "
        f"recording: {diag.get('recording')}  (record_seq {diag.get('record_seq')})",
        f"- subject_pts: {diag.get('subject_pts')}  · chest_pts: "
        f"{diag.get('chest_pts')}  · cz: {diag.get('cz')}",
        f"- cz_std: **{diag.get('cz_std_mm')} mm** (tail n={diag.get('cz_tail_n')})  · "
        f"cz_span: {diag.get('cz_span_mm')} mm",
        f"- n_residuals: {diag.get('n_residuals')} / n_pts {diag.get('n_pts')}  → "
        f"residual_ratio **{diag.get('residual_ratio')}**",
        f"- n_clusters: **{diag.get('n_clusters')}**",
        f"- RR ({diag.get('rr_method')}): **{diag.get('rr_bpm')} BPM** · "
        f"peak {diag.get('rr_bpm_peak')} · centroid {diag.get('rr_bpm_centroid')}",
        f"- SNR: **{diag.get('rr_snr')}**  · conf: {diag.get('rr_conf')}  · "
        f"rr_n: {diag.get('rr_n')}",
        f"- radar: RR {diag.get('radar_rr_bpm')} · SNR "
        f"{diag.get('radar_rr_snr')} · n {diag.get('radar_rr_n')}",
        f"- GT: {diag.get('gt')}",
        f"- flags: `{flags or 'none'}`",
        "",
        "## Δ vs Run 4 baseline",
        "",
        f"| Metric | Run 4 | Current | Δ |",
        f"|---|---:|---:|---:|",
        f"| cz_std (mm)      | {base.get('cz_std_mm')} | {diag.get('cz_std_mm')} | "
        f"{delta.get('cz_std_mm_delta')} |",
        f"| settled SNR      | {base.get('settled_snr')} | {diag.get('rr_snr')} | "
        f"{delta.get('snr_delta')} |",
        f"| n_clusters       | {base.get('n_clusters')} | {diag.get('n_clusters')} | "
        f"{delta.get('n_clusters_delta')} |",
        f"| residual_ratio   | {round(base.get('residual_ratio', 0), 3)} | "
        f"{diag.get('residual_ratio')} | {delta.get('residual_ratio_delta')} |",
        "",
    ]

    if recent:
        lines += [
            "## Last-K RR history (most recent windows)",
            "",
            "| t (relative) | RR | peak | centroid | SNR | conf |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
        t0 = recent[0]["t"]
        for r in recent:
            lines.append(
                f"| {r['t']-t0:+.1f}s | {r['rr_bpm']:.2f} | "
                f"{r.get('peak')} | {r.get('centroid')} | "
                f"{r['snr']:.2f} | {r['conf']} |"
            )
        lines.append("")

    if log_tail:
        lines += [
            f"## per_frame.log tail — {run_dir or '(no run dir)'}",
            "",
            "```",
            log_tail,
            "```",
            "",
        ]

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default=VIEWER_DEFAULT,
                    help="Base URL of the Jetson viewer (default %(default)s)")
    ap.add_argument("--jetson", default=JETSON_DEFAULT,
                    help="SSH target for per_frame.log tail (default %(default)s)")
    ap.add_argument("--run-dir", default=None,
                    help="Recording run dir path on the Jetson. "
                         "Omit to auto-discover the most-recent runs/<ts>/.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown report path (default /tmp/qc_<ts>/quality_report.md)")
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out.parent if args.out
               else Path(f"/tmp/qc_{ts}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md   = args.out or (out_dir / "quality_report.md")
    out_json = out_dir / "diag.json"
    jpeg_p   = out_dir / "topdown.jpg"

    # 1. /diag
    try:
        diag = http_get(f"{args.viewer}/diag")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("⛔ /diag endpoint missing — viewer needs restart with latest viewer.py",
                  file=sys.stderr)
            sys.exit(2)
        raise
    except Exception as e:
        print(f"⛔ Cannot reach /diag at {args.viewer}: {e}", file=sys.stderr)
        sys.exit(2)

    # 2. JPEG topdown
    jpeg_present = False
    try:
        raw = http_get_bytes(f"{args.viewer}/topdown_stream", timeout=4.0)
        jpg = extract_last_jpeg(raw)
        if jpg:
            jpeg_p.write_bytes(jpg)
            jpeg_present = True
    except Exception as e:
        print(f"(jpeg fetch failed: {e})", file=sys.stderr)

    # 3. per_frame.log tail (optional)
    run_dir = args.run_dir
    if run_dir is None and diag.get("recording"):
        run_dir = find_active_run_dir(args.jetson)
    log_tail = None
    if run_dir:
        log_tail = ssh_tail_log(args.jetson, run_dir, n_lines=10)

    # 4. compose headline + report
    has_subject = bool(diag.get("subject_pts", 0))
    head, exit_code = headline(diag, has_subject, jpeg_present)
    md = fmt_md(diag, jpeg_p if jpeg_present else None,
                run_dir, log_tail, head)

    out_md.write_text(md)
    out_json.write_text(json.dumps(diag, indent=2, default=str))
    print(f"# wrote {out_md}", file=sys.stderr)
    print(f"# wrote {out_json}", file=sys.stderr)
    if jpeg_present:
        print(f"# wrote {jpeg_p}", file=sys.stderr)
    print(head)            # last line — what tail -1 sees
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
