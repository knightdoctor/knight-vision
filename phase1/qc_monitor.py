"""Background QC monitor — auto-fires phase1/quality_check.py on events.

Two triggers:

* **New run dir** — polls the Jetson's ``phase1/runs/`` over SSH every
  ``--poll-runs-s`` seconds; when a new ``<ts>/`` appears, runs
  quality_check.py against the live viewer and writes a fresh report
  tagged with the new run id.
* **SNR threshold crossing** — polls ``/diag`` every ``--poll-diag-s``
  seconds and tracks the most-recent ``rr_snr``; fires quality_check.py
  whenever SNR transitions across MEDIUM (3.0) or HIGH (5.0).

Designed to run on the Mac alongside the Polar bridge; runs forever
until killed. Each fired headline is appended to ``--log-file`` (default
``/tmp/qc_monitor.log``) and printed to stdout for live tailing.

Usage:
    .venv-local/bin/python phase1/qc_monitor.py &
    tail -f /tmp/qc_monitor.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_QC = _HERE / "quality_check.py"


VIEWER_DEFAULT = "http://192.168.1.90:5005"
JETSON_DEFAULT = "phil@192.168.1.90"


# Thresholds aligned with KVConfig.snr_*_threshold (config.py).
SNR_MEDIUM = 3.0
SNR_HIGH   = 5.0


def list_run_dirs(jetson: str) -> set[str]:
    """Set of run-dir basenames currently on the Jetson."""
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", jetson,
        "ls ~/knight-vision/phase1/runs/ 2>/dev/null",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return {ln.strip() for ln in out.stdout.splitlines()
                if ln.strip() and ln.strip().startswith("2026")}
    except Exception:
        return set()


def snr_bucket(snr: float | None) -> str:
    if snr is None:
        return "unknown"
    if snr >= SNR_HIGH:
        return "HIGH"
    if snr >= SNR_MEDIUM:
        return "MEDIUM"
    return "LOW"


def fire(reason: str, viewer: str, jetson: str, log_path: Path,
         run_dir: str | None, python_bin: str) -> None:
    """Run quality_check.py and append headline to the log + stdout."""
    args = [python_bin, str(_QC), "--viewer", viewer, "--jetson", jetson]
    if run_dir:
        args += ["--run-dir", run_dir]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30)
        headline = (out.stdout.splitlines()[-1] if out.stdout.strip()
                    else "(no headline)")
    except subprocess.TimeoutExpired:
        headline = "(quality_check timed out)"
    line = f"[{ts}] {reason}: {headline}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", default=VIEWER_DEFAULT)
    ap.add_argument("--jetson", default=JETSON_DEFAULT)
    ap.add_argument("--poll-runs-s", type=float, default=5.0)
    ap.add_argument("--poll-diag-s", type=float, default=4.0)
    ap.add_argument("--log-file", type=Path,
                    default=Path("/tmp/qc_monitor.log"))
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to run quality_check.py")
    args = ap.parse_args()

    print(f"[qc_monitor] started; viewer={args.viewer} jetson={args.jetson}",
          flush=True)
    print(f"[qc_monitor] log → {args.log_file}", flush=True)

    known_runs = list_run_dirs(args.jetson)
    print(f"[qc_monitor] initial run dirs: {len(known_runs)}", flush=True)

    last_snr_bucket: str | None = None
    last_runs_poll = 0.0
    last_diag_poll = 0.0

    while True:
        now = time.time()

        # New-run-dir trigger
        if now - last_runs_poll >= args.poll_runs_s:
            last_runs_poll = now
            current = list_run_dirs(args.jetson)
            new = current - known_runs
            if new:
                for r in sorted(new):
                    fire(f"new_run_dir={r}", args.viewer, args.jetson,
                         args.log_file,
                         f"~/knight-vision/phase1/runs/{r}", args.python)
                known_runs = current

        # SNR-threshold-cross trigger
        if now - last_diag_poll >= args.poll_diag_s:
            last_diag_poll = now
            try:
                req = urllib.request.Request(
                    f"{args.viewer}/diag",
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    diag = json.loads(r.read())
                snr = diag.get("rr_snr")
                bucket = snr_bucket(snr)
                if last_snr_bucket is not None and bucket != last_snr_bucket:
                    fire(f"snr_cross {last_snr_bucket}→{bucket} (snr={snr})",
                         args.viewer, args.jetson, args.log_file,
                         None, args.python)
                last_snr_bucket = bucket
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError):
                pass  # viewer down / restarting — try again next tick

        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[qc_monitor] stopped", flush=True)
