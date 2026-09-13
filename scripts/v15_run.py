"""Run v1.5 trial specs in parallel single-threaded workers, resuming checkpointed ones."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_SEGMENTS = 60


ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def run(spec):
    spec = Path(spec)
    name = json.loads(spec.read_text())["name"]
    failure = Path("runs/v15") / name / "failure.json"
    if failure.exists():
        return name, "FAILED(preserved)"
    log = Path("runs/v15") / name / "log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    for segment in range(MAX_SEGMENTS):
        with log.open("a") as stream:
            p = subprocess.run(
                [sys.executable, "scripts/v15_trial.py", "--spec", str(spec)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=150,
                env=ENV,
            )
        if p.returncode == 0:
            return name, "COMPLETE"
        if p.returncode != 7:
            reason = f"exit {p.returncode} (see log.txt)"
            break
    else:
        reason = f"{MAX_SEGMENTS}-segment limit"
    failure.write_text(
        json.dumps({"status": "FAILED", "name": name, "failure_reason": reason}) + "\n"
    )
    return name, "FAILED " + reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        for name, status in pool.map(run, args.specs):
            print(f"[{time.monotonic() - started:7.0f}s] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
