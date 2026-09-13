"""Run many v1.3 trial specs in parallel single-threaded processes.

Each trial is re-invoked in a fresh, independently guarded process whenever it exits
7 after a safe checkpoint; completed results are preserved. Failures are recorded,
never retried silently beyond the segment limit.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v13_external import spent

ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
MAX_SEGMENTS = 400


def run(spec_path, log_dir):
    name = Path(spec_path).stem
    log = Path(log_dir) / f"{name}.log"
    started = time.time()
    for segment in range(MAX_SEGMENTS):
        with open(log, "a") as handle:
            code = subprocess.run(
                [sys.executable, "scripts/v13_trial.py", "--spec", str(spec_path)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=ENV,
            ).returncode
        if code == 0:
            return {
                "spec": str(spec_path),
                "status": "COMPLETE",
                "segments": segment + 1,
                "seconds": time.time() - started,
            }
        if code != 7:
            return {
                "spec": str(spec_path),
                "status": f"FAILED exit {code}",
                "segments": segment + 1,
                "log": str(log),
            }
    return {"spec": str(spec_path), "status": "SEGMENT_LIMIT", "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-dir", default="runs/v13-logs")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    specs = sorted(args.specs)
    # Externally frozen tasks (Pong, see experiments/v13_external_frozen.json) have
    # already spent their confirmatory seeds. No override exists: reproducing that
    # result is done by exact policy replay, never by retraining on those seeds.
    refused = [s for s in specs if spent(json.loads(Path(s).read_text()))]
    if refused:
        print(json.dumps({"refused_spent_confirmatory_specs": refused}), flush=True)
        return 2
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, s, args.log_dir) for s in specs]
        outcomes = []
        for future in futures:
            outcome = future.result()
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
    failed = [o for o in outcomes if o["status"] != "COMPLETE"]
    print(json.dumps({"complete": len(outcomes) - len(failed), "failed": len(failed)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
