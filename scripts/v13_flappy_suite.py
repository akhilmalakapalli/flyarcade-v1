"""Run Flappy rescue specs through the rescue trial wrapper, resuming on exit 7.

Same contract as scripts/v13_suite.py (fresh guarded process per segment, completed
results preserved, spent externally frozen confirmatory seeds refused).
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
    log = Path(log_dir) / f"{Path(spec_path).stem}.log"
    started = time.time()
    for segment in range(MAX_SEGMENTS):
        with open(log, "a") as handle:
            code = subprocess.run(
                [sys.executable, "scripts/v13_flappy_trial.py", "--spec", str(spec_path)],
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
            return {"spec": str(spec_path), "status": f"FAILED exit {code}", "log": str(log)}
    return {"spec": str(spec_path), "status": "SEGMENT_LIMIT", "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log-dir", default="runs/v13-flappy-rescue/logs")
    args = parser.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    specs = sorted(args.specs)
    refused = [s for s in specs if spent(json.loads(Path(s).read_text()))]
    if refused:
        print(json.dumps({"refused_spent_confirmatory_specs": refused}), flush=True)
        return 2
    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = [f.result() for f in [pool.submit(run, s, args.log_dir) for s in specs]]
    for o in outcomes:
        print(json.dumps(o), flush=True)
    failed = [o for o in outcomes if o["status"] != "COMPLETE"]
    print(json.dumps({"complete": len(outcomes) - len(failed), "failed": len(failed)}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
