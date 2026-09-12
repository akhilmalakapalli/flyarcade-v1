"""Replay every completed trial in a separate guarded process and record verdicts."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    # v1 trials only. v1.1 lives under runs/v11-* with a different checkpoint
    # format and its own determinism check; this script verifies the frozen v1 grid.
    names = sorted(
        p.parent.name
        for p in Path("runs").glob("*/result.json")
        if not p.parent.name.startswith("v11-")
    )
    if not names:
        raise SystemExit("no completed trials found")
    verdicts = []
    for i, name in enumerate(names):
        print(f"Replay {i + 1}/{len(names)}: {name}", flush=True)
        proc = subprocess.run(
            [sys.executable, "scripts/replay_trial.py", "--name", name],
            timeout=135,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"replay failed for {name}")
        verdicts.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    report = {
        "trials_replayed": len(verdicts),
        "all_match": all(v["status"] == "MATCH" for v in verdicts),
        "scope": (
            "Bit-identical replay of held-out, sensory-noise and ablation evaluations "
            "from each saved final checkpoint. Pre-training evaluations predate the "
            "surviving checkpoint and are not replayed."
        ),
        "max_replay_seconds": max(v["resources"]["elapsed_seconds"] for v in verdicts),
        "max_rss_snapshot_mb": max(v["resources"]["rss_mb"] for v in verdicts),
        "verdicts": verdicts,
    }
    Path("artifacts/replay_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in list(report)[:5]}, indent=2))


if __name__ == "__main__":
    main()
