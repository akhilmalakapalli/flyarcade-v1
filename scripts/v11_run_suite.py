"""Run the frozen v1.1 confirmatory grid sequentially; each trial keeps its guard."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    plan = json.loads(Path("experiments/v11_run_plan.json").read_text())
    trials = [
        ["--task", task, "--condition", condition, "--seed", str(seed)]
        for task in plan["tasks"]
        for condition in plan["conditions"]
        for seed in plan["seeds"]
    ]
    for i, args in enumerate(trials):
        print(f"Trial {i + 1}/{len(trials)}: {' '.join(args)}", flush=True)
        result = subprocess.run([sys.executable, "scripts/v11_run_trial.py", *args], timeout=300)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
