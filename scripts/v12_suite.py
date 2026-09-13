"""Sequential confirmatory grid, explicitly resuming bounded checkpoint segments."""

import json
import subprocess
import sys
from pathlib import Path


def main():
    plan = json.loads(Path("experiments/v12_run_plan.json").read_text())
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            for seed in plan["training_seeds"]:
                for segment in range(20):
                    command = [
                        sys.executable,
                        "scripts/v12_trial.py",
                        "--task",
                        task,
                        "--condition",
                        condition,
                        "--seed",
                        str(seed),
                    ]
                    result = subprocess.run(command, timeout=125)
                    if result.returncode == 0:
                        break
                    if result.returncode != 7:
                        raise SystemExit(result.returncode)
                else:
                    raise SystemExit("20-segment guard reached; inspect checkpoints")


if __name__ == "__main__":
    main()
