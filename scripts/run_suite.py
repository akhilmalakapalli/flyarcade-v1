"""Run the frozen trial grid sequentially; each trial has its own 120 s guard."""

import subprocess
import sys


def main():
    trials = []
    for task in ("catch", "dodge"):
        for condition in ("learning", "frozen", "rewired", "readout_only"):
            for seed in range(3):
                trials.append(["--task", task, "--condition", condition, "--seed", str(seed)])
        for seed in range(3):
            trials.append(["--task", task, "--seed", str(seed), "--episodes", "100"])
            trials.append(
                [
                    "--task",
                    task,
                    "--seed",
                    str(seed),
                    "--source",
                    "dodge" if task == "catch" else "catch",
                ]
            )
    for i, args in enumerate(trials):
        print(f"Trial {i + 1}/{len(trials)}: {' '.join(args)}", flush=True)
        result = subprocess.run([sys.executable, "scripts/run_trial.py", *args], timeout=135)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
