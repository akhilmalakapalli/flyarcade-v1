"""Run Snake trials sequentially, re-invoking any trial that asked to resume."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESUME_EXIT = 7


def run(condition, seed, mode, plan, max_segments=40):
    name = ("snake" if mode == "confirm" else "snake-dev") + f"-{condition}-{seed}"
    for _ in range(max_segments):
        if (Path("runs") / name / "result.json").exists():
            return
        result = subprocess.run(
            [
                sys.executable,
                "scripts/snake_run_trial.py",
                "--condition",
                condition,
                "--seed",
                str(seed),
                "--mode",
                mode,
                "--plan",
                plan,
            ],
            timeout=300,
        )
        if result.returncode == RESUME_EXIT:
            continue
        if result.returncode:
            raise SystemExit(result.returncode)
        return
    raise SystemExit(f"{name}: exceeded the resume-segment budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dev", "confirm"], default="confirm")
    parser.add_argument("--plan", default="experiments/snake_run_plan.json")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    conditions = args.conditions or plan["conditions"]
    seeds = args.seeds if args.seeds is not None else plan["seeds"]
    for condition in conditions:
        for seed in seeds:
            print(f"--- snake {args.mode} {condition} seed {seed}", flush=True)
            run(condition, seed, args.mode, args.plan)


if __name__ == "__main__":
    main()
