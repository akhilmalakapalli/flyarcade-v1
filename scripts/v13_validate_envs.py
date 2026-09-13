"""M1: validate target environments and the Flappy solvability gate on fresh seeds.

Writes artifacts/v13/environment_validation.json. Uses only env_validation seeds.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade_v13.environments import LEVELS, TARGETS, make_env, reference_action
from flyarcade_v13.study import digest, evaluate_baseline, seed_for

EPISODES = 100
OUTPUT = Path("artifacts/v13/environment_validation.json")


def summary(rows):
    s = np.array([r["success"] for r in rows])
    return {"mean_success": float(s.mean()), "sd": float(s.std(ddof=1)), "episodes": len(rows)}


def main():
    report = {"episodes_per_condition": EPISODES, "tasks": {}}
    for task in TARGETS:
        seeds = [seed_for(task, "env_validation", index=i) for i in range(EPISODES)]
        entry = {}
        for kind in ("random", "heuristic_v12", "reference"):
            rows = evaluate_baseline(task, seeds, kind)
            entry[kind] = summary(rows)
            if task == "flappy":
                entry[kind]["deaths"] = {
                    str(k): sum(r["death"] == k for r in rows) for k in {r["death"] for r in rows}
                }
                entry[kind]["obstacles_passed_distribution"] = np.bincount(
                    [r["obstacles_passed"] for r in rows], minlength=7
                ).tolist()
        # Determinism: identical trajectories from identical seeds.
        traces = []
        for _ in range(2):
            h = hashlib.sha256()
            for s in seeds[:10]:
                env = make_env(task, s)
                while not env.done:
                    obs = env.observe()
                    assert np.isfinite(obs).all() and obs.min() >= 0 and obs.max() <= 1
                    h.update(obs.tobytes())
                    env.step(reference_action(env))
            traces.append(h.hexdigest())
        entry["deterministic_trajectory_sha256"] = traces[0]
        entry["deterministic"] = traces[0] == traces[1]
        entry["curriculum_levels"] = {
            level: summary([_run(make_env(task, s, level)) for s in seeds[:40]])
            for level in LEVELS[task]
            if level != "target"
        }
        report["tasks"][task] = entry
    flappy = report["tasks"]["flappy"]["reference"]["mean_success"]
    report["flappy_gate"] = {
        "required": 0.70,
        "preferred": 0.80,
        "oracle_mean_success": flappy,
        "passed": flappy >= 0.70,
        "dynamics_changed": False,
        "note": "short-horizon MPC over exact deterministic dynamics of the visible obstacle; "
        "unchanged v1.2 Flappy. The v1.2 reactive heuristic overshoots the gap.",
    }
    report["target_source_sha256"] = digest("src/flyarcade/v12/environments.py")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                t: {
                    k: v["mean_success"]
                    for k, v in e.items()
                    if isinstance(v, dict) and "mean_success" in v
                }
                for t, e in report["tasks"].items()
            },
            indent=1,
        )
    )
    print("flappy gate", report["flappy_gate"]["passed"], flappy)


def _run(env):
    while not env.done:
        env.step(reference_action(env))
    return env.metrics()


if __name__ == "__main__":
    main()
