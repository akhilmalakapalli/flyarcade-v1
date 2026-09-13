"""Freeze the bounded development selection before new held-out experiments."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import PURPOSES, code_files, code_hash, seed_for


def main():
    output = Path("experiments/v12_run_plan.json")
    if output.exists():
        raise SystemExit("Frozen plan already exists; preserved.")
    development = json.loads(Path("experiments/v12_development_plan.json").read_text())
    definitions = {
        "flappy": {
            "primary": "obstacles_passed / 6",
            "reward": "+0.01 survival +0.02 distance improvement +1 pipe; -1 death",
            "horizon": 138,
            "dynamics": "gravity -0.009; flap velocity 0.055; speed 0.04; gap halfwidth 0.16",
        },
        "pong": {
            "primary": "hits / attempts",
            "reward": "+1 hit, -1 miss; +0.02 intercept error reduction",
            "horizon": 144,
            "dynamics": "paddle speed 0.07; halfwidth 0.14; vx -0.06; repeated serves",
        },
        "breakout": {
            "primary": "bricks_destroyed / 5",
            "reward": "+1 brick, +0.1 interception, -1 miss; +0.01 intercept error improvement",
            "horizon": 160,
            "dynamics": "five bricks; paddle speed 0.07; halfwidth 0.14; three misses end episode",
        },
        "snake": {
            "primary": "min(food_collected / 10, 1)",
            "reward": "-0.01 step, +1 food, -1 collision",
            "horizon": 120,
            "dynamics": (
                "6x6; relative left/straight/right; starvation 40; tail vacates unless growing"
            ),
        },
    }
    tasks = {}
    devhashes = {}
    for task, env in TASKS.items():
        reports = []
        for candidate in range(2):
            path = Path(f"artifacts/v12/development-{task}-{candidate}.json")
            report = json.loads(path.read_text())
            reports.append(sum(r["success"] for r in report["after"]) / len(report["after"]))
            devhashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        chosen = max(range(2), key=lambda i: reports[i])
        tasks[task] = {
            **definitions[task],
            "actions": env.actions,
            "observation_width": env.width,
            "training_episodes": development["confirmation_budgets"][task],
            "selected_candidate": chosen,
            "development_means": reports,
            "hyperparameters": development["candidates"][task][chosen],
            "feature_fitting_seeds": [seed_for(task, "fit", index=i) for i in range(24)],
        }
    stdpaths = sorted(Path("artifacts/v12").glob("*-standardizer.json"))
    assert len(stdpaths) == 16
    plan = {
        "version": "flyarcade-v1.2-stage-a",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "HHMI Janelia MaleCNS male-cns:v1.0",
        "source_graph_sha256": hashlib.sha256(
            Path("data/malecns-v1.0/graph.npz").read_bytes()
        ).hexdigest(),
        "tasks": tasks,
        "conditions": ["biological", "frozen", "rewired"],
        "reference_policies": ["random", "heuristic"],
        "training_seeds": [0, 1, 2],
        "evaluation_episodes": 40,
        "seed_rule": (
            "100000000 + task_index*10000000 + purpose_offset + seed*10000 + episode; "
            "task order is tasks key order"
        ),
        "purpose_offsets": PURPOSES,
        "development_episodes": 300,
        "development_eval_episodes": 20,
        "development_selection": development["selection_rule"],
        "success_criteria": {
            "scope": "every biological training seed must meet all three tests",
            "heldout_improvement": "after > before",
            "beats_random": "after > random + 0.05",
            "state_dependence": "fixed-history observation intervention action diversity > 0.05",
        },
        "perturbations": {
            "sensory_noise": (
                "independent Gaussian SD 0.1 on normalized observations, clipped [0,1]"
            ),
            "edge_ablation": "Bernoulli 10%, fixed mask per task/trained seed",
            "neuron_ablation": (
                "Bernoulli 10%, fixed mask per task/trained seed; follows edge draws"
            ),
            "retraining": False,
            "evaluation": (
                "all biological policies, including unsuccessful ones; interpretation "
                "limited for failures"
            ),
        },
        "topology_control": (
            "directed double-edge swaps, 10 attempts per edge, seeds 700+seed; preserves "
            "in/out degree and outgoing count assignments, not incoming strength; not "
            "uniform random sampling"
        ),
        "architecture": (
            "frozen MaleCNS LIF recurrent core; standardized descending features; "
            "trainable artificial actor-critic; greedy held-out evaluation"
        ),
        "core_optimization": (
            "v12 fixed CSR multiplication verified bit-identical to historical "
            "edge-bincount implementation on biological and rewired graphs, with masks, "
            "300 ticks"
        ),
        "representation": (
            "post-hoc matched exogenous behavior; separate seed block; no tuning from results"
        ),
        "uncertainty": "descriptive bootstrap over three independent training seeds; no p-values",
        "resource_policy": (
            "unchanged guards; independent <=120 second segments checkpoint at "
            "episode/evaluation boundaries after 70 seconds; at most 20 segments per "
            "trial"
        ),
        "historical": (
            "Catch/Dodge v1 and v1.1 and historical absolute-action Snake unchanged and not retuned"
        ),
        "code_sha256": code_hash(),
        "code_files": [str(p) for p in code_files()],
        "standardizer_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in stdpaths
        },
        "development_hashes": devhashes,
    }
    output.write_text(json.dumps(plan, indent=2) + "\n")
    Path("artifacts/v12/environment_definitions.json").write_text(
        json.dumps(tasks, indent=2) + "\n"
    )
    print("Frozen", output, plan["code_sha256"])


if __name__ == "__main__":
    main()
