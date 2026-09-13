"""M9: freeze the v1.3 confirmatory protocol and write its trial specs.

Reads artifacts/v13/development/selection.json (written only after every task passed
the development gate), hashes code, graphs, standardizers and environments, writes
experiments/v13_run_plan.json and one spec per confirmatory trial. Refuses to overwrite.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from flyarcade_v13.environments import TARGETS
from flyarcade_v13.gates import CONSTANT_ACTION, ENTROPY_FLOOR, GAP_CLOSURE, MARGIN, PROBE
from flyarcade_v13.study import (
    CONFIRMATORY_PURPOSES,
    code_files,
    code_hash,
    digest,
    graph_hash,
    load_topology,
    standardizer_path,
)

PLAN = Path("experiments/v13_run_plan.json")
SELECTION = Path("artifacts/v13/development/selection.json")
SPECS = Path("artifacts/v13/confirmatory/specs")
SEEDS = [0, 1, 2, 3, 4]
EVAL_EPISODES = 100
CONF = {
    "train_purpose": "conf_train",
    "eval_purpose": "conf_eval",
    "model_purpose": "conf_model",
    "rollout_purpose": "conf_rollout",
}


def main():
    if PLAN.exists():
        print(f"{PLAN}: already frozen; refusing to overwrite")
        return 1
    selection = json.loads(SELECTION.read_text())
    if not all(selection["development_gate_passed"][t] for t in TARGETS):
        raise SystemExit("development gate not passed for every task; freeze refused")
    status = subprocess.run(["git", "status", "--porcelain", "src/flyarcade_v13", "scripts/v13_trial.py"], capture_output=True, text=True).stdout
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tasks = {}
    standardizers = {}
    graphs = {"biological": graph_hash(load_topology("biological"))}
    for seed in SEEDS:
        graphs[f"rewired-{seed}"] = graph_hash(load_topology(f"rewired-{seed}"))
    for task in TARGETS:
        chosen = selection["tasks"][task]
        fly = {k: v for k, v in chosen["fly_config"].items() if k not in ("seed", "task")}
        sensory = {k: v for k, v in chosen["sensory_config"].items() if k not in ("seed", "task")}
        fly.update(CONF)
        sensory.update(CONF)
        for topology in ["biological"] + [f"rewired-{s}" for s in SEEDS]:
            path = standardizer_path(task, topology, fly["ticks"], fly["readout"])
            standardizers[str(path)] = digest(path)
        tasks[task] = {
            "architecture": chosen["architecture"],
            "fly_config": fly,
            "sensory_config": sensory,
            "development_evidence": chosen["evidence"],
            "environment_class": f"flyarcade.v12.environments.{TARGETS[task].__name__}",
            "observation_width": TARGETS[task].width,
            "actions": TARGETS[task].actions,
            "horizon": TARGETS[task].horizon,
            "reference_policy": "flappy_oracle (MPC)" if task == "flappy" else "v1.2 heuristic",
        }
    plan = {
        "version": "1.3",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_commit_at_freeze": commit,
        "uncommitted_code_changes_at_freeze": bool(status.strip()),
        "code_sha256": code_hash(),
        "code_files": {str(p): digest(p) for p in code_files()},
        "environment_source_sha256": digest("src/flyarcade/v12/environments.py"),
        "source_graph_sha256": digest("data/malecns-v1.0/graph.npz"),
        "graph_sha256": graphs,
        "standardizer_sha256": standardizers,
        "tasks": tasks,
        "conditions": {
            "biological": "authentic MaleCNS core, selected learner and budget",
            "frozen": "identical initialisation, no learning (zero-transition budget; before == after)",
            "rewired": "degree-preserving rewired core k (seed 700+k) for training seed k, own standardizer, same learner/budget",
            "sensory": "engineered observation into the same learner family, no SNN, same budget",
            "random": "uniform random actions (matched evaluation seeds, RNG seed + 2e9)",
            "reference": "heuristic/oracle (matched evaluation seeds)",
        },
        "training_seeds": SEEDS,
        "evaluation_episodes": EVAL_EPISODES,
        "confirmatory_purposes": list(CONFIRMATORY_PURPOSES),
        "perturbations": {
            "applied_to": "biological learned policies, no retraining",
            "sensory_noise_sd": 0.1,
            "edge_ablation_fraction": 0.1,
            "neuron_ablation_fraction": 0.1,
            "masks": "Bernoulli, seeded by conf_perturb seed per task/training seed",
            "recurrent_state": "GRU hidden state and neural core reset at every episode start",
        },
        "success_criteria": [
            "biological mean after > biological mean before",
            f"biological mean after > random mean + {MARGIN}",
            f">= 4/5 biological seeds > matched random + {MARGIN}",
            f"mean random-to-reference gap closure >= {GAP_CLOSURE}",
            f"mean fixed-history state dependence > {PROBE}",
            f"not constant-action (greedy top-action fraction < {CONSTANT_ACTION} every seed); final entropy > {ENTROPY_FLOOR}",
            "all numerical values finite",
        ],
        "analysis": "per-seed values, means, seed SD and descriptive bootstrap intervals; no p-values",
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "blas": "Accelerate (single-thread env vars set by v13_suite.py)",
        },
        "invalidation_rule": "any numerical-result-affecting change after this timestamp invalidates confirmatory results; further tuning requires a new version",
    }
    PLAN.write_text(json.dumps(plan, indent=1) + "\n")
    sha = digest(PLAN)
    SPECS.mkdir(parents=True, exist_ok=True)
    count = 0
    for task, entry in tasks.items():
        for seed in SEEDS:
            base_eval = {"purpose": "conf_eval", "episodes": EVAL_EPISODES, "probe_purpose": "conf_probe"}
            trials = {
                "biological": ({**entry["fly_config"], "topology": "biological"}, {**base_eval, "perturbations": True, "perturb_purpose": "conf_perturb"}),
                "frozen": ({**entry["fly_config"], "topology": "biological", "learn": False, "transitions": 0}, base_eval),
                "rewired": ({**entry["fly_config"], "topology": f"rewired-{seed}"}, base_eval),
                "sensory": (entry["sensory_config"], base_eval),
            }
            for condition, (config, evaluation) in trials.items():
                name = f"v13-{task}-{condition}-{seed}"
                spec = {
                    "name": name,
                    "config": {**config, "task": task, "seed": seed},
                    "evaluation": evaluation,
                    "output": f"runs/{name}",
                    "plan": {"path": str(PLAN), "sha256": sha},
                }
                (SPECS / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
                count += 1
    print(f"{PLAN}: frozen sha256={sha}; {count} confirmatory specs")
    print(hashlib.sha256(PLAN.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
