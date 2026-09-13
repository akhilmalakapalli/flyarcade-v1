"""Replay a saved biological checkpoint without learning or overwriting results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import build, code_hash, evaluate, restore_checkpoint, seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    guard = ResourceGuard()
    planpath = Path("experiments/v12_run_plan.json")
    plan = json.loads(planpath.read_text())
    assert code_hash() == plan["code_sha256"]
    path = Path("runs") / f"v12-{args.task}-biological-{args.seed}" / "result.json"
    result = json.loads(path.read_text())
    graph = load_graph("data/malecns-v1.0/graph.npz")
    controller = build(
        graph,
        args.task,
        args.seed,
        Standardizer.load(f"artifacts/v12/{args.task}-biological-standardizer.json"),
        plan["tasks"][args.task]["hyperparameters"],
    )
    restore_checkpoint(
        path.with_name("checkpoint.npz"),
        controller,
        {
            "plan_sha256": hashlib.sha256(planpath.read_bytes()).hexdigest(),
            "code_sha256": code_hash(),
        },
    )
    rng = np.random.default_rng(seed_for(args.task, "perturb", args.seed))
    masks = {"edge_mask": rng.random(graph.e) >= 0.1, "neuron_mask": rng.random(graph.n) >= 0.1}
    for name, kwargs in [
        ("after", {}),
        ("sensory_noise", {"noise": 0.1}),
        ("edge_ablation", {"edge_mask": masks["edge_mask"]}),
        ("neuron_ablation", {"neuron_mask": masks["neuron_mask"]}),
    ]:
        rows = evaluate(controller, args.task, result["evaluation_seeds"], guard=guard, **kwargs)
        assert rows == result["evaluations"][name], name
        print(args.task, args.seed, name, "MATCH", np.mean([r["success"] for r in rows]))


if __name__ == "__main__":
    main()
