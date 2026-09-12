"""Replay one saved trial's held-out evaluations from its checkpoint alone.

Each trial runs in its own process so the unchanged 120 s runtime guard applies
per trial exactly as during training. Only the final boundary checkpoint is used;
the pre-training evaluation cannot be replayed from it and is not claimed here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import evaluate, restore
from flyarcade.resources import ResourceGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    directory = Path("runs") / args.name
    result = json.loads((directory / "result.json").read_text())
    config = result["config"]
    guard = ResourceGuard(directory=directory)
    graph = load_graph("data/malecns-v1.0/graph.npz")
    if config["condition"] == "rewired":
        graph = degree_preserving_control(graph, seed=700 + config["seed"], swaps_per_edge=10)
    digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    if digest != result["graph_sha256"]:
        raise ValueError("regenerated graph digest differs from the recorded trial graph")
    controller = Controller(
        graph, config["seed"], recurrent_learning=config["condition"] in ("learning", "rewired")
    )
    restore(controller, directory / "checkpoint.npz", digest)
    seeds = result["evaluation_seeds"]
    perturb_rng = np.random.default_rng(800 + config["seed"])
    edge_mask = perturb_rng.random(graph.e) >= 0.1
    neuron_mask = perturb_rng.random(graph.n) >= 0.1
    replayed = {
        "after": evaluate(controller, config["task"], seeds, guard=guard),
        "sensory_noise_0.1": evaluate(controller, config["task"], seeds, noise=0.1, guard=guard),
        "edge_ablation_0.1": evaluate(
            controller, config["task"], seeds, edge_mask=edge_mask, guard=guard
        ),
        "neuron_ablation_0.1": evaluate(
            controller, config["task"], seeds, neuron_mask=neuron_mask, guard=guard
        ),
    }
    mismatches = [key for key, value in replayed.items() if value != result[key]]
    print(
        json.dumps(
            {
                "trial": args.name,
                "graph_sha256_match": True,
                "evaluations_replayed": sorted(replayed),
                "mismatches": mismatches,
                "status": "MATCH" if not mismatches else "MISMATCH",
                "resources": guard.check(),
            }
        ),
        flush=True,
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
