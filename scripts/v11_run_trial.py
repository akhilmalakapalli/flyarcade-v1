"""One frozen v1.1 confirmatory trial on seeds never used for tuning.

Conditions mirror the v1.1 protocol: the biological circuit with e-prop
actor-critic learning, the same circuit with learning disabled, a degree-preserving
rewired circuit with the identical learning rule and budget, and the random and
heuristic reference policies.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.experiment import (
    FEATURE_FIT_SEEDS,
    confirm_evaluation_seeds,
    confirm_training_seed,
    episode,
)
from flyarcade.v11.features import Standardizer, fit_for_graph

CONDITIONS = ("eprop", "frozen", "rewired")


def greedy_rows(controller, task, seeds, guard=None, **kwargs):
    """Evaluate the learned policy's mode with all learning switched off."""
    from flyarcade.games import LaneGame

    rows = []
    old_rng = controller.rng
    try:
        for seed in seeds:
            if guard is not None:
                guard.check()
            game = LaneGame(task, seed)
            controller.reset()
            controller.rng = np.random.default_rng(seed + 500_000)
            actions, score = [], 0
            while not game.done:
                features = controller.rates(game.observe(), **kwargs)
                _, softmax, _ = controller.motor.policy(features)
                action = int(softmax.argmax())
                actions.append(action)
                _, _, _, info = game.step(action)
                score += info["score"]
            rows.append(
                {
                    "success": score / (game.horizon // 4),
                    "action_counts": np.bincount(actions, minlength=3).tolist(),
                }
            )
    finally:
        controller.rng = old_rng
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["catch", "dodge"], required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(Path("experiments/v11_run_plan.json").read_text())
    if args.seed not in plan["seeds"]:
        raise SystemExit(f"preregistered seeds are {plan['seeds']}")
    name = f"v11-{args.task}-{args.condition}-{args.seed}"
    directory = Path("runs") / name
    directory.mkdir(exist_ok=True)
    report_path = directory / "result.json"
    if report_path.exists():
        print(f"{name}: completed result preserved", flush=True)
        return
    guard = ResourceGuard(directory=directory)
    started = time.monotonic()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    source_digest = json.loads(Path("artifacts/malecns_manifest.json").read_text())["graph_sha256"]
    if args.condition == "rewired":
        graph = degree_preserving_control(graph, seed=700 + args.seed, swaps_per_edge=10)
    digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    # Standardise against this trial's own graph using the frozen procedure, so the
    # rewired topology control is calibrated exactly as the biological arm is rather
    # than inheriting the biological graph's statistics.
    config = dict(plan["hyperparameters"])
    standardizer = fit_for_graph(
        graph,
        lambda g: V11Controller(g, args.seed, stage=plan["stage"]),
        FEATURE_FIT_SEEDS,
        guard=guard,
    )
    if args.condition != "rewired":
        saved = Standardizer.load("artifacts/v11_standardizer.json")
        if not np.allclose(saved.mean, standardizer.mean) or not np.allclose(
            saved.scale, standardizer.scale
        ):
            raise SystemExit("biological standardiser drifted from the frozen artifact")
    controller = V11Controller(
        graph, args.seed, stage=plan["stage"], standardizer=standardizer, **config
    )
    code_hash = hashlib.sha256()
    for path in sorted(Path("src/flyarcade/v11").rglob("*.py")):
        code_hash.update(str(path).encode() + path.read_bytes())
    seeds = confirm_evaluation_seeds(args.seed, plan["evaluation_episodes"])
    before = greedy_rows(controller, args.task, seeds, guard=guard)
    curve = []
    if args.condition != "frozen":
        for ep in range(plan["training_episodes"]):
            curve.append(
                episode(
                    controller,
                    args.task,
                    confirm_training_seed(args.seed, ep),
                    training=True,
                    guard=guard,
                )
            )
    after = greedy_rows(controller, args.task, seeds, guard=guard)
    # Lesions are run only now that state-dependent learning exists; in v1 they were
    # applied to a collapsed policy and were therefore uninformative.
    perturb_rng = np.random.default_rng(800 + args.seed)
    edge_mask = perturb_rng.random(graph.e) >= 0.1
    neuron_mask = perturb_rng.random(graph.n) >= 0.1
    sensory_noise = greedy_rows(controller, args.task, seeds, guard=guard, noise=0.1)
    edge_ablation = greedy_rows(controller, args.task, seeds, guard=guard, edge_mask=edge_mask)
    neuron_ablation = greedy_rows(
        controller, args.task, seeds, guard=guard, neuron_mask=neuron_mask
    )
    np.savez_compressed(
        directory / "checkpoint.npz",
        actor=controller.motor.actor,
        critic=controller.motor.critic,
        magnitude=controller.core.magnitude,
        standardizer_mean=standardizer.mean,
        standardizer_scale=standardizer.scale,
    )
    result = {
        "graph_sha256": digest,
        "source_graph_sha256": source_digest,
        "plan_sha256": hashlib.sha256(
            Path("experiments/v11_run_plan.json").read_bytes()
        ).hexdigest(),
        "v11_code_sha256": code_hash.hexdigest(),
        "config": {
            "task": args.task,
            "condition": args.condition,
            "seed": args.seed,
            "stage": plan["stage"],
            "episodes": plan["training_episodes"] if args.condition != "frozen" else 0,
            "hyperparameters": config,
        },
        "completed_episodes": len(curve),
        "training": curve,
        "before": before,
        "after": after,
        "sensory_noise_0.1": sensory_noise,
        "edge_ablation_0.1": edge_ablation,
        "neuron_ablation_0.1": neuron_ablation,
        "evaluation_seeds": seeds,
        "random": [episode(None, args.task, s, baseline="random") for s in seeds],
        "heuristic": [episode(None, args.task, s, baseline="heuristic") for s in seeds],
        "evaluation": "greedy policy mode, learning disabled, seeds never used for tuning",
        "standardizer_sha256": hashlib.sha256(
            standardizer.mean.tobytes() + standardizer.scale.tobytes()
        ).hexdigest(),
        "resources": {**guard.check(), "wall_seconds": time.monotonic() - started},
        "status": "COMPLETE",
        "control": {k: v for k, v in graph.provenance.items() if k in ("attempts", "accepted")},
        "training_seed_rule": "5,000,000 + 10,000*seed + episode",
        "evaluation_seed_rule": "6,000,000 + 100*seed + i",
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trial": name,
                "before": float(np.mean([r["success"] for r in before])),
                "after": float(np.mean([r["success"] for r in after])),
                "seconds": result["resources"]["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
