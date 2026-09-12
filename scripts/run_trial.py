"""One bounded, resumable, preregistered seeded experiment trial."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import checkpoint, episode, evaluate, restore
from flyarcade.resources import ResourceGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["catch", "dodge"], required=True)
    parser.add_argument(
        "--condition", choices=["learning", "frozen", "rewired", "readout_only"], default="learning"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--source", choices=["catch", "dodge"])
    args = parser.parse_args()
    if args.seed not in (0, 1, 2) or args.episodes not in (100, 200):
        raise SystemExit("Preregistered seeds 0,1,2 and budgets 100/200 only.")
    if args.source and (args.source == args.task or args.episodes != 200):
        raise SystemExit("Transfer requires different source and 200 total episodes.")
    name = f"{args.task}-{args.condition}-{args.seed}-{args.episodes}"
    if args.source:
        name += "-from-" + args.source
    directory = Path("runs") / name
    directory.mkdir(exist_ok=True)
    report_path = directory / "result.json"
    if report_path.exists():
        print(f"{name}: completed result preserved", flush=True)
        return
    guard = ResourceGuard(directory=directory)
    graph = load_graph("data/malecns-v1.0/graph.npz")
    source_digest = json.loads(Path("artifacts/malecns_manifest.json").read_text())["graph_sha256"]
    if args.condition == "rewired":
        graph = degree_preserving_control(graph, seed=700 + args.seed, swaps_per_edge=10)
    digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    controller = Controller(
        graph, args.seed, recurrent_learning=args.condition in ("learning", "rewired")
    )
    config = vars(args)
    code_hash = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        code_hash.update(str(path).encode() + path.read_bytes())
    model_code_sha256 = code_hash.hexdigest()
    state = {
        "graph_sha256": digest,
        "source_graph_sha256": source_digest,
        "config": config,
        "model_code_sha256": model_code_sha256,
        "completed_episodes": 0,
        "training": [],
    }
    checkpoint_path = directory / "checkpoint.npz"
    eval_seeds = list(range(3_000_000 + 100 * args.seed, 3_000_020 + 100 * args.seed))
    if checkpoint_path.exists():
        state = restore(controller, checkpoint_path, digest)
        if state["config"] != config or state["model_code_sha256"] != model_code_sha256:
            raise ValueError("checkpoint config/code mismatch")
    else:
        state["before"] = evaluate(controller, args.task, eval_seeds, guard=guard)
        checkpoint(controller, checkpoint_path, state)
    initial_recurrent = controller.core.base.copy()
    initial_motor = Controller(graph, args.seed).motor.weights.copy()
    try:
        for ep in range(state["completed_episodes"], args.episodes):
            task = args.source if args.source and ep < 100 else args.task
            row = episode(
                controller,
                task,
                1000 + args.seed * 10000 + ep,
                training=args.condition != "frozen",
                guard=guard,
            )
            state["training"].append({**row, "task": task, "episode": ep + 1})
            state["completed_episodes"] = ep + 1
            if (ep + 1) % 25 == 0:
                checkpoint(controller, checkpoint_path, state)
        after = evaluate(controller, args.task, eval_seeds, guard=guard)
        perturb_rng = np.random.default_rng(800 + args.seed)
        edge_mask = perturb_rng.random(graph.e) >= 0.1
        neuron_mask = perturb_rng.random(graph.n) >= 0.1
        noise = evaluate(controller, args.task, eval_seeds, noise=0.1, guard=guard)
        ablation = evaluate(controller, args.task, eval_seeds, edge_mask=edge_mask, guard=guard)
        neuron_ablation = evaluate(
            controller, args.task, eval_seeds, neuron_mask=neuron_mask, guard=guard
        )
        result = {
            **state,
            "after": after,
            "sensory_noise_0.1": noise,
            "edge_ablation_0.1": ablation,
            "neuron_ablation_0.1": neuron_ablation,
            "evaluation_seeds": eval_seeds,
            "random": [episode(None, args.task, s, baseline="random") for s in eval_seeds],
            "heuristic": [episode(None, args.task, s, baseline="heuristic") for s in eval_seeds],
            "recurrent_l1_change": float(
                np.abs(controller.core.magnitude - initial_recurrent).sum()
            ),
            "readout_l1_change": float(np.abs(controller.motor.weights - initial_motor).sum()),
            "spikes_per_neuron_tick_last_eval": controller.spike_count
            / (graph.n * controller.tick_count),
            "resources": guard.check(),
            "status": "COMPLETE",
            "control": {k: v for k, v in graph.provenance.items() if k in ("attempts", "accepted")},
            "training_seed_rule": "1000 + seed*10000 + episode_index",
            "evaluation_policy_seed_rule": "environment_seed + 500000",
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "trial": name,
                    "before": np.mean([r["success"] for r in state["before"]]),
                    "after": np.mean([r["success"] for r in after]),
                    "seconds": result["resources"]["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    except Exception:
        # Preserve the last complete 25-episode boundary. Partial episode weights
        # and RNG must never overwrite a resumable boundary checkpoint.
        print(f"{name}: stopped; last complete checkpoint preserved", flush=True)
        raise


if __name__ == "__main__":
    main()
