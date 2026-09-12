"""One resumable, independently guarded Snake trial.

Snake episodes lengthen as the policy improves, so a full training budget does not
fit in one 120-second guarded process. Rather than raise the guard, this follows the
v1 pattern: train until the guard is close, checkpoint at an episode boundary, and
exit asking to be resumed. Each process is independently guarded; the suite
re-invokes until the trial completes. A partial episode never overwrites a
checkpoint.
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
from flyarcade.v11.features import Standardizer
from flyarcade.v11.snake import CHANNELS
from flyarcade.v11.snake import features as snake_features
from flyarcade.v11.snake_experiment import (
    aggregate,
    confirm_evaluation_seeds,
    confirm_training_seed,
    dev_evaluation_seeds,
    dev_training_seed,
    episode,
    episodes_to_threshold,
    evaluate,
    learning_curve_auc,
)

CONDITIONS = ("eprop", "frozen", "rewired")
RESUME_EXIT = 7
SEGMENT_SECONDS = 85.0
CHECKPOINT_EVERY = 50


def build(graph, seed, standardizer, config):
    return V11Controller(
        graph,
        seed,
        stage="readout",
        standardizer=standardizer,
        encoder=snake_features,
        channels=CHANNELS,
        observation_width=CHANNELS,
        actions=4,
        **config,
    )


def save(path, controller, state):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        actor=controller.motor.actor,
        critic=controller.motor.critic,
        magnitude=controller.core.magnitude,
        metadata=np.frombuffer(json.dumps(state).encode(), dtype=np.uint8),
    )
    tmp.replace(path)


def restore(path, controller):
    with np.load(path, allow_pickle=False) as data:
        controller.motor.actor = data["actor"].copy()
        controller.motor.critic = data["critic"].copy()
        controller.core.magnitude = data["magnitude"].copy()
        return json.loads(data["metadata"].tobytes().decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["dev", "confirm"], default="confirm")
    parser.add_argument("--plan", default="experiments/snake_run_plan.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    episodes = plan["training_episodes"]
    prefix = "snake" if args.mode == "confirm" else "snake-dev"
    name = f"{prefix}-{args.condition}-{args.seed}"
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
    # Each topology is standardised against its own rates by the same frozen
    # procedure; see scripts/snake_fit_features.py.
    standardizer_path = (
        plan["standardizer_rewired"].format(seed=args.seed)
        if args.condition == "rewired"
        else plan["standardizer"]
    )
    standardizer = Standardizer.load(standardizer_path)
    config = dict(plan["hyperparameters"])
    controller = build(graph, args.seed, standardizer, config)
    code_hash = hashlib.sha256()
    for path in sorted(Path("src/flyarcade/v11").rglob("*.py")):
        code_hash.update(str(path).encode() + path.read_bytes())
    if args.mode == "confirm":
        seeds = confirm_evaluation_seeds(args.seed, plan["evaluation_episodes"])
        train_seed = confirm_training_seed
    else:
        seeds = dev_evaluation_seeds(args.seed, plan["evaluation_episodes"])
        train_seed = dev_training_seed

    checkpoint_path = directory / "checkpoint.npz"
    if checkpoint_path.exists():
        state = restore(checkpoint_path, controller)
        if state["config"] != config or state["graph_sha256"] != digest:
            raise ValueError("checkpoint config/graph mismatch")
    else:
        state = {
            "config": config,
            "graph_sha256": digest,
            "completed_episodes": 0,
            "training": [],
            "before": evaluate(controller, seeds, guard=guard),
        }
        save(checkpoint_path, controller, state)

    if args.condition != "frozen":
        while state["completed_episodes"] < episodes:
            ep = state["completed_episodes"]
            state["training"].append(episode(controller, train_seed(args.seed, ep), training=True))
            state["completed_episodes"] = ep + 1
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                save(checkpoint_path, controller, state)
                if guard.snapshot()["elapsed_seconds"] > SEGMENT_SECONDS:
                    print(
                        f"{name}: {state['completed_episodes']}/{episodes} episodes; "
                        "checkpointed, resume required",
                        flush=True,
                    )
                    raise SystemExit(RESUME_EXIT)
            guard.check()
        save(checkpoint_path, controller, state)

    after = evaluate(controller, seeds, guard=guard)
    # Perturbations use the frozen learned weights and fixed masks sampled once.
    perturb_rng = np.random.default_rng(800 + args.seed)
    perturbations = {
        "sensory_noise_0.1": {"noise": 0.1},
        "edge_ablation_0.1": {"edge_mask": perturb_rng.random(graph.e) >= 0.1},
    }
    for fraction in (0.05, 0.10, 0.25):
        perturbations[f"neuron_ablation_{fraction:.2f}"] = {
            "neuron_mask": perturb_rng.random(graph.n) >= fraction
        }
    perturbed = {
        key: evaluate(controller, seeds, guard=guard, **kwargs)
        for key, kwargs in perturbations.items()
    }
    curve = state["training"]
    result = {
        "graph_sha256": digest,
        "source_graph_sha256": source_digest,
        "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
        "v11_code_sha256": code_hash.hexdigest(),
        "standardizer_path": standardizer_path,
        "standardizer_sha256": hashlib.sha256(
            standardizer.mean.tobytes() + standardizer.scale.tobytes()
        ).hexdigest(),
        "config": {
            "task": "snake",
            "condition": args.condition,
            "seed": args.seed,
            "mode": args.mode,
            "episodes": len(curve),
            "hyperparameters": config,
        },
        "completed_episodes": len(curve),
        "training": curve,
        "before": state["before"],
        "after": after,
        **perturbed,
        "evaluation_seeds": seeds,
        "random": [episode(None, s, baseline="random") for s in seeds],
        "heuristic": [episode(None, s, baseline="heuristic") for s in seeds],
        "metrics": {
            "before": aggregate(state["before"]),
            "after": aggregate(after),
            "perturbations": {key: aggregate(rows) for key, rows in perturbed.items()},
            "learning_curve_auc_food": learning_curve_auc(curve, "food"),
            "learning_curve_auc_steps": learning_curve_auc(curve, "steps"),
            "episodes_to_one_food": episodes_to_threshold(curve, 1.0, "food"),
            "episodes_to_two_food": episodes_to_threshold(curve, 2.0, "food"),
            "final_entropy": float(np.mean([r["entropy"] for r in curve[-200:]]))
            if curve
            else None,
            "final_firing_rate": float(np.mean([r["firing_rate"] for r in curve[-200:]]))
            if curve
            else None,
            "final_abs_delta": float(np.mean([r["abs_delta"] for r in curve[-200:]]))
            if curve
            else None,
            "final_actor_update_abs": float(np.mean([r["actor_update_abs"] for r in curve[-200:]]))
            if curve
            else None,
        },
        "evaluation": "greedy policy mode, learning disabled, seeds never used for tuning",
        "resources": {**guard.check(), "wall_seconds": time.monotonic() - started},
        "status": "COMPLETE",
        "control": {k: v for k, v in graph.provenance.items() if k in ("attempts", "accepted")},
        "training_seed_rule": "8,000,000 + 10,000*seed + episode (dev: 7,000,000 block)",
        "evaluation_seed_rule": "9,000,000 + 100*seed + i (dev: 7,500,000 block)",
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trial": name,
                "before_food": result["metrics"]["before"]["food"],
                "after_food": result["metrics"]["after"]["food"],
                "after_steps": result["metrics"]["after"]["steps"],
                "seconds": result["resources"]["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
