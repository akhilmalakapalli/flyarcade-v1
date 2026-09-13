"""A new-version trial, resumable at episode/evaluation boundaries under unchanged guards."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import (
    CONDITIONS,
    array_hash,
    build,
    code_hash,
    episode,
    evaluate,
    graph_hash,
    restore_checkpoint,
    save_checkpoint,
    seed_for,
    state_probe,
)

RESUME_EXIT = 7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    path = Path("experiments/v12_run_plan.json")
    plan = json.loads(path.read_text())
    if plan["code_sha256"] != code_hash():
        raise ValueError(
            "code changed after freeze: version/invalidate consistently before proceeding"
        )
    directory = Path("runs") / f"v12-{args.task}-{args.condition}-{args.seed}"
    directory.mkdir(exist_ok=True)
    result_path = directory / "result.json"
    expected = {
        "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": plan["code_sha256"],
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
    }
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("completed result identity mismatch; file preserved")
        print(f"{directory.name}: completed result preserved", flush=True)
        return 0
    guard = ResourceGuard(directory=directory)
    graph = load_graph(
        f"data/v12/rewired-{args.seed}.npz"
        if args.condition == "rewired"
        else "data/malecns-v1.0/graph.npz"
    )
    topology = f"rewired-{args.seed}" if args.condition == "rewired" else "biological"
    stdpath = Path(f"artifacts/v12/{args.task}-{topology}-standardizer.json")
    stdhash = hashlib.sha256(stdpath.read_bytes()).hexdigest()
    if plan["standardizer_hashes"][str(stdpath)] != stdhash:
        raise ValueError("feature standardizer changed after freeze")
    expected.update(graph_sha256=graph_hash(graph), standardizer_sha256=stdhash)
    controller = build(
        graph,
        args.task,
        args.seed,
        Standardizer.load(stdpath),
        plan["tasks"][args.task]["hyperparameters"],
    )
    core_hash = array_hash(controller.core.magnitude)
    initial_actor = array_hash(controller.motor.actor)
    initial_critic = array_hash(controller.motor.critic)
    checkpoint = directory / "checkpoint.npz"
    seeds = [seed_for(args.task, "eval", args.seed, i) for i in range(plan["evaluation_episodes"])]
    episodes = plan["tasks"][args.task]["training_episodes"]
    if checkpoint.exists():
        state = restore_checkpoint(checkpoint, controller, expected)
    else:
        state = {
            **expected,
            "completed_episodes": 0,
            "training": [],
            "evaluations": {},
            "segments": [],
            "evaluation_seeds": seeds,
            "training_seeds": [seed_for(args.task, "train", args.seed, i) for i in range(episodes)],
            "source_graph_sha256": plan["source_graph_sha256"],
            "initial_core_sha256": core_hash,
            "initial_actor_sha256": initial_actor,
            "initial_critic_sha256": initial_critic,
        }
        state["before"] = evaluate(controller, args.task, seeds, guard=guard)
        save_checkpoint(checkpoint, controller, state)

    def pause_if_needed():
        if guard.snapshot()["elapsed_seconds"] > 70:
            state["segments"].append(guard.check())
            save_checkpoint(checkpoint, controller, state)
            print(
                f"{directory.name}: checkpoint {state['completed_episodes']}/{episodes}; resume",
                flush=True,
            )
            return True
        return False

    while state["completed_episodes"] < episodes:
        ep = state["completed_episodes"]
        row = episode(
            controller,
            args.task,
            seed_for(args.task, "train", args.seed, ep),
            training=args.condition != "frozen",
            guard=guard,
        )
        state["training"].append(row)
        state["completed_episodes"] += 1
        if state["completed_episodes"] % 25 == 0:
            save_checkpoint(checkpoint, controller, state)
        if pause_if_needed():
            return RESUME_EXIT
    save_checkpoint(checkpoint, controller, state)
    perturb = np.random.default_rng(seed_for(args.task, "perturb", args.seed))
    edge_mask = perturb.random(graph.e) >= 0.1
    neuron_mask = perturb.random(graph.n) >= 0.1
    evaluations = {
        "after": {},
        "random": {"baseline": "random"},
        "heuristic": {"baseline": "heuristic"},
    }
    if args.condition == "biological":
        evaluations.update(
            sensory_noise={"noise": 0.1},
            edge_ablation={"edge_mask": edge_mask},
            neuron_ablation={"neuron_mask": neuron_mask},
        )
    for key, kwargs in evaluations.items():
        if key in state["evaluations"]:
            continue
        if "baseline" in kwargs:
            rows = [episode(None, args.task, s, guard=guard, **kwargs) for s in seeds]
        else:
            rows = evaluate(controller, args.task, seeds, guard=guard, **kwargs)
        state["evaluations"][key] = rows
        save_checkpoint(checkpoint, controller, state)
        if pause_if_needed():
            return RESUME_EXIT
    state["state_probe"] = state_probe(controller, args.task, args.seed, guard=guard)
    if array_hash(controller.core.magnitude) != core_hash:
        raise ValueError("Stage A core changed")
    if args.condition == "frozen" and (
        array_hash(controller.motor.actor) != initial_actor
        or array_hash(controller.motor.critic) != initial_critic
    ):
        raise ValueError("frozen readout changed")
    state["segments"].append(guard.check())
    save_checkpoint(checkpoint, controller, state)
    result = {
        **state,
        "final_core_sha256": array_hash(controller.core.magnitude),
        "final_actor_sha256": array_hash(controller.motor.actor),
        "final_critic_sha256": array_hash(controller.motor.critic),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "perturbation_mask_sha256": {
            "edge": array_hash(edge_mask),
            "neuron": array_hash(neuron_mask),
        },
        "hyperparameters": plan["tasks"][args.task]["hyperparameters"],
        "evaluation": "greedy, no learning",
        "status": "COMPLETE",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trial": directory.name,
                "before": np.mean([r["success"] for r in state["before"]]),
                "after": np.mean([r["success"] for r in state["evaluations"]["after"]]),
                "segments": len(state["segments"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
