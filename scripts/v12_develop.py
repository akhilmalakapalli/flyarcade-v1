"""Two predeclared readout-only attempts; no confirmatory seed is accessed."""

import argparse
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import build, episode, evaluate, seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--candidate", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    output = Path(f"artifacts/v12/development-{args.task}-{args.candidate}.json")
    if output.exists():
        print(f"{output}: preserved")
        return
    plan = json.loads(Path("experiments/v12_development_plan.json").read_text())
    config = plan["candidates"][args.task][args.candidate]
    guard = ResourceGuard()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    std = Standardizer.load(f"artifacts/v12/{args.task}-biological-standardizer.json")
    controller = build(graph, args.task, standardizer=std, config=config)
    seeds = [seed_for(args.task, "dev_eval", index=i) for i in range(20)]
    before = evaluate(controller, args.task, seeds, guard=guard)
    curve = []
    for ep in range(plan["episodes"]):
        curve.append(
            episode(
                controller,
                args.task,
                seed_for(args.task, "dev_train", index=ep),
                training=True,
                guard=guard,
            )
        )
    after = evaluate(controller, args.task, seeds, guard=guard)
    report = {
        "task": args.task,
        "candidate": args.candidate,
        "hyperparameters": config,
        "split": "development",
        "before": before,
        "after": after,
        "training": curve,
        "mean_success": float(np.mean([r["success"] for r in after])),
        "resources": guard.check(),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("task", "candidate", "mean_success", "resources")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
