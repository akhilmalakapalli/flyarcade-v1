"""Development-only tuning measurement; never uses held-out test seeds."""

import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import episode, evaluate
from flyarcade.resources import ResourceGuard


def main():
    guard = ResourceGuard()
    controller = Controller(load_graph("data/malecns-v1.0/graph.npz"), seed=99)
    seeds = list(range(2_000_000, 2_000_020))
    reports = []
    before = evaluate(controller, "catch", seeds, guard=guard)
    for ep in range(200):
        episode(controller, "catch", 1_900_000 + ep, training=True, guard=guard)
        if (ep + 1) % 50 == 0:
            results = evaluate(controller, "catch", seeds, guard=guard)
            reports.append(
                {"episodes": ep + 1, "success": float(np.mean([r["success"] for r in results]))}
            )
            print(reports[-1], flush=True)
    Path("artifacts/development.json").write_text(
        json.dumps(
            {
                "split": "development-only",
                "seed": 99,
                "before": float(np.mean([r["success"] for r in before])),
                "learning_curve": reports,
                "resources": guard.check(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
