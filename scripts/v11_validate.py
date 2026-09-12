"""Validate one combined v1.1 configuration on development seeds before freezing.

Coordinate-wise sweeps optimise each axis independently, so the combination has to
be checked as a whole, and on development seeds that the sweeps did not use. This
script still runs entirely on development seeds; confirmatory seeds stay untouched.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.develop_support import RANDOM_BASELINE, rejections, run_one
from flyarcade.v11.features import Standardizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--stage", default="readout")
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tasks", nargs="+", default=["catch", "dodge"])
    parser.add_argument("--out", default="artifacts/v11_development/validation.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    graph = load_graph("data/malecns-v1.0/graph.npz")
    standardizer = Standardizer.load("artifacts/v11_standardizer.json")
    report = {"config": config, "stage": args.stage, "episodes": args.episodes, "tasks": {}}
    for task in args.tasks:
        rows = [
            run_one(graph, standardizer, task, s, config, args.episodes, args.stage)
            for s in args.dev_seeds
        ]
        reasons = rejections(rows)
        report["tasks"][task] = {
            "rejected": bool(reasons),
            "rejection_reasons": reasons,
            "mean_after": float(np.mean([r["after"] for r in rows])),
            "min_after": float(np.min([r["after"] for r in rows])),
            "random_baseline": RANDOM_BASELINE[task],
            "mean_over_random": float(np.mean([r["over_random"] for r in rows])),
            "seeds_above_random": int(sum(r["over_random"] > 0.02 for r in rows)),
            "seeds": args.dev_seeds,
            "mean_state_dependence": float(np.mean([r["state_dependence"] for r in rows])),
            "mean_final_entropy": float(np.mean([r["final_entropy"] for r in rows])),
            "train_first_fifth": float(np.mean([r["train_first_fifth"] for r in rows])),
            "train_last_fifth": float(np.mean([r["train_last_fifth"] for r in rows])),
            "max_trial_seconds": float(np.max([r["seconds"] for r in rows])),
            "trials": rows,
        }
        block = report["tasks"][task]
        print(
            f"{task}: after={block['mean_after']:.3f} (min {block['min_after']:.3f}) "
            f"random={block['random_baseline']:.3f} over={block['mean_over_random']:+.3f} "
            f"above={block['seeds_above_random']}/{len(args.dev_seeds)} "
            f"ent={block['mean_final_entropy']:.2f} sd={block['mean_state_dependence']:.2f} "
            f"train {block['train_first_fifth']:.3f}->{block['train_last_fifth']:.3f}"
            + (" REJECTED " + "; ".join(block["rejection_reasons"]) if block["rejected"] else ""),
            flush=True,
        )
    report["resources"] = ResourceGuard().check()
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
