"""Bounded v1.1 development search on development seeds only.

Never touches a v1 held-out seed or a v1.1 confirmatory seed. A full grid over the
seven requested axes is unaffordable on CPU, so the search is coordinate-wise: start
from a documented baseline, sweep one axis at a time across several development
seeds, and keep a change only if it improves consistently. Every configuration is
recorded, including rejected ones, so the search is auditable rather than a report
of whatever happened to win.

Rejection criteria are applied before any performance comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.v11.develop_support import rejections, run_one
from flyarcade.v11.features import Standardizer

BASELINE = {
    "actor_lr": 0.2,
    "critic_lr": 0.5,
    "discount": 0.9,
    "trace_decay": 0.7,
    "entropy_coefficient": 0.01,
    "exploration_floor": 0.05,
}
AXES = {
    "actor_lr": [0.1, 0.2, 0.4, 0.8],
    "critic_lr": [0.2, 0.5, 1.0],
    "discount": [0.85, 0.9, 0.95],
    "trace_decay": [0.5, 0.7, 0.9],
    "entropy_coefficient": [0.003, 0.01, 0.03],
    "exploration_floor": [0.02, 0.05, 0.1],
    "core_lr": [0.0, 0.02, 0.1],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=sorted(AXES), required=True)
    parser.add_argument("--task", choices=["catch", "dodge"], default="catch")
    parser.add_argument(
        "--stage", choices=["readout", "descending", "recurrent"], default="readout"
    )
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--base", type=str, default=None, help="JSON overriding BASELINE")
    args = parser.parse_args()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    standardizer = Standardizer.load("artifacts/v11_standardizer.json")
    base = dict(BASELINE)
    if args.base:
        base.update(json.loads(Path(args.base).read_text()))
    directory = Path("artifacts/v11_development")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for value in AXES[args.axis]:
        config = dict(base)
        config[args.axis] = value
        if args.stage == "readout":
            config.pop("core_lr", None)
        rows = [
            run_one(graph, standardizer, args.task, s, config, args.episodes, args.stage)
            for s in args.dev_seeds
        ]
        reasons = rejections(rows)
        record = {
            "axis": args.axis,
            "value": value,
            "config": config,
            "rejected": bool(reasons),
            "rejection_reasons": reasons,
            "mean_after": float(np.mean([r["after"] for r in rows])),
            "min_after": float(np.min([r["after"] for r in rows])),
            "mean_over_random": float(np.mean([r["over_random"] for r in rows])),
            "seeds_above_random": int(sum(r["over_random"] > 0.02 for r in rows)),
            "mean_improvement": float(np.mean([r["improvement"] for r in rows])),
            "trials": rows,
        }
        records.append(record)
        print(
            f"{args.axis}={value}: after={record['mean_after']:.3f} "
            f"(min {record['min_after']:.3f}) over_random={record['mean_over_random']:+.3f} "
            f"seeds_above={record['seeds_above_random']}/{len(rows)} "
            f"ent={np.mean([r['final_entropy'] for r in rows]):.2f} "
            f"{'REJECTED ' + '; '.join(reasons) if reasons else ''}",
            flush=True,
        )
    name = f"{args.stage}_{args.task}_{args.axis}.json"
    (directory / name).write_text(
        json.dumps(
            {
                "axis": args.axis,
                "task": args.task,
                "stage": args.stage,
                "episodes": args.episodes,
                "dev_seeds": args.dev_seeds,
                "baseline": base,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote artifacts/v11_development/{name}")


if __name__ == "__main__":
    main()
