"""Fit task/topology-specific development standardizers without confirmatory data."""

import argparse
import json
from pathlib import Path

from flyarcade.connectome import degree_preserving_control, load_graph, save_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import fit_standardizer, graph_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    guard = ResourceGuard()
    root = Path("artifacts/v12")
    cache = Path("data/v12")
    cache.mkdir(exist_ok=True)
    biological = load_graph("data/malecns-v1.0/graph.npz")
    for name in ("biological", "rewired-0", "rewired-1", "rewired-2"):
        output = root / f"{args.task}-{name}-standardizer.json"
        if output.exists():
            print(f"{output}: preserved", flush=True)
            continue
        if name == "biological":
            graph = biological
        else:
            path = cache / f"{name}.npz"
            if path.exists():
                graph = load_graph(path)
            else:
                graph = degree_preserving_control(
                    biological, seed=700 + int(name[-1]), swaps_per_edge=10
                )
                save_graph(graph, path)
        standardizer, record = fit_standardizer(graph, args.task, guard=guard)
        standardizer.save(output)
        record.update(
            graph_sha256=graph_hash(graph), task=args.task, topology=name, resources=guard.check()
        )
        output.with_suffix(".meta.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{args.task}/{name}: fitted {record['samples']} development states", flush=True)


if __name__ == "__main__":
    main()
