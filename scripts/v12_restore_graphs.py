"""Recreate ignored rewired caches while preserving frozen feature-fit artifacts."""

import hashlib
import json
from pathlib import Path

from flyarcade.connectome import degree_preserving_control, load_graph, save_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v12.study import graph_hash


def main():
    guard = ResourceGuard()
    source = Path("data/malecns-v1.0/graph.npz")
    plan = json.loads(Path("experiments/v12_run_plan.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == plan["source_graph_sha256"]
    biological = load_graph(source)
    root = Path("data/v12")
    root.mkdir(exist_ok=True)
    for seed in range(3):
        path = root / f"rewired-{seed}.npz"
        if path.exists():
            graph = load_graph(path)
        else:
            graph = degree_preserving_control(biological, seed=700 + seed, swaps_per_edge=10)
            save_graph(graph, path)
        for task in plan["tasks"]:
            meta = json.loads(
                Path(f"artifacts/v12/{task}-rewired-{seed}-standardizer.meta.json").read_text()
            )
            assert graph_hash(graph) == meta["graph_sha256"]
        guard.check()
        print(path, "verified")


if __name__ == "__main__":
    main()
