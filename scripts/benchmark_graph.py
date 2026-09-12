"""Bounded synthetic graph benchmark. Does not measure learning or biology."""

import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph, save_graph, synthetic_graph
from flyarcade.resources import ResourceGuard


def main():
    guard = ResourceGuard()
    timings = {}
    start = time.perf_counter()
    graph = synthetic_graph(n=1024, edges=16384, seed=42)
    timings["generate_seconds"] = time.perf_counter() - start
    guard.check()
    start = time.perf_counter()
    control = degree_preserving_control(graph, seed=43, swaps_per_edge=2)
    timings["rewire_seconds"] = time.perf_counter() - start
    guard.check()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "graph.npz"
        start = time.perf_counter()
        save_graph(control, path)
        reloaded = load_graph(path)
        timings["save_load_seconds"] = time.perf_counter() - start
        np.testing.assert_array_equal(control.post, reloaded.post)
        archive_bytes = path.stat().st_size
    report = {
        "kind": "synthetic-software-benchmark",
        "seed": 42,
        "control_seed": 43,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "graph": graph.summary(),
        "accepted_swaps": control.provenance["accepted"],
        "attempted_swaps": control.provenance["attempts"],
        "archive_bytes": archive_bytes,
        "timings": timings,
        "resources": guard.check(),
    }
    Path("artifacts/graph_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
