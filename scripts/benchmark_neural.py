"""Small CPU benchmark on the authentic circuit, not a learning claim."""

import json
import time
from pathlib import Path

from flyarcade.connectome import load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import episode
from flyarcade.resources import ResourceGuard


def main():
    guard = ResourceGuard()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    controller = Controller(graph, seed=987)
    start = time.perf_counter()
    for seed in range(5):
        episode(controller, "catch", 4_000_000 + seed, training=True, guard=guard)
    elapsed = time.perf_counter() - start
    report = {
        "kind": "authentic-graph-software-throughput",
        "neurons": graph.n,
        "edges": graph.e,
        "actions": 120,
        "neural_ticks": 480,
        "elapsed_seconds": elapsed,
        "actions_per_second": 120 / elapsed,
        "last_episode_spikes_per_neuron_tick": controller.spike_count
        / (controller.tick_count * graph.n),
        "resources": guard.check(),
        "training_outcomes_used_for_model_selection": False,
    }
    Path("artifacts/neural_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
