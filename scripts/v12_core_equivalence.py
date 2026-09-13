"""Check the v1.2 fixed sparse executor against historical LIF on authentic graphs."""

import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.core import EpropCore
from flyarcade.v12.frozen_core import FrozenCore
from flyarcade.v12.study import build, graph_hash


def main():
    guard = ResourceGuard()
    rows = []
    for path in ("data/malecns-v1.0/graph.npz", "data/v12/rewired-0.npz"):
        graph = load_graph(path)
        signs = build(graph, "pong").core.signs
        original = EpropCore(graph, signs)
        optimized = FrozenCore(EpropCore(graph, signs))
        rng = np.random.default_rng(983)
        edge_mask = rng.random(graph.e) >= 0.1
        neuron_mask = rng.random(graph.n) >= 0.1
        for tick in range(300):
            options = (
                {}
                if tick < 100
                else {"edge_mask": edge_mask}
                if tick < 200
                else {"neuron_mask": neuron_mask}
            )
            drive = rng.uniform(0, 1.5, graph.n)
            assert np.array_equal(original.tick(drive, **options), optimized.tick(drive, **options))
            assert np.array_equal(original.voltage, optimized.voltage)
            guard.check()
        rows.append(
            {
                "path": path,
                "graph_sha256": graph_hash(graph),
                "ticks": 300,
                "spikes_and_voltages": "bit-identical",
                "phases": ["intact", "edge10%", "neuron10%"],
            }
        )
    Path("artifacts/v12/core_equivalence.json").write_text(
        json.dumps({"status": "PASS", "graphs": rows, "resources": guard.check()}, indent=2) + "\n"
    )
    print("Authentic biological and rewired core equivalence PASS")


if __name__ == "__main__":
    main()
