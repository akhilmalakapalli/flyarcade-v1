"""Fit the frozen descending-rate standardisation on v1.1 development seeds only.

Runs the frozen core under a uniform-random behaviour policy so the statistics do
not depend on any policy that will later be trained, and never touches a v1 held-out
seed or a v1.1 confirmatory seed.
"""

import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.experiment import FEATURE_FIT_SEEDS
from flyarcade.v11.features import fit_for_graph


def main():
    guard = ResourceGuard()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    standardizer = fit_for_graph(
        graph,
        lambda g: V11Controller(g, 0, stage="readout"),
        FEATURE_FIT_SEEDS,
        guard=guard,
    )

    standardizer.save("artifacts/v11_standardizer.json")
    report = {
        "width": standardizer.width,
        "seeds": "FEATURE_FIT_SEEDS (4,200,000 block), uniform-random behaviour policy",
        "mean_abs": float(np.abs(standardizer.mean).mean()),
        "scale_mean": float(standardizer.scale.mean()),
        "scale_floor_hits": int((standardizer.scale <= 0.02).sum()),
        "gain": standardizer.gain,
        "resources": guard.check(),
    }
    Path("artifacts/v11_standardizer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
