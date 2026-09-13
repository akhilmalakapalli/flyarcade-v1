"""Replay all historical Catch/Dodge checkpoint evaluations without file mutation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from v11_run_trial import greedy_rows

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.features import Standardizer


def main():
    guard = ResourceGuard()
    rows = []
    for task in ("catch", "dodge"):
        for condition in ("eprop", "frozen", "rewired"):
            for seed in range(3):
                directory = Path("runs") / f"v11-{task}-{condition}-{seed}"
                path = directory / "result.json"
                original = path.read_bytes()
                result = json.loads(original)
                graph = load_graph(
                    f"data/v12/rewired-{seed}.npz"
                    if condition == "rewired"
                    else "data/malecns-v1.0/graph.npz"
                )
                with np.load(directory / "checkpoint.npz", allow_pickle=False) as checkpoint:
                    std = Standardizer(
                        checkpoint["standardizer_mean"], checkpoint["standardizer_scale"]
                    )
                    config = result["config"]
                    controller = V11Controller(
                        graph,
                        seed,
                        stage=config["stage"],
                        standardizer=std,
                        **config["hyperparameters"],
                    )
                    controller.motor.actor = checkpoint["actor"].copy()
                    controller.motor.critic = checkpoint["critic"].copy()
                    controller.core.magnitude = checkpoint["magnitude"].copy()
                assert (
                    greedy_rows(controller, task, result["evaluation_seeds"], guard=guard)
                    == result["after"]
                )
                assert path.read_bytes() == original
                rows.append(
                    {
                        "trial": directory.name,
                        "status": "exact match",
                        "result_sha256": hashlib.sha256(original).hexdigest(),
                    }
                )
                print(directory.name, "MATCH", flush=True)
    Path("artifacts/v12/historical_replay.json").write_text(
        json.dumps({"status": "PASS", "trials": rows, "resources": guard.check()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
