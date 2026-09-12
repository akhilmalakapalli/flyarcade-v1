"""Check trial invariants and exactly replay two saved held-out evaluations."""

import hashlib
import json
from pathlib import Path

from flyarcade.connectome import load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import evaluate, restore
from flyarcade.resources import ResourceGuard


def main():
    guard = ResourceGuard()
    summary = json.loads(Path("artifacts/results_summary.json").read_text())
    checked = 0
    for name, digest in summary["result_files"].items():
        path = Path("runs") / name / "result.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("result file hash mismatch")
        result = json.loads(path.read_text())
        if result["completed_episodes"] != result["config"]["episodes"]:
            raise ValueError("incomplete training history")
        if len(result["training"]) != result["completed_episodes"]:
            raise ValueError("history length mismatch")
        condition = result["config"]["condition"]
        if condition == "frozen":
            if result["before"] != result["after"]:
                raise ValueError("frozen evaluation drift")
            if result["recurrent_l1_change"] or result["readout_l1_change"]:
                raise ValueError("frozen weights changed")
        if condition == "readout_only" and result["recurrent_l1_change"]:
            raise ValueError("readout-only recurrent weights changed")
        train_seeds = {
            1000 + result["config"]["seed"] * 10000 + ep
            for ep in range(result["config"]["episodes"])
        }
        if train_seeds.intersection(result["evaluation_seeds"]):
            raise ValueError("training/evaluation seed overlap")
        checked += 1
    graph = load_graph("data/malecns-v1.0/graph.npz")
    digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    replay = []
    for task in ("catch", "dodge"):
        directory = Path("runs") / f"{task}-learning-0-200"
        result = json.loads((directory / "result.json").read_text())
        controller = Controller(graph, seed=0)
        restore(controller, directory / "checkpoint.npz", digest)
        actual = evaluate(controller, task, result["evaluation_seeds"], guard=guard)
        if actual != result["after"]:
            raise ValueError("held-out checkpoint replay differed")
        replay.append(str(directory))
    report = {
        "status": "PASS",
        "trial_hashes_and_invariants_checked": checked,
        "exact_checkpoint_replays": replay,
        "resources": guard.check(),
        "limitations": "Verifies implementation consistency, not physiological validity.",
    }
    Path("artifacts/scientific_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
