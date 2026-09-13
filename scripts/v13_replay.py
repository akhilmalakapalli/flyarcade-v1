"""Replay a completed v1.3 trial's evaluations from its saved policy (no training).

    v13_replay.py --run runs/v13-pong-biological-0 [--perturbations]

Restores final parameters from policy.npz, verifies their hash, rebuilds the feature
source and re-runs clean (and perturbation) greedy evaluations on the recorded seeds,
requiring exact row equality. Also restores the full trainer checkpoint and verifies
the optimiser state is finite and restorable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flyarcade_v13.study import (
    Trainer,
    build_model,
    evaluate_baseline,
    evaluate_policy,
    load_topology,
    make_source,
    params_hash,
    seed_for,
    state_probe,
)


def replay(run, perturbations=False, baselines=False):
    run = Path(run)
    result = json.loads((run / "result.json").read_text())
    config, ev, task = result["config"], result["evaluation"], result["task"]
    with np.load(run / "policy.npz", allow_pickle=False) as data:
        final = {k[6:]: data[k] for k in data.files if k.startswith("final_")}
        initial = {k[8:]: data[k] for k in data.files if k.startswith("initial_")}
    assert params_hash(final) == result["final_params_sha256"], "final params hash"
    assert params_hash(initial) == result["initial_params_sha256"], "initial params hash"
    assert all(np.isfinite(v).all() for v in final.values())
    source = make_source(config, ev.get("batch", 16))
    model = build_model(config, source.width)
    assert params_hash(model.params) == result["initial_params_sha256"], "initialisation replay"
    seeds = result["evaluation_seeds"]
    checks = {}
    checks["after"] = evaluate_policy(model, final, source, task, seeds) == result["evaluations"]["after"]
    checks["before"] = evaluate_policy(model, initial, source, task, seeds) == result["evaluations"]["before"]
    checks["state_probe"] = (
        state_probe(model, final, source, task, ev["probe_purpose"], config["seed"]) == result["state_probe"]
    )
    if baselines:
        for kind in ("random", "reference"):
            checks[kind] = evaluate_baseline(task, seeds, kind) == result["evaluations"][kind]
    if perturbations and ev.get("perturbations"):
        rng = np.random.default_rng(seed_for(task, ev["perturb_purpose"], config["seed"]))
        graph = load_topology(config["topology"])
        edge, neuron = rng.random(graph.e) >= 0.1, rng.random(graph.n) >= 0.1
        for key, kwargs in (
            ("sensory_noise", {"noise": 0.1}),
            ("edge_ablation", {"edge_mask": edge}),
            ("neuron_ablation", {"neuron_mask": neuron}),
        ):
            checks[key] = evaluate_policy(model, final, source, task, seeds, **kwargs) == result["evaluations"][key]
    trainer = Trainer(config)
    state = trainer.load(run / "checkpoint.pkl")
    checks["checkpoint_params_match_policy"] = params_hash(trainer.model.params) == result["final_params_sha256"]
    checks["optimizer_finite_restorable"] = trainer.optimizer.t == result["optimizer_steps"] and all(
        np.isfinite(v).all() for v in list(trainer.optimizer.m.values()) + list(trainer.optimizer.v.values())
    )
    checks["checkpoint_transitions"] = state["counters"]["transitions"] == result["transitions"]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    args = parser.parse_args()
    checks = replay(args.run, args.perturbations, args.baselines)
    print(json.dumps({"run": args.run, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
