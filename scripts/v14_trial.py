"""Bounded v1.4 development trial; never executes confirmatory or validation seeds."""

import argparse
import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np

from flyarcade.resources import ResourceGuard
from flyarcade_v14.imitation import warm_start
from flyarcade_v14.study import (
    Trainer,
    array_hash,
    code_hash,
    config_hash,
    digest,
    evaluate_baseline,
    evaluate_policy,
    graph_hash,
    load_topology,
    make_source,
    mean_success,
    params_hash,
    seed_for,
    standardizer_path,
    state_probe,
)


def peak_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == "darwin" else 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    path = Path(args.spec)
    spec = json.loads(path.read_text())
    plan = Path("experiments/v14/development_plan.json")
    protocol = json.loads(plan.read_text())
    assert spec["plan_sha256"] == digest(plan)
    assert spec["code_sha256"] == code_hash()
    assert spec["phase"] in ("primary", "ticks", "all_readout", "longer", "curriculum", "rescue")
    assert spec["config"]["task"] in protocol["tasks"]
    out = Path("runs/v14") / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    identity = {
        "name": spec["name"],
        "spec_sha256": digest(path),
        "plan_sha256": digest(plan),
        "code_sha256": code_hash(),
    }
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert all(result.get(k) == v for k, v in identity.items())
        print(spec["name"], "completed result preserved", flush=True)
        return 0
    guard = ResourceGuard(directory=out)
    trainer = Trainer(spec["config"])
    c = trainer.config
    task = trainer.task
    core_hash = array_hash(trainer.source.base_magnitude)
    stdpath = standardizer_path(task, "biological", c["ticks"], c["readout"])
    identity.update(
        graph_sha256=graph_hash(load_topology("biological")),
        standardizer_sha256=digest(stdpath),
        config_sha256=config_hash(c),
    )
    checkpoint = out / "checkpoint.pkl"
    if checkpoint.exists():
        stored = trainer.load(checkpoint)
        extra = stored["extra"]
        assert extra["identity"] == identity
    else:
        extra = {"identity": identity, "evaluations": {}, "segments": [], "imitation": None}

    def save():
        state = trainer.state()
        state["extra"] = extra
        temp = checkpoint.with_suffix(".tmp")
        with temp.open("wb") as stream:
            pickle.dump(state, stream, protocol=5)
        temp.replace(checkpoint)

    def pause():
        if guard.snapshot()["elapsed_seconds"] > 70:
            extra["segments"].append({**guard.check(), "peak_memory_mb": peak_mb()})
            save()
            print(spec["name"], trainer.transitions, "checkpoint/resume", flush=True)
            return True
        return False

    source = make_source(c, 16)
    eval_seeds = [seed_for(task, "dev_eval", c["seed"], i) for i in range(40)]
    curve_seeds = [seed_for(task, "curve", c["seed"], i) for i in range(16)]
    if "before" not in extra["evaluations"]:
        extra["evaluations"]["before"] = evaluate_policy(
            trainer.model, trainer.initial_params, source, task, eval_seeds
        )
        save()
    if spec["phase"] == "rescue" and extra["imitation"] is None:
        extra["imitation"] = warm_start(trainer, guard)
        save()
        if pause():
            return 7
    while not trainer.done():
        guard.check()
        trainer.train_step()
        if trainer.transitions % 16384 == 0 or trainer.done():
            rows = evaluate_policy(trainer.model, trainer.model.params, source, task, curve_seeds)
            trainer.curve.append({"transitions": trainer.transitions, "score": mean_success(rows)})
        save()
        if pause():
            return 7
    if "after" not in extra["evaluations"]:
        extra["evaluations"]["after"] = evaluate_policy(
            trainer.model, trainer.model.params, source, task, eval_seeds
        )
        save()
    if "random" not in extra["evaluations"]:
        extra["evaluations"]["random"] = evaluate_baseline(task, eval_seeds, "random")
        save()
    probe = state_probe(trainer.model, trainer.model.params, source, task, "dev_probe", c["seed"])
    counts = np.sum([r["action_counts"] for r in extra["evaluations"]["after"]], axis=0)
    extra["segments"].append({**guard.check(), "peak_memory_mb": peak_mb()})
    save()
    assert array_hash(trainer.source.base_magnitude) == core_hash
    result = {
        **identity,
        "phase": spec["phase"],
        "config": c,
        "architecture": c["arch"],
        "score": mean_success(extra["evaluations"]["after"]),
        "development_seed": c["seed"],
        "training_transitions": trainer.transitions,
        "training_episodes_started": trainer.episodes_started,
        "training_seed_rule": "seed_for(task,train,development_seed,episode_index)",
        "evaluation_seeds": eval_seeds,
        "curve_seeds": curve_seeds,
        "evaluations": extra["evaluations"],
        "state_probe": probe,
        "action_distribution": (counts / counts.sum()).tolist(),
        "dominant_action_fraction": float(counts.max() / counts.sum()),
        "policy_entropy": float(
            np.mean([r["policy_entropy"] for r in extra["evaluations"]["after"]])
        ),
        "training_history": trainer.history,
        "training_curve": trainer.curve,
        "wall_clock_training_seconds": trainer.wall,
        "peak_memory_mb": max(r["peak_memory_mb"] for r in extra["segments"]),
        "segments": extra["segments"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "initial_policy_sha256": params_hash(trainer.initial_params),
        "final_policy_sha256": params_hash(trainer.model.params),
        "fixed_core_sha256": core_hash,
        "final_core_sha256": array_hash(trainer.source.base_magnitude),
        "optimizer": "Adam",
        "lr_schedule": "linear decay to 5 percent floor",
        "imitation": extra["imitation"],
        "finite": trainer.model.finite(),
        "failure_reason": None,
        "status": "COMPLETE",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(spec["name"], result["score"], "probe", probe["score"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
