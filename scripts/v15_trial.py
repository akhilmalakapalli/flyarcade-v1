"""One bounded, resumable v1.5 trial. Exit 0 = complete, 7 = checkpointed, resume me.

Never touches validation or confirmatory seeds: training uses train/diagnosis_training
seeds, checkpoint choice uses checkpoint_select seeds, the score uses dev_eval seeds.
"""

import argparse
import json
import os
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import v15_path  # noqa: F401
from flyarcade_v15.provenance import code_hash
from flyarcade_v15.seeds import seed_for
from flyarcade_v15.trainer import (
    Trainer,
    array_hash,
    config_hash,
    digest,
    ensure_standardizer,
    evaluate,
    params_hash,
    state_probe,
    summarize_rows,
)

from flyarcade.resources import ResourceGuard
from flyarcade_v14.study import evaluate_baseline, graph_hash, load_topology

PAUSE_SECONDS = float(os.environ.get("V15_PAUSE_SECONDS", 70))
DEV_EVAL_EPISODES = 100


def peak_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == "darwin" else 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    assert spec["code_sha256"] == code_hash(), "code changed since the spec was written"
    out = Path("runs/v15") / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    identity = {
        "name": spec["name"],
        "phase": spec["phase"],
        "spec_sha256": digest(spec_path),
        "code_sha256": code_hash(),
    }
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert all(result[k] == v for k, v in identity.items())
        return 0
    guard = ResourceGuard(directory=out)
    config = spec["config"]
    if config.get("source", "fly") == "fly":
        std = ensure_standardizer(
            config["task"],
            config.get("ticks", 4),
            config.get("readout", "descending"),
            config.get("std_floor", 0.02),
            guard=guard,
        )
    else:
        std = None
    trainer = Trainer(config)
    c, task = trainer.config, trainer.task
    checkpoint = out / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            stored = pickle.load(stream)
        trainer.load(stored)
        extra = stored["extra"]
        assert extra["identity"] == identity
    else:
        extra = {"identity": identity, "segments": [], "final": {}}

    def save():
        state = trainer.state()
        state["extra"] = extra
        tmp = checkpoint.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            pickle.dump(state, stream, protocol=5)
        tmp.replace(checkpoint)

    def pause():
        if guard.snapshot()["elapsed_seconds"] > PAUSE_SECONDS:
            extra["segments"].append({**guard.check(), "peak_memory_mb": peak_mb()})
            save()
            return True
        return False

    core_hash = array_hash(trainer.source.base_magnitude) if c["source"] == "fly" else None
    if c["imitation"]:
        raise NotImplementedError("imitation warm-start is a rescue method; not enabled")
    while not trainer.done():
        guard.check()
        trainer.train_step()
        if trainer.checkpoint_due():
            trainer.evaluate_checkpoint()
        save()
        if pause():
            return 7
    params, chosen_at = trainer.chosen_params()
    dev_seeds = [seed_for(task, "dev_eval", c["seed"], i) for i in range(DEV_EVAL_EPISODES)]
    final = extra["final"]
    if "chosen" not in final:
        final["chosen"] = evaluate(c, params, dev_seeds)
        save()
        if pause():
            return 7
    if "last" not in final and chosen_at != trainer.transitions:
        final["last"] = evaluate(c, trainer.model.params, dev_seeds)
        save()
        if pause():
            return 7
    if "probe" not in final:
        final["probe"] = state_probe(c, params, "dev_probe", c["seed"])
        final["random"] = evaluate_baseline(task, dev_seeds, "random")
        final["reference"] = evaluate_baseline(task, dev_seeds, "reference")
        save()
    extra["segments"].append({**guard.check(), "peak_memory_mb": peak_mb()})
    save()
    if core_hash is not None:
        assert array_hash(trainer.source.base_magnitude) == core_hash
    policy_path = out / "policy.npz"
    np.savez(policy_path, **params)
    chosen = summarize_rows(final["chosen"])
    last = summarize_rows(final["last"]) if "last" in final else chosen
    h = trainer.history
    result = {
        **identity,
        "task": task,
        "config": c,
        "config_sha256": config_hash(c),
        "seed": c["seed"],
        "graph_sha256": graph_hash(load_topology("biological")),
        "standardizer": str(std) if std else None,
        "standardizer_sha256": digest(std) if std else None,
        "score": chosen["score"],
        "chosen_checkpoint_transitions": chosen_at,
        "last_checkpoint_score": last["score"],
        "action_distribution": chosen["action_distribution"],
        "dominant_action_fraction": chosen["dominant_action_fraction"],
        "policy_entropy": chosen["policy_entropy"],
        "state_probe": final["probe"],
        "random_score": float(np.mean([r["success"] for r in final["random"]])),
        "reference_score": float(np.mean([r["success"] for r in final["reference"]])),
        "dev_eval_rows": final["chosen"],
        "dev_eval_seeds": dev_seeds,
        "training_curve": trainer.curve,
        "training_history": h,
        "training_transitions": trainer.transitions,
        "training_episodes_started": trainer.episodes_started,
        "level_transitions": trainer.level_transitions,
        "wall_clock_training_seconds": trainer.wall,
        "peak_memory_mb": max(s["peak_memory_mb"] for s in extra["segments"]),
        "segments": extra["segments"],
        "optimizer": "Adam(betas=0.9,0.999, eps=1e-5)",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "policy_path": str(policy_path),
        "policy_sha256": params_hash(params),
        "final_policy_sha256": params_hash(trainer.model.params),
        "initial_policy_sha256": params_hash(trainer.initial_params),
        "fixed_core_sha256": core_hash,
        "finite": trainer.model.finite() and all(np.isfinite(v).all() for v in params.values()),
        "failure_reason": None,
        "status": "COMPLETE",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        spec["name"],
        "score",
        round(result["score"], 4),
        "probe",
        final["probe"]["score"],
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
