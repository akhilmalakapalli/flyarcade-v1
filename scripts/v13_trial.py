"""Run one v1.3 trial from a JSON spec, resumable under the unchanged resource guards.

A spec names a trainer configuration, an evaluation block and an output directory.
The process checkpoints the complete trainer (policy, optimiser, environments,
neural state, RNGs) at rollout/evaluation boundaries after 70 seconds and exits 7;
re-running the same command resumes exactly. A completed result is never rewritten.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from flyarcade.resources import ResourceGuard
from flyarcade_v13.study import (
    Trainer,
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

RESUME_EXIT = 7
PAUSE_SECONDS = 70
CURVE_INDEX_OFFSET = 100_000  # learning-curve episodes never reuse final evaluation seeds


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for key in ("name", "config", "evaluation", "output"):
        if key not in spec:
            raise ValueError(f"spec missing {key}")
    return spec


def identity(spec, trainer):
    c = trainer.config
    ident = {
        "name": spec["name"],
        "config_sha256": config_hash(c),
        "code_sha256": code_hash(),
        "evaluation": spec["evaluation"],
    }
    if c["source"] == "fly":
        ident["graph_sha256"] = graph_hash(load_topology(c["topology"]))
        path = c.get("standardizer") or standardizer_path(
            c["task"], c["topology"], c["ticks"], c["readout"]
        )
        ident["standardizer_sha256"] = digest(path)
    return ident


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    trainer = Trainer(spec["config"])
    ident = identity(spec, trainer)
    plan = spec.get("plan")
    if plan is not None:
        frozen = json.loads(Path(plan["path"]).read_text())
        if frozen["code_sha256"] != ident["code_sha256"]:
            raise ValueError("code changed after freeze: confirmatory run refused")
        if digest(plan["path"]) != plan["sha256"]:
            raise ValueError("run plan changed after freeze")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if any(result.get(k) != v for k, v in ident.items()):
            raise ValueError(f"{out}: completed result identity mismatch; file preserved")
        print(f"{spec['name']}: completed result preserved", flush=True)
        return 0
    guard = ResourceGuard(directory=out)
    checkpoint = out / "checkpoint.pkl"
    task, c, ev = trainer.task, trainer.config, spec["evaluation"]
    eval_seeds = [seed_for(task, ev["purpose"], c["seed"], i) for i in range(ev["episodes"])]
    curve_seeds = [
        seed_for(task, ev["purpose"], c["seed"], CURVE_INDEX_OFFSET + i)
        for i in range(c["curve_episodes"])
    ]
    if checkpoint.exists():
        state = trainer.load(checkpoint)
        extra = state["extra"]
        if extra["identity"] != ident:
            raise ValueError("checkpoint identity mismatch")
    else:
        extra = {"identity": ident, "evaluations": {}, "segments": []}
    trainer.extra = extra

    def save():
        state = trainer.state()
        state["extra"] = extra
        import pickle

        tmp = checkpoint.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(state, handle, protocol=5)
        tmp.replace(checkpoint)

    def pause():
        if guard.snapshot()["elapsed_seconds"] > PAUSE_SECONDS:
            extra["segments"].append(guard.check())
            save()
            print(
                f"{spec['name']}: checkpoint at {trainer.transitions}/{c['transitions']} "
                "transitions; resume",
                flush=True,
            )
            return True
        return False

    eval_source = make_source(c, ev.get("batch", 16))
    if "before" not in extra["evaluations"]:
        extra["evaluations"]["before"] = evaluate_policy(
            trainer.model, trainer.initial_params, eval_source, task, eval_seeds
        )
        extra["curve_before"] = mean_success(
            evaluate_policy(trainer.model, trainer.initial_params, eval_source, task, curve_seeds)
        )
        save()
    while not trainer.done():
        guard.check()
        row = trainer.train_step()
        if c["curve_episodes"] and (trainer.updates % c["eval_every"] == 0 or trainer.done()):
            rows = evaluate_policy(
                trainer.model, trainer.model.params, eval_source, task, curve_seeds
            )
            trainer.curve.append(
                {
                    "update": row["update"],
                    "transitions": row["transitions"],
                    "episodes": row["episodes"],
                    "success": mean_success(rows),
                    "entropy": row.get("entropy"),
                }
            )
        if not trainer.done() and pause():
            return RESUME_EXIT
    save()
    evaluations = {"after": {}}
    if ev.get("baselines", True):
        evaluations.update(random={"baseline": "random"}, reference={"baseline": "reference"})
        if task == "flappy":
            evaluations["heuristic_v12"] = {"baseline": "heuristic_v12"}
    masks = {}
    if ev.get("perturbations"):
        rng = np.random.default_rng(seed_for(task, ev["perturb_purpose"], c["seed"]))
        graph = load_topology(c["topology"])
        masks = {"edge": rng.random(graph.e) >= 0.1, "neuron": rng.random(graph.n) >= 0.1}
        evaluations.update(
            sensory_noise={"noise": 0.1},
            edge_ablation={"edge_mask": masks["edge"]},
            neuron_ablation={"neuron_mask": masks["neuron"]},
        )
    for key, kwargs in evaluations.items():
        if key in extra["evaluations"]:
            continue
        guard.check()
        if "baseline" in kwargs:
            rows = evaluate_baseline(task, eval_seeds, kwargs["baseline"])
        else:
            rows = evaluate_policy(
                trainer.model, trainer.model.params, eval_source, task, eval_seeds, **kwargs
            )
        extra["evaluations"][key] = rows
        save()
        if pause():
            return RESUME_EXIT
    if "state_probe" not in extra:
        extra["state_probe"] = state_probe(
            trainer.model, trainer.model.params, eval_source, task, ev["probe_purpose"], c["seed"]
        )
        extra["state_probe_before"] = state_probe(
            trainer.model, trainer.initial_params, eval_source, task, ev["probe_purpose"], c["seed"]
        )
    extra["segments"].append(guard.check())
    save()
    policy = out / "policy.npz"
    np.savez(
        policy,
        **{f"final_{k}": v for k, v in trainer.model.params.items()},
        **{f"initial_{k}": v for k, v in trainer.initial_params.items()},
    )
    means = {k: mean_success(v) for k, v in extra["evaluations"].items()}
    result = {
        **ident,
        "status": "COMPLETE",
        "task": task,
        "config": c,
        "evaluation_seeds": eval_seeds,
        "curve_seeds": curve_seeds,
        "means": means,
        "evaluations": extra["evaluations"],
        "state_probe": extra["state_probe"],
        "state_probe_before": extra["state_probe_before"],
        "curve_before": extra["curve_before"],
        "curve": trainer.curve,
        "history": trainer.history,
        "level_transitions": trainer.level_transitions,
        "transitions": trainer.transitions,
        "episodes": len(trainer.completed),
        "updates": trainer.updates,
        "train_wall_seconds": trainer.wall,
        "segments": extra["segments"],
        "initial_params_sha256": params_hash(trainer.initial_params),
        "final_params_sha256": params_hash(trainer.model.params),
        "optimizer_steps": trainer.optimizer.t,
        "finite": trainer.model.finite()
        and all(np.isfinite(v).all() for v in trainer.optimizer.m.values()),
        "policy_npz_sha256": digest(policy),
        "checkpoint_pkl_sha256": digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "perturbation_mask_sha256": {
            k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in masks.items()
        },
        "evaluation_mode": "greedy argmax, no learning, fresh neural and recurrent state",
    }
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"trial": spec["name"], **means}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
