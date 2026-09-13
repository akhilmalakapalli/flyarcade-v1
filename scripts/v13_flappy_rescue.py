"""Flappy rescue: development validation gate, protocol freeze, confirmatory report.

  gate    - apply the rescue development-validation gate to runs/v13-flappy-rescue/validate
  freeze  - only if that gate passes and no plan exists: write
            experiments/v13_flappy/run_plan.json and the confirmatory specs
  report  - aggregate confirmatory results against the criterion frozen in the plan

Measurements reuse flyarcade_v13.gates.seed_summary (gap closure, state probe, greedy
action fraction, entropy, finiteness) so definitions match the multitask study; the
thresholds are the stricter rescue thresholds and are written into the plan at freeze.
Nothing here trains.
"""

import argparse
import copy
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import flyarcade_v13_flappy as rescue
from flyarcade_v13.gates import seed_summary
from flyarcade_v13.study import graph_hash, load_topology

ROOT = Path("artifacts/v13/flappy-rescue")
VALIDATE = Path("runs/v13-flappy-rescue/validate")
CONFIRM = Path("runs/v13-flappy-rescue/confirm")
SPECS = Path("experiments/v13_flappy/specs")
PLAN = Path("experiments/v13_flappy/run_plan.json")
SEEDS = (0, 1, 2, 3, 4)
CONF_EVAL_EPISODES = 100
THRESHOLDS = {
    "random_margin": 0.10,
    "min_seeds_beating_margin": 4,
    "mean_success_minimum": 0.60,
    "mean_success_preferred": 0.70,
    "gap_closure": 0.50,
    "state_dependence": 0.05,
    "constant_action_fraction": 0.98,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_block(results, *, confirmatory):
    t = THRESHOLDS
    seeds = {r["name"]: seed_summary(r) for r in results}
    rows = list(seeds.values())
    mean = lambda k: float(np.mean([s[k] for s in rows]))  # noqa: E731
    beating = sum(s["after"] > s["random"] + t["random_margin"] for s in rows)
    graphs_ok = all(
        r.get("graph_sha256") == graph_hash(load_topology(r["config"]["topology"]))
        for r in results
        if r["config"]["source"] == "fly"
    )
    checks = {
        "1_mean_after_gt_before": mean("after") > mean("before"),
        "2_mean_after_gt_random_plus_0.10": mean("after") > mean("random") + t["random_margin"],
        "3_at_least_4_of_5_seeds_beat_random_plus_0.10": beating >= t["min_seeds_beating_margin"],
        "4_mean_success_ge_0.60": mean("after") >= t["mean_success_minimum"],
        "5_mean_gap_closure_ge_0.50": mean("gap_closure") >= t["gap_closure"],
        "6_state_dependence_gt_0.05": mean("state_probe") > t["state_dependence"],
        "7_no_constant_action_policy": all(
            s["max_greedy_action_fraction"] < t["constant_action_fraction"] for s in rows
        ),
        "8_all_parameters_finite": all(s["finite"] for s in rows),
        "9_frozen_malecns_graphs_match_disk": graphs_ok,
    }
    if not confirmatory:
        checks["preferred_mean_ge_0.70 (reported, not required)"] = (
            mean("after") >= t["mean_success_preferred"]
        )
    required = {k: v for k, v in checks.items() if "not required" not in k}
    return {
        "passed": all(required.values()),
        "checks": checks,
        "means": {
            k: mean(k)
            for k in ("after", "before", "random", "reference", "gap_closure", "state_probe")
        },
        "seeds_beating_random_plus_0.10": beating,
        "seed_count": len(rows),
        "seeds": seeds,
    }


def load_results(directory, prefix=""):
    paths = sorted(Path(directory).glob(f"{prefix}*/result.json"))
    return [json.loads(p.read_text()) for p in paths]


def gate():
    results = load_results(VALIDATE)
    if len(results) != 5:
        raise SystemExit(f"expected 5 validation results, found {len(results)}")
    report = evaluate_block(results, confirmatory=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "validation_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "seeds"}, indent=2))
    return report


def freeze():
    report = gate()
    if not report["passed"]:
        raise SystemExit("development validation gate failed: freeze refused")
    if PLAN.exists():
        raise SystemExit(f"{PLAN} exists; a frozen plan is never rewritten")
    template = json.loads(next(iter(sorted((SPECS / "validate").glob("*.json")))).read_text())
    frozen = {
        k: v
        for k, v in template["config"].items()
        if k not in ("seed", "train_purpose", "eval_purpose", "standardizer")
    }
    ticks = frozen["ticks"]
    standardizers = {
        "biological": f"artifacts/v13/standardizers/flappy-biological-t{ticks}-descending.json"
    }
    for seed in SEEDS:
        standardizers[f"rewired-{seed}"] = (
            f"artifacts/v13/standardizers/flappy-rewired-{seed}-t{ticks}-descending.json"
        )
    missing = [p for p in standardizers.values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"fit these standardizers (same procedure) before freezing: {missing}")
    conf = {
        "train_purpose": "conf_train",
        "eval_purpose": "conf_eval",
        "model_purpose": "conf_model",
        "rollout_purpose": "conf_rollout",
    }
    evaluation = {
        "purpose": "conf_eval",
        "episodes": CONF_EVAL_EPISODES,
        "probe_purpose": "conf_probe",
    }
    written = []
    for condition in ("biological", "rewired", "sensory"):
        for seed in SEEDS:
            config = {**copy.deepcopy(frozen), **conf, "seed": seed}
            ev = dict(evaluation)
            if condition == "biological":
                config.update(topology="biological", standardizer=standardizers["biological"])
                ev.update(perturbations=True, perturb_purpose="conf_perturb")
            elif condition == "rewired":
                config.update(
                    topology=f"rewired-{seed}", standardizer=standardizers[f"rewired-{seed}"]
                )
            else:
                config["source"] = "sensory"
                config.pop("ticks", None)
            name = f"flappy-confirm-{condition}-s{seed}"
            spec = {
                "name": name,
                "stage": "flappy_rescue_confirm",
                "condition": condition,
                "config": config,
                "evaluation": ev,
                "output": str(CONFIRM / name),
            }
            path = SPECS / "confirm" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec, indent=1) + "\n")
            written.append(str(path))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    plan = {
        "task": "flappy",
        "frozen_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "frozen_after": "development validation gate passed; before any confirmatory run",
        "source_commit_at_freeze": commit,
        "code_sha256_extended": rescue.code_hash(),
        "base_multitask_code_sha256": rescue._ORIGINAL["code_hash"](),
        "environment": "unchanged flyarcade.v12.environments.Flappy (target level only)",
        "environment_source_sha256": sha("src/flyarcade/v12/environments.py"),
        "graph_sha256": {
            t: graph_hash(load_topology(t))
            for t in ["biological"] + [f"rewired-{s}" for s in SEEDS]
        },
        "standardizers": {t: {"path": p, "sha256": sha(p)} for t, p in standardizers.items()},
        "standardizer_procedure": (
            "flyarcade_v13.study.fit_standardizer on feature_fit development seeds, "
            "one per topology"
        ),
        "frozen_config": frozen,
        "frozen_config_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest(),
        "defaults_applied": "flyarcade_v13.study.DEFAULTS for every key not listed",
        "recurrent_core": "MaleCNS recurrent weights frozen (fixed-core Stage A); no extra neurons",
        "conditions": {
            "biological": (
                "authentic MaleCNS core + frozen learner/budget (+ post-training perturbations)"
            ),
            "untrained": "each biological trial's initial policy on the same seeds ('before')",
            "rewired": (
                "degree-preserving rewired core k for seed k, its own standardizer, "
                "same learner/budget"
            ),
            "sensory": "engineered observation into the same learner and budget, no SNN",
            "random": "uniform random actions on matched evaluation seeds",
            "mpc": "flappy MPC oracle on matched evaluation seeds",
        },
        "seeds": list(SEEDS),
        "seed_purposes": [
            "conf_train",
            "conf_model",
            "conf_rollout",
            "conf_eval",
            "conf_probe",
            "conf_perturb",
        ],
        "seed_note": "no Flappy confirmatory seed had been used by any session before this freeze",
        "evaluation": (
            f"{CONF_EVAL_EPISODES} greedy episodes per seed, learning off, "
            "target environment, fresh state"
        ),
        "thresholds": THRESHOLDS,
        "success_criterion": [
            "mean biological learned > mean biological untrained",
            "mean biological learned > random + 0.10",
            ">= 4/5 biological seeds individually > random + 0.10",
            "mean biological success >= 0.60",
            "mean random-to-MPC gap closure >= 0.50",
            "policy state dependence > 0.05",
            "no constant-action policy (greedy top-action fraction < 0.98 on every seed)",
            "all parameters finite",
            "saved checkpoints replay exactly",
            "frozen MaleCNS weights unchanged (graph hashes match)",
        ],
        "target": (
            "mean >= 0.70 (reported; the protocol is not changed if confirmatory lands below it)"
        ),
        "development_gate": str(ROOT / "validation_gate.json"),
        "development_gate_sha256": sha(ROOT / "validation_gate.json"),
        "specs": written,
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"froze {PLAN} with {len(written)} confirmatory specs")


def report():
    plan = json.loads(PLAN.read_text())
    for spec_path in plan["specs"]:  # results must come from exactly the frozen specs
        spec = json.loads(Path(spec_path).read_text())
        if not (Path(spec["output"]) / "result.json").exists():
            raise SystemExit(f"missing confirmatory result for frozen spec {spec_path}")
    blocks = {
        c: load_results(CONFIRM, f"flappy-confirm-{c}-")
        for c in ("biological", "rewired", "sensory")
    }
    if any(len(v) != 5 for v in blocks.values()):
        raise SystemExit({c: len(v) for c, v in blocks.items()})
    criterion = evaluate_block(blocks["biological"], confirmatory=True)
    per = {c: {r["name"]: seed_summary(r) for r in rs} for c, rs in blocks.items()}
    means = lambda c, k: float(np.mean([s[k] for s in per[c].values()]))  # noqa: E731
    bio = blocks["biological"]
    perturb = {}
    for key in ("sensory_noise", "edge_ablation", "neuron_ablation"):
        if all(key in r["evaluations"] for r in bio):
            perturb[key] = float(
                np.mean([np.mean([x["success"] for x in r["evaluations"][key]]) for r in bio])
            )
    summary = {
        "plan_sha256": sha(PLAN),
        "biological_after": means("biological", "after"),
        "biological_before_untrained": means("biological", "before"),
        "rewired_after": means("rewired", "after"),
        "sensory_after": means("sensory", "after"),
        "random": means("biological", "random"),
        "mpc": means("biological", "reference"),
        "gap_closure": means("biological", "gap_closure"),
        "state_dependence": means("biological", "state_probe"),
        "perturbations_biological": perturb,
        "criterion_without_replay": criterion,
        "per_seed": per,
    }
    (ROOT / "confirmatory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("per_seed", "criterion_without_replay")},
            indent=2,
        )
    )
    print("criterion (replay checked separately):", criterion["passed"], criterion["checks"])
    for c, rows in per.items():
        print(c, [round(s["after"], 3) for s in rows.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate", "freeze", "report"])
    args = parser.parse_args()
    rescue.install()
    {"gate": gate, "freeze": freeze, "report": report}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
