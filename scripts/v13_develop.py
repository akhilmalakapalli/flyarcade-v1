"""Bounded v1.3 development: plan, stage spec generation and result collection.

    v13_develop.py plan                      write experiments/v13_development_plan.json
    v13_develop.py stage NAME                write specs for a predeclared stage
    v13_develop.py collect NAME              aggregate a completed stage

Every generated stage is appended to the plan's ``stages_generated`` log with its
full configuration list before any of its trials run, and every result is appended
to artifacts/v13/development/log.jsonl. Only development seed purposes are used.
"""

import argparse
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flyarcade_v13.environments import TARGETS
from flyarcade_v13.gates import seed_summary, task_gate
from flyarcade_v13.study import (
    CONFIRMATORY_PURPOSES,
    DEFAULTS,
    DEVELOPMENT_PURPOSES,
    PURPOSE_BLOCK,
    SEED_BASE,
    TASK_BLOCK,
    purpose_range,
)

PLAN = Path("experiments/v13_development_plan.json")
ROOT = Path("artifacts/v13/development")
SCREEN_BUDGET = {"pong": 150_000, "flappy": 200_000, "breakout": 200_000, "snake": 300_000}
FULL_BUDGET = {"pong": 300_000, "flappy": 500_000, "breakout": 500_000, "snake": 1_000_000}
HISTORICAL_SEED_BLOCKS = {
    "v1 training 1000+10000*seed+ep, evaluation 3000000+100*seed+i (+17/+500000 derived)": [
        0,
        3_600_000,
    ],
    "v1.1 development/confirmatory 4000000-6000500": [4_000_000, 6_100_000],
    "historical absolute-action Snake 7000000-9000500": [7_000_000, 9_100_000],
    "v1.2 all purposes 100000000-140000000": [100_000_000, 140_000_000],
    "v1.2 derived neural RNG (+1e9)": [1_100_000_000, 1_140_000_000],
    "v1.2 derived behaviour RNG (+2e9)": [2_100_000_000, 2_140_000_000],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def seed_table():
    table = {}
    for task in TARGETS:
        table[task] = {p: list(purpose_range(task, p)) for p in DEVELOPMENT_PURPOSES}
        table[task].update({p: list(purpose_range(task, p)) for p in CONFIRMATORY_PURPOSES})
    return table


def check_disjoint(table):
    blocks = []
    for task, purposes in table.items():
        for purpose, (lo, hi) in purposes.items():
            for offset in (0, 1_000_000_000, 2_000_000_000):
                blocks.append((lo + offset, hi + offset, f"{task}/{purpose}+{offset}"))
    for label, (lo, hi) in HISTORICAL_SEED_BLOCKS.items():
        blocks.append((lo, hi, label))
    blocks.sort()
    for (lo1, hi1, a), (lo2, _, b) in zip(blocks, blocks[1:], strict=False):
        assert hi1 <= lo2, f"seed overlap: {a} / {b}"
    return True


def write_plan():
    if PLAN.exists():
        print(f"{PLAN}: preserved")
        return
    table = seed_table()
    plan = {
        "version": "1.3-development",
        "created_at": now(),
        "status": "development (mutable log; not a confirmatory protocol)",
        "seed_namespace": {
            "rule": "200e6 + task_index*100e6 + purpose_index*5e6 + seed*250000 + index",
            "base": SEED_BASE,
            "task_block": TASK_BLOCK,
            "purpose_block": PURPOSE_BLOCK,
            "task_order": list(TARGETS),
            "development_purposes": list(DEVELOPMENT_PURPOSES),
            "reserved_confirmatory_purposes_never_used_in_development": list(CONFIRMATORY_PURPOSES),
            "derived": "neural input RNG = episode seed + 1e9; random-baseline RNG = seed + 2e9",
            "ranges": table,
            "historical_blocks": HISTORICAL_SEED_BLOCKS,
            "disjoint_verified": check_disjoint(table),
        },
        "capacity_ladder": [
            "S: sensory-only PPO (engineered observation -> same learner family)",
            "A2: frozen fly core + MLP PPO",
            "A3: frozen fly core + GRU PPO (only if A2 inadequate)",
            "curriculum if sparse reward limits learning",
            "feature integration / readout scope variants if diagnostics justify",
            "B/C recurrent plasticity only as fallback",
        ],
        "budgets": {
            "screen_transitions": SCREEN_BUDGET,
            "full_transitions": FULL_BUDGET,
            "screen_seeds": 2,
            "screen_eval_episodes": 40,
            "validation_seeds": 5,
            "validation_eval_episodes": 60,
        },
        "defaults": DEFAULTS,
        "selection_rule": (
            "screening rank = mean dev gap closure - 0.5 * SD across seeds; configs with a "
            "constant-action or non-finite seed are ranked last. Top 3 per task go to full "
            "validation. Among validated configs passing the development gate, the simplest "
            "rung (sensory < MLP < GRU < curriculum) is chosen, then highest mean gap closure."
        ),
        "development_gate": {
            "per_task_over_5_validation_seeds": [
                "mean after > mean before",
                "mean after > mean random + 0.05",
                ">= 4/5 seeds individually > matched random + 0.05",
                "mean fixed-history state dependence > 0.05",
                "finite model/optimizer values",
                "final rollout policy entropy > 1e-3 on every seed",
                "no constant-action collapse (greedy top-action fraction < 0.98 every seed)",
                "mean gap closure >= 0.15 (preferred >= 0.25)",
            ],
            "reference_policy": {
                "flappy": "MPC oracle (stricter than the 0.0875 v1.2 heuristic)",
                "pong": "v1.2 heuristic",
                "breakout": "v1.2 heuristic",
                "snake": "v1.2 heuristic",
            },
        },
        "stages_generated": [],
        "amendments": [],
    }
    PLAN.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{PLAN}: written")


# ------------------------------------------------------------------ stages
def spec(stage, task, index, seed, config, evaluation):
    name = f"{stage}-{task}-c{index:02d}-s{seed}"
    return {
        "name": name,
        "stage": stage,
        "config_index": index,
        "config": {"task": task, "seed": seed, **config},
        "evaluation": evaluation,
        "output": f"runs/v13-dev/{stage}/{name}",
    }


SCREEN_EVAL = {"purpose": "screen_eval", "episodes": 40, "probe_purpose": "dev_probe"}
VALIDATE_EVAL = {"purpose": "validate_eval", "episodes": 60, "probe_purpose": "dev_probe"}
SCREEN = {"train_purpose": "screen_train", "eval_purpose": "screen_eval"}
VALIDATE = {"train_purpose": "validate_train", "eval_purpose": "validate_eval"}


def grid(**factors):
    keys = list(factors)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*factors.values())]


def stage_sensory_screen(tasks):
    configs = grid(lr=[3e-4, 1e-3], shaping=[0.0, 1.0], ent=[0.003, 0.01])
    specs = []
    for task in tasks:
        for i, c in enumerate(configs):
            for seed in (0, 1):
                full = {"source": "sensory", "transitions": SCREEN_BUDGET[task], **SCREEN, **c}
                specs.append(spec("sensory_screen", task, i, seed, full, SCREEN_EVAL))
    return specs


def stage_mlp_screen(tasks):
    """A2 coarse screen: 2^4 factorial over learning rate, entropy, ticks and shaping."""
    configs = grid(lr=[3e-4, 1e-3], ent=[0.003, 0.01], ticks=[4, 8], shaping=[0.0, 1.0])
    specs = []
    for task in tasks:
        for i, c in enumerate(configs):
            for seed in (0, 1):
                full = {"source": "fly", "arch": "mlp", "transitions": SCREEN_BUDGET[task], **SCREEN, **c}
                specs.append(spec("mlp_screen", task, i, seed, full, SCREEN_EVAL))
    return specs


def _top(stage, task, count):
    rows = json.loads((ROOT / f"{stage}.json").read_text())["configs"]
    ranked = [r for r in rows if r["task"] == task and not r["collapsed_seed"]]
    ranked.sort(key=lambda r: -r["rank_score"])
    chosen = []
    for row in ranked[:count]:
        spec_path = ROOT / "specs" / stage / f"{stage}-{task}-c{row['config_index']:02d}-s0.json"
        config = json.loads(spec_path.read_text())["config"]
        config = {k: v for k, v in config.items() if k not in ("seed", "task")}
        chosen.append((row["config_index"], config))
    return chosen


def validation_stage(name, source_stage, count):
    def build(tasks):
        specs = []
        for task in tasks:
            for index, config in _top(source_stage, task, count):
                config = {**config, **VALIDATE, "transitions": FULL_BUDGET[task]}
                for seed in range(5):
                    specs.append(spec(name, task, index, seed, config, VALIDATE_EVAL))
        return specs

    return build


STAGES = {
    "sensory_screen": stage_sensory_screen,
    "mlp_screen": stage_mlp_screen,
    "mlp_validate": validation_stage("mlp_validate", "mlp_screen", 3),
    "sensory_validate": validation_stage("sensory_validate", "sensory_screen", 1),
}


def load_stage_module():
    extra = Path("experiments/v13_stages.json")
    if extra.exists():
        return json.loads(extra.read_text())
    return {}


def generate(name, tasks):
    declared = load_stage_module()
    if name in STAGES:
        specs = STAGES[name](tasks)
    elif name in declared:
        specs = expand_declared(name, declared[name], tasks)
    else:
        raise SystemExit(f"unknown stage {name}")
    directory = ROOT / "specs" / name
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    written = 0
    for s in specs:
        path = directory / f"{s['name']}.json"
        if path.exists():
            if json.loads(path.read_text()) != s:
                raise SystemExit(f"{path} exists with a different configuration; refusing")
            continue
        path.write_text(json.dumps(s, indent=1) + "\n")
        written += 1
    plan["stages_generated"].append(
        {
            "stage": name,
            "generated_at": now(),
            "tasks": tasks,
            "specs": len(specs),
            "new_specs": written,
            "configs": sorted(
                {json.dumps({**s["config"], "seed": None}, sort_keys=True) for s in specs}
            ),
            "evaluation": specs[0]["evaluation"] if specs else None,
        }
    )
    PLAN.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{name}: {len(specs)} specs ({written} new) in {directory}")


def expand_declared(name, declaration, tasks):
    """Stages declared in experiments/v13_stages.json (added as development proceeds)."""
    specs = []
    evaluation = VALIDATE_EVAL if declaration.get("validation") else SCREEN_EVAL
    purposes = VALIDATE if declaration.get("validation") else SCREEN
    seeds = declaration.get("seeds", [0, 1])
    for task in tasks:
        entries = declaration["tasks"].get(task)
        if entries is None:
            continue
        for i, c in enumerate(entries):
            budget = (FULL_BUDGET if declaration.get("full_budget") else SCREEN_BUDGET)[task]
            for seed in seeds:
                config = {"transitions": budget, **purposes, **declaration.get("common", {}), **c}
                specs.append(spec(name, task, c.get("index", i) if "index" in c else i, seed,
                                  {k: v for k, v in config.items() if k != "index"}, evaluation))  # fmt: skip
    return specs


# ------------------------------------------------------------------ collection
def collect(name):
    results = []
    for path in sorted(Path(f"runs/v13-dev/{name}").glob("*/result.json")):
        results.append(json.loads(path.read_text()))
    specs = sorted((ROOT / "specs" / name).glob("*.json"))
    groups = {}
    for r in results:
        m = re.match(rf"{re.escape(name)}-(\w+)-c(\d+)-s(\d+)", r["name"])
        task, index = m.group(1), int(m.group(2))
        groups.setdefault((task, index), []).append(r)
    rows = []
    log = ROOT / "log.jsonl"
    logged = set()
    if log.exists():
        logged = {json.loads(line)["name"] for line in log.read_text().splitlines() if line}
    with open(log, "a") as handle:
        for r in results:
            if r["name"] in logged:
                continue
            handle.write(
                json.dumps(
                    {
                        "name": r["name"],
                        "stage": name,
                        "config": r["config"],
                        "summary": seed_summary(r),
                        "state_probe_before": r["state_probe_before"]["score"],
                        "transitions": r["transitions"],
                        "episodes": r["episodes"],
                        "train_wall_seconds": r["train_wall_seconds"],
                        "curve": [(c["transitions"], c["success"]) for c in r["curve"]],
                        "result_sha256": None,
                    }
                )
                + "\n"
            )
    for (task, index), group in sorted(groups.items()):
        summaries = [seed_summary(r) for r in group]
        gaps = [s["gap_closure"] for s in summaries]
        collapsed = any(
            s["max_greedy_action_fraction"] >= 0.98 or not s["finite"] for s in summaries
        )
        config = {k: v for k, v in group[0]["config"].items() if k != "seed"}
        rows.append(
            {
                "task": task,
                "config_index": index,
                "seeds": len(group),
                "mean_after": float(np.mean([s["after"] for s in summaries])),
                "mean_before": float(np.mean([s["before"] for s in summaries])),
                "mean_random": float(np.mean([s["random"] for s in summaries])),
                "mean_reference": float(np.mean([s["reference"] for s in summaries])),
                "mean_gap_closure": float(np.mean(gaps)),
                "sd_gap_closure": float(np.std(gaps)),
                "mean_state_probe": float(np.mean([s["state_probe"] for s in summaries])),
                "collapsed_seed": collapsed,
                "rank_score": (-1e9 if collapsed else 0)
                + float(np.mean(gaps) - 0.5 * np.std(gaps)),
                "gate_if_validation": task_gate(group) if len(group) >= 5 else None,
                "config": {
                    k: config[k]
                    for k in sorted(config)
                    if DEFAULTS.get(k) != config[k] or k in ("task", "source", "transitions")
                },
                "wall_seconds": float(np.mean([r["train_wall_seconds"] for r in group])),
            }
        )
    rows.sort(key=lambda r: (r["task"], -r["rank_score"]))
    out = ROOT / f"{name}.json"
    out.write_text(
        json.dumps(
            {
                "stage": name,
                "collected_at": now(),
                "results": len(results),
                "specs": len(specs),
                "configs": rows,
            },
            indent=1,
        )
        + "\n"
    )
    for row in rows:
        gate = row["gate_if_validation"]
        print(
            f"{row['task']:8s} c{row['config_index']:02d} n={row['seeds']} after={row['mean_after']:.3f} "
            f"rand={row['mean_random']:.3f} ref={row['mean_reference']:.3f} gap={row['mean_gap_closure']:.3f}"
            f"±{row['sd_gap_closure']:.3f} probe={row['mean_state_probe']:.2f} "
            f"{'COLLAPSE ' if row['collapsed_seed'] else ''}"
            f"{'' if gate is None else ('GATE PASS' if gate['passed'] else 'gate fail ' + ','.join(k for k, v in gate['checks'].items() if not v))} "
            f"{row['config']}"
        )
    print(f"{len(results)}/{len(specs)} results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "stage", "collect", "amend"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--tasks", default=",".join(TARGETS))
    parser.add_argument("--note")
    args = parser.parse_args()
    if args.command == "plan":
        write_plan()
    elif args.command == "stage":
        generate(args.name, args.tasks.split(","))
    elif args.command == "collect":
        collect(args.name)
    elif args.command == "amend":
        plan = json.loads(PLAN.read_text())
        plan["amendments"].append({"at": now(), "note": args.note})
        PLAN.write_text(json.dumps(plan, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
