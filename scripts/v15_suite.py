"""Execute the committed v1.5 development plan: stages A, B, C, rescue; lock selections.

Every branch below is pre-specified in experiments/v15/development_plan.json. Completed
trials and failure records are preserved; re-running resumes. Validation seeds are
never touched here.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import v15_path  # noqa: F401
from flyarcade_v15.provenance import code_hash
from flyarcade_v15.selection import select, still_rising, summarize
from flyarcade_v15.trainer import digest

PLAN = Path("experiments/v15/development_plan.json")
ROOT = Path("artifacts/v15")
SPECS = Path("experiments/v15/specs/development")
WORKERS = 8


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"identity mismatch: {path}"
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")


def tag(config):
    if config["source"] == "sensory":
        return f"{config['arch']}-sensory"
    parts = [config["arch"]]
    if config["arch"] != "linear" and config["hidden"] != 128:
        parts.append(f"h{config['hidden']}")
    parts += [f"t{config['ticks']}", config["readout"]]
    return "-".join(parts)


def base_config(plan, task, **overrides):
    return {
        **plan["ppo"],
        "task": task,
        "source": "fly",
        "readout": "descending",
        "transitions": plan["budgets"][task],
        **overrides,
    }


def run_batch(plan, jobs):
    """jobs: list of (task, stage, name, config, role). Returns records in order."""
    specs = []
    for task, stage, name, config, role in jobs:
        for seed in plan["development_seeds"]:
            spec = {
                "name": f"{name}-s{seed}",
                "phase": stage,
                "role": role,
                "plan_sha256": digest(PLAN),
                "code_sha256": code_hash(),
                "config": {**config, "seed": seed},
            }
            path = SPECS / f"{spec['name']}.json"
            write_once(path, spec)
            specs.append(str(path))
    pending = [s for s in specs if not (Path("runs/v15") / Path(s).stem / "result.json").exists()]
    if pending:
        subprocess.run([sys.executable, "scripts/v15_prepare.py", *pending], check=True)
        subprocess.run(
            [sys.executable, "scripts/v15_run.py", *pending, "--workers", str(WORKERS)], check=True
        )
    records = []
    for task, stage, name, config, role in jobs:
        results = []
        for seed in plan["development_seeds"]:
            run = Path("runs/v15") / f"{name}-s{seed}"
            if (run / "result.json").exists():
                results.append(json.loads((run / "result.json").read_text()))
            else:
                failure = json.loads((run / "failure.json").read_text())
                results.append({**failure, "status": "FAILED"})
        record = summarize(name, stage, config, results, role=role)
        record["specs"] = [str(SPECS / f"{name}-s{s}.json") for s in plan["development_seeds"]]
        record["results"] = results
        write_once(ROOT / "configurations" / f"{name}.json", strip(record))
        records.append(record)
        print(
            f"CONFIG {name:48s} role={role:8s} mean={record['mean']} "
            f"scores={record['scores']} eligible={record['eligible']}",
            flush=True,
        )
    return records


def strip(record):
    return {k: v for k, v in record.items() if k != "results"}


def best_of(records):
    best = select(records, "headline")
    if best is None:  # fall back to best stable headline record, labelled
        stable = [r for r in records if r["role"] == "headline" and r["stable"]]
        best = max(stable, key=lambda r: r["utility"]) if stable else None
    return best


def main():
    plan = json.loads(PLAN.read_text())
    assert plan["code_sha256"] == code_hash(), "code changed after the plan was committed"
    assert plan["suite_sha256"] == digest(__file__), "suite changed after the plan was committed"
    tasks = plan["tasks"]
    records = {t: [] for t in tasks}
    log = {t: [] for t in tasks}

    # Stage A: learner x integration time at the leakage-free descending readout.
    jobs = []
    for task in tasks:
        for arm in plan["stage_A"]["arms"]:
            if arm.get("tasks") and task not in arm["tasks"]:
                continue
            config = base_config(plan, task, **arm["overrides"])
            name = f"{task}-A-{tag(config)}"
            jobs.append((task, "A", name, config, "headline"))
    for record in run_batch(plan, jobs):
        records[record["config"]["task"]].append(record)
    stage_a = {t: best_of(records[t]) for t in tasks}
    for t in tasks:
        log[t].append(f"stage A best: {stage_a[t]['name'] if stage_a[t] else None}")
    write_once(
        ROOT / "stage_A_selection.json", {t: strip(v) if v else None for t, v in stage_a.items()}
    )

    # Stage B: readout scope (headline) + leakage controls, on the stage-A winner.
    jobs = []
    for task in tasks:
        winner = stage_a[task]
        for arm in plan["stage_B"]["arms"]:
            config = copy.deepcopy(winner["config"])
            config.update(arm["overrides"])
            name = f"{task}-B-{tag(config)}"
            jobs.append((task, "B", name, config, arm["role"]))
    for record in run_batch(plan, jobs):
        records[record["config"]["task"]].append(record)
    best = {t: best_of(records[t]) for t in tasks}
    for t in tasks:
        log[t].append(f"stage B best: {best[t]['name']}")

    # Stage C: conditional, ordered, one change at a time on the current best.
    for step in plan["stage_C"]["steps"]:
        jobs = []
        for task in tasks:
            current = best[task]
            if current["mean"] >= plan["stretch_targets"][task]:
                log[task].append(f"{step['name']}: skipped, stretch target reached")
                continue
            if task not in step["tasks"]:
                continue
            config = copy.deepcopy(current["config"])
            if step["name"] == "C1_longer":
                results = [r for r in current["results"] if r.get("status") == "COMPLETE"]
                if not still_rising(results, config["transitions"]):
                    log[task].append("C1_longer: skipped, best checkpoints not in last 25%")
                    continue
                config["transitions"] *= 2
            elif step["name"] == "C2_capacity":
                if config["arch"] == "linear" or config["source"] == "sensory":
                    log[task].append("C2_capacity: skipped, linear learner")
                    continue
                config.update(hidden=256, core=128)
            elif step["name"] == "C3_optimisation":
                config.update(step["overrides"][task])
            name = f"{task}-{step['name'].split('_')[0]}-{tag(config)}"
            if config.get("transitions") != best[task]["config"]["transitions"]:
                name += f"-{config['transitions'] // 1024}k"
            if step["name"] == "C3_optimisation":
                name += "-" + step["suffix"][task]
            jobs.append((task, step["name"], name, config, "headline"))
        for record in run_batch(plan, jobs):
            task = record["config"]["task"]
            records[task].append(record)
            best[task] = best_of(records[task])
        for task in tasks:
            log[task].append(f"after {step['name']}: best {best[task]['name']}")

    # SECONDARY RESCUE PHASE: separately labelled; never enters the headline selection.
    jobs = []
    rescue = plan["rescue"]
    for task in tasks:
        if task in rescue["tasks"] and best[task]["mean"] < plan["useful_performance"][task]:
            config = copy.deepcopy(best[task]["config"])
            config["imitation"] = rescue["imitation_transitions"]
            jobs.append((task, "rescue", f"{task}-R-{tag(config)}-imitation", config, "rescue"))
    for record in run_batch(plan, jobs):
        records[record["config"]["task"]].append(record)

    selected = {}
    for task in tasks:
        headline = select(records[task], "headline")
        rescue_best = select(records[task], "rescue")
        assert len(records[task]) <= plan["max_development_configurations"][task]
        selected[task] = {
            "stage_A_architecture_winner": strip(stage_a[task]),
            "headline_selected": strip(headline) if headline else None,
            "rescue_selected": strip(rescue_best)
            if rescue_best and (headline is None or rescue_best["utility"] > headline["utility"])
            else None,
            "configurations": [strip(r) for r in records[task]],
            "log": log[task],
        }
    write_once(
        ROOT / "selected_configs.json",
        {
            "plan_sha256": digest(PLAN),
            "code_sha256": code_hash(),
            "selection_data": "development only",
            "tasks": selected,
        },
    )
    print("All v1.5 selections locked. Commit before validation.", flush=True)


if __name__ == "__main__":
    main()
