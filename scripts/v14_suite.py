"""Bounded primary comparison then declared secondary factors; lock before validation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

from flyarcade_v14.selection import select, summarize
from flyarcade_v14.study import code_hash, digest

ROOT = Path("artifacts/v14")
PLAN = Path("experiments/v14/development_plan.json")


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"identity mismatch: {path}"
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")


def run_configuration(task, phase, config, plan):
    label = f"{task}-{phase}-{config['arch']}"
    paths = []
    results = []
    subprocess.run(
        [
            sys.executable,
            "scripts/v14_prepare.py",
            "--task",
            task,
            "--ticks",
            str(config["ticks"]),
            "--readout",
            config["readout"],
        ],
        check=True,
        timeout=125,
    )
    for seed in plan["training_seeds"]:
        name = f"{label}-s{seed}"
        spec = {
            "name": name,
            "phase": phase,
            "config": {**config, "seed": seed},
            "plan_sha256": digest(PLAN),
            "code_sha256": code_hash(),
        }
        path = Path("experiments/v14/specs") / f"{name}.json"
        write_once(path, spec)
        paths.append(str(path))
        result_path = Path("runs/v14") / name / "result.json"
        failure_path = ROOT / f"{name}-failure.json"
        if failure_path.exists():
            results.append(json.loads(failure_path.read_text()))
            continue
        for segment in range(20):
            result = subprocess.run(
                [sys.executable, "scripts/v14_trial.py", "--spec", str(path)], timeout=125
            )
            if result.returncode == 0:
                break
            if result.returncode != 7:
                failure = {
                    "status": "FAILED",
                    "name": name,
                    "failure_reason": f"exit {result.returncode}",
                    "spec_sha256": digest(path),
                    "phase": phase,
                }
                (ROOT / f"{name}-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                results.append(failure)
                break
        else:
            failure = {
                "status": "FAILED",
                "name": name,
                "failure_reason": "20-segment limit",
                "spec_sha256": digest(path),
                "phase": phase,
            }
            (ROOT / f"{name}-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            results.append(failure)
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
    record = summarize(label, phase, config, results)
    record["specs"] = paths
    output = ROOT / "configurations" / f"{label}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        "CONFIGURATION", label, "mean", record["mean"], "eligible", record["eligible"], flush=True
    )
    return record


def main():
    plan = json.loads(PLAN.read_text())
    manifest = json.loads((ROOT / "implementation.json").read_text())
    assert manifest["code_sha256"] == code_hash()
    primary = {}
    for task in plan["tasks"]:
        rows = []
        for arch in plan["primary_learners"]:
            config = {
                **plan["ppo"],
                "task": task,
                "arch": arch,
                **plan["architectures"][arch],
                "source": "fly",
                "topology": "biological",
                "ticks": 4,
                "readout": "descending",
                "transitions": plan["primary_transitions"],
            }
            rows.append(run_configuration(task, "primary", config, plan))
        primary[task] = {"configurations": rows, "selected": select(rows)}
        write_once(ROOT / f"{task}-primary-selection.json", primary[task])
    # Secondary runs cannot start until every primary architecture comparison exists.
    selected = {}
    for task in plan["tasks"]:
        rows = list(primary[task]["configurations"])
        best = select(rows)
        reasons = []
        for factor in plan["secondary_protocol"]:
            if best is None:
                reasons.append("all primary runs unstable or failed")
                break
            if best["eligible"] and best["mean"] >= plan["development_targets"][task]:
                reasons.append("development target reached; no further escalation")
                break
            if task not in factor["tasks"]:
                continue
            config = copy.deepcopy(best["config"])
            phase = factor["name"]
            if phase == "ticks":
                config["ticks"] = 8
            elif phase == "all_readout":
                config["readout"] = "all"
                config["transitions"] = 65536
            elif phase == "longer":
                config["transitions"] = 131072
            elif phase == "curriculum":
                config["transitions"] = 131072
                config["curriculum"] = [["easy", 0.25], ["intermediate", 0.25], ["target", 0.5]]
            rows.append(run_configuration(task, phase, config, plan))
            best = select(rows)
        rescue = plan["rescue"]
        if (
            best is not None
            and task in rescue["tasks"]
            and (not best["eligible"] or best["mean"] < plan["useful_performance"][task])
        ):
            config = copy.deepcopy(best["config"])
            config["transitions"] = 131072
            config["curriculum"] = [["target", 1.0]]
            rows.append(run_configuration(task, "rescue", config, plan))
            best = select(rows)
        assert len(rows) <= plan["max_development_configurations"][task]
        selected[task] = {
            "primary_selected": primary[task]["selected"],
            "final_selected": best,
            "configurations": rows,
            "stopping_reasons": reasons or ["bounded protocol exhausted"],
        }
        (ROOT / "progress.json").write_text(json.dumps(selected, indent=2) + "\n")
    write_once(
        ROOT / "selected_configs.json",
        {
            "plan_sha256": digest(PLAN),
            "code_sha256": code_hash(),
            "selection_data": "development only",
            "tasks": selected,
        },
    )
    print(
        "All selections locked. Ready for fresh validation; no confirmatory execution.", flush=True
    )


if __name__ == "__main__":
    main()
