"""Audit v1.4 development and validation without rerunning historical/validation seeds."""

import gzip
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from flyarcade.resources import ResourceGuard
from flyarcade_v14.selection import select
from flyarcade_v14.study import (
    PURPOSE_BLOCK,
    PURPOSES,
    SEED_SLOT,
    TASK_BLOCK,
    Trainer,
    build_model,
    code_hash,
    config_hash,
    digest,
    evaluate_policy,
    graph_hash,
    load_topology,
    make_source,
    params_hash,
    seed_for,
    standardizer_path,
)

ROOT = Path("artifacts/v14")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    guard = ResourceGuard()
    planpath = Path("experiments/v14/development_plan.json")
    plan = json.loads(planpath.read_text())
    selected = json.loads((ROOT / "selected_configs.json").read_text())
    validation = json.loads((ROOT / "validation_summary.json").read_text())
    summary = json.loads((ROOT / "development_summary.json").read_text())
    implementation = json.loads((ROOT / "implementation.json").read_text())
    assert implementation["code_sha256"] == selected["code_sha256"] == code_hash()
    assert validation["selection_sha256"] == digest(ROOT / "selected_configs.json")
    for filename, expected_hash in implementation["orchestration"].items():
        assert digest(filename) == expected_hash
    validation_manifest = json.loads((ROOT / "validation_manifest.json").read_text())
    assert validation_manifest["validation_script_sha256"] == digest("scripts/v14_validate.py")
    assert not plan["confirmatory_testing"] and not validation["confirmatory_testing"]
    assert (
        subprocess.check_output(["git", "branch", "--show-current"]).decode().strip()
        == "flyarcade-v1.4-performance"
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "ee7d012", implementation["code_commit"]], check=True
    )
    protected = json.loads((ROOT / "historical_hashes.json").read_text())
    for path, expected in protected.items():
        assert file_hash(path) == expected, f"historical mutation: {path}"
    intervals = []
    for task in plan["tasks"]:
        for purpose in PURPOSES:
            for offset in (0, 1000000000, 2000000000):
                lo = seed_for(task, purpose) + offset
                hi = lo + 3 * SEED_SLOT - 1
                assert lo > 12000000000
                assert all(hi < a or lo > b for a, b in intervals)
                intervals.append((lo, hi))
        base = plan["seed_namespace"]["base"] + plan["tasks"].index(task) * TASK_BLOCK
        for index in (15, 16, 17):
            reserved = (base + index * PURPOSE_BLOCK, base + (index + 1) * PURPOSE_BLOCK - 1)
            assert all(b < reserved[0] or a > reserved[1] for a, b in intervals)
    archive = json.loads(
        gzip.decompress((ROOT / "development_results_archive.json.gz").read_bytes())
    )
    results = {}
    configuration_count = {}
    curriculum = {}
    peaks = []
    segments = []
    for task, task_selection in selected["tasks"].items():
        configurations = task_selection["configurations"]
        configuration_count[task] = len(configurations)
        assert len(configurations) <= plan["max_development_configurations"][task]
        primary = [r for r in configurations if r["phase"] == "primary"]
        assert {r["config"]["arch"] for r in primary} == {"linear", "mlp", "gru"}
        assert task_selection["primary_selected"] == select(primary)
        assert task_selection["final_selected"] == select(configurations)
        for record in configurations:
            assert len(record["specs"]) == 3
            for specpath in record["specs"]:
                spec = json.loads(Path(specpath).read_text())
                c = spec["config"]
                seed = c["seed"]
                path = Path("runs/v14") / spec["name"] / "result.json"
                if not path.exists():
                    failure = json.loads((ROOT / f"{spec['name']}-failure.json").read_text())
                    assert failure["status"] == "FAILED"
                    continue
                r = json.loads(path.read_text())
                results[str(path)] = r
                assert archive[str(path)] == r
                assert digest(path) == summary["result_files"][str(path)]
                assert r["spec_sha256"] == digest(specpath)
                assert r["plan_sha256"] == digest(planpath)
                assert r["code_sha256"] == code_hash()
                assert r["graph_sha256"] == graph_hash(load_topology("biological"))
                std_path = standardizer_path(task, "biological", c["ticks"], c["readout"])
                assert digest(std_path) == r["standardizer_sha256"]
                meta = json.loads(std_path.with_suffix(".meta.json").read_text())
                assert meta["seeds"] == [
                    seed_for(task, "feature_fit", index=i) for i in range(len(meta["seeds"]))
                ]
                assert meta["samples"] >= 8192 and meta["normalise"] is False
                assert meta["plan_sha256"] == digest(planpath)
                for key, value in plan["ppo"].items():
                    assert c[key] == value, (task, key)
                assert r["fixed_core_sha256"] == r["final_core_sha256"]
                assert r["initial_policy_sha256"] != r["final_policy_sha256"]
                assert r["finite"] and r["failure_reason"] is None
                assert r["training_transitions"] == c["transitions"]
                assert len(r["training_history"]) == c["transitions"] // 2048
                assert [row["transitions"] for row in r["training_curve"]] == list(
                    range(16384, c["transitions"] + 1, 16384)
                )
                assert r["evaluation_seeds"] == [
                    seed_for(task, "dev_eval", seed, i) for i in range(40)
                ]
                assert r["score"] == float(
                    np.mean([row["success"] for row in r["evaluations"]["after"]])
                )
                if r["phase"] == "primary":
                    assert (
                        c["transitions"] == 65536
                        and c["ticks"] == 4
                        and c["readout"] == "descending"
                    )
                checkpoint = Path(r["checkpoint_path"])
                assert digest(checkpoint) == r["checkpoint_sha256"]
                with checkpoint.open("rb") as stream:
                    state = pickle.load(stream)
                assert params_hash(state["params"]) == r["final_policy_sha256"]
                assert state["config_sha256"] == config_hash(r["config"])
                if r["phase"] == "curriculum":
                    curriculum[spec["name"]] = state["level_transitions"]
                    assert sum(state["level_transitions"].values()) == c["transitions"]
                    assert state["level_transitions"]["target"] > 0
                if r["phase"] == "rescue":
                    assert r["imitation"]["transitions"] == 8192
                else:
                    assert r["imitation"] is None
                peaks.append(r["peak_memory_mb"])
                segments.extend(r["segments"])
    validation_files = {}
    for path in sorted((ROOT / "validation").glob("*.json")):
        r = json.loads(path.read_text())
        task = r["task"]
        seed = r["development_seed"]
        assert r["purpose"] in ("primary_validation", "final_validation") and not r["learning"]
        assert r["evaluation_seeds"] == [seed_for(task, r["purpose"], seed, i) for i in range(100)]
        assert r["selection_sha256"] == digest(ROOT / "selected_configs.json")
        assert r["score"] == float(np.mean([row["success"] for row in r["rows"]]))
        assert len(r["rows"]) == 100
        validation_files[str(path)] = digest(path)
    expected = sum(
        3 * (1 + (r["primary_selected"]["name"] != r["final_selected"]["name"]))
        for r in selected["tasks"].values()
        if r["primary_selected"] and r["final_selected"]
    )
    assert len(validation_files) == expected
    # Replay checkpoint inference on NEW software-test seeds, never validation seeds.
    replay = []
    for task, entry in selected["tasks"].items():
        chosen = entry["final_selected"]
        if chosen is None:
            continue
        r = results[f"runs/v14/{chosen['name']}-s0/result.json"]
        c = r["config"]
        with Path(r["checkpoint_path"]).open("rb") as stream:
            state = pickle.load(stream)
        source = make_source(c, 3)
        model = build_model(c, source.width)
        seeds = [seed_for(task, "software", index=500000 + i) for i in range(3)]
        a = evaluate_policy(model, state["params"], source, task, seeds)
        b = evaluate_policy(model, state["params"], make_source(c, 3), task, seeds)
        assert a == b
        replay.append(
            {"task": task, "checkpoint": r["checkpoint_path"], "software_seed_replay": "MATCH"}
        )
        guard.check()
    # Independently repeat a primary development training trajectory, without rescoring it.
    r = results.get("runs/v14/catch-primary-linear-s0/result.json")
    if r:
        trainer = Trainer(r["config"])
        while not trainer.done():
            guard.check()
            trainer.train_step()
        assert params_hash(trainer.model.params) == r["final_policy_sha256"]
        skip = {"wall_seconds", "transitions_per_second"}
        assert [{k: v for k, v in row.items() if k not in skip} for row in trainer.history] == [
            {k: v for k, v in row.items() if k not in skip} for row in r["training_history"]
        ]
    assert max(peaks) < 2048 and max(s["elapsed_seconds"] for s in segments) < 120
    assert file_hash(plan["graph_file"]) == plan["graph_file_sha256"]
    graph = load_topology("biological")
    assert graph.n == 2040
    report = {
        "status": "PASS",
        "historical_files_preserved": len(protected),
        "primary_trials": sum(r["phase"] == "primary" for r in results.values()),
        "total_completed_development_trials": len(results),
        "configurations_per_task": configuration_count,
        "validation_evaluations": len(validation_files),
        "validation_files": validation_files,
        "checkpoint_replays": replay,
        "replay_seed_scope": "fresh software tests only; no validation/historical rescoring",
        "training_replay": (
            "Catch Linear seed0 full65536 transitions, parameters and numerical history MATCH"
        ),
        "confirmatory_testing": False,
        "seed_namespace_disjoint": True,
        "fixed_substrate": True,
        "source_graph_sha256": plan["graph_file_sha256"],
        "plan_sha256": digest(planpath),
        "selection_sha256": digest(ROOT / "selected_configs.json"),
        "summary_sha256": digest(ROOT / "development_summary.json"),
        "code_sha256": code_hash(),
        "max_trial_peak_memory_mb": max(peaks),
        "max_process_seconds": max(s["elapsed_seconds"] for s in segments),
        "resources": guard.check(),
    }
    (ROOT / "curriculum_exposure.json").write_text(json.dumps(curriculum, indent=2) + "\n")
    (ROOT / "scientific_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("validation_files", "checkpoint_replays")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
