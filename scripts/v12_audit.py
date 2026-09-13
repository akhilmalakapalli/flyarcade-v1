"""Read-only scientific verification plus representative checkpoint replay."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v12.study import (
    array_hash,
    build,
    code_hash,
    episode,
    evaluate,
    graph_hash,
    restore_checkpoint,
    seed_for,
)

ROOT = Path("artifacts/v12")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    guard = ResourceGuard()
    planpath = Path("experiments/v12_run_plan.json")
    plan = json.loads(planpath.read_text())
    summary = json.loads((ROOT / "results_summary.json").read_text())
    assert plan["code_sha256"] == code_hash()
    assert plan["source_graph_sha256"] == digest("data/malecns-v1.0/graph.npz")
    historical = json.loads((ROOT / "historical_hashes.json").read_text())
    for path, expected in historical.items():
        assert digest(path) == expected, f"historical file modified: {path}"
    for path, expected in {**plan["standardizer_hashes"], **plan["development_hashes"]}.items():
        assert digest(path) == expected, path
    archive = json.loads(gzip.decompress((ROOT / "results_archive.json.gz").read_bytes()))
    assert len(archive) == 54
    for path, record in archive.items():
        assert json.loads(Path(path).read_text()) == record
    for task in plan["tasks"]:
        representation = json.loads((ROOT / f"representation-{task}.json").read_text())
        assert representation["post_hoc"] and len(representation["rows"]) == 6
        assert len({r["state_sha256"] for r in representation["rows"]}) == 1
        for row in representation["rows"]:
            assert row["episode_seeds"] == [
                seed_for(task, "representation", index=i) for i in range(20)
            ]
    assert json.loads((ROOT / "historical_replay.json").read_text())["status"] == "PASS"
    assert json.loads((ROOT / "core_equivalence.json").read_text())["status"] == "PASS"
    graph = load_graph("data/malecns-v1.0/graph.npz")
    graphs = {"biological": graph}
    controls = {}
    for seed in range(3):
        control = load_graph(f"data/v12/rewired-{seed}.npz")
        assert np.array_equal(graph.neuron_ids, control.neuron_ids)
        assert graph.e == control.e
        for endpoint in ("pre", "post"):
            assert np.array_equal(
                np.bincount(getattr(graph, endpoint), minlength=graph.n),
                np.bincount(getattr(control, endpoint), minlength=graph.n),
            )
        assert np.array_equal(np.sort(graph.counts), np.sort(control.counts))
        assert np.array_equal(
            np.bincount(graph.pre, weights=graph.counts, minlength=graph.n),
            np.bincount(control.pre, weights=control.counts, minlength=graph.n),
        )
        controls[str(seed)] = {
            "graph_sha256": graph_hash(control),
            "degree_and_outgoing_strength": True,
        }
        graphs[f"rewired-{seed}"] = control
    all_blocks = []
    checked = []
    replay = []
    resources = []
    for task in plan["tasks"]:
        for purpose, count in [
            ("fit", 24),
            ("dev_train", 300),
            ("dev_eval", 20),
            ("train", plan["tasks"][task]["training_episodes"]),
            ("eval", 40),
            ("representation", 20),
            ("probe", 1000),
        ]:
            block = set(seed_for(task, purpose, seed, i) for seed in range(3) for i in range(count))
            assert min(block) > 10_000_000
            assert not any(block & previous for previous in all_blocks)
            all_blocks.append(block)
        for topology in graphs:
            path = ROOT / f"{task}-{topology}-standardizer.meta.json"
            meta = json.loads(path.read_text())
            assert meta["seeds"] == plan["tasks"][task]["feature_fitting_seeds"]
            assert meta["purpose"] == "development_feature_fit"
            assert meta["graph_sha256"] == graph_hash(graphs[topology])
        for condition in plan["conditions"]:
            for seed in plan["training_seeds"]:
                path = Path("runs") / f"v12-{task}-{condition}-{seed}" / "result.json"
                result = json.loads(path.read_text())
                checkpoint = path.with_name("checkpoint.npz")
                assert digest(path) == summary["result_files"][str(path)]
                assert digest(checkpoint) == result["checkpoint_sha256"]
                assert result["plan_sha256"] == digest(planpath)
                assert result["code_sha256"] == code_hash()
                assert result["source_graph_sha256"] == plan["source_graph_sha256"]
                topology = f"rewired-{seed}" if condition == "rewired" else "biological"
                assert result["graph_sha256"] == graph_hash(graphs[topology])
                assert result["standardizer_sha256"] == digest(
                    ROOT / f"{task}-{topology}-standardizer.json"
                )
                with np.load(checkpoint, allow_pickle=False) as arrays:
                    assert array_hash(arrays["magnitude"]) == result["final_core_sha256"]
                    for key in ("actor", "critic"):
                        assert array_hash(arrays[key]) == result[f"final_{key}_sha256"]
                assert result["status"] == "COMPLETE"
                assert result["initial_core_sha256"] == result["final_core_sha256"]
                if condition == "frozen":
                    for key in ("actor", "critic"):
                        assert result[f"initial_{key}_sha256"] == result[f"final_{key}_sha256"]
                    assert result["before"] == result["evaluations"]["after"]
                else:
                    assert result["initial_actor_sha256"] != result["final_actor_sha256"]
                n = plan["tasks"][task]["training_episodes"]
                assert result["completed_episodes"] == len(result["training"]) == n
                assert result["training_seeds"] == [
                    seed_for(task, "train", seed, i) for i in range(n)
                ]
                assert result["evaluation_seeds"] == [
                    seed_for(task, "eval", seed, i) for i in range(40)
                ]
                assert len(result["before"]) == 40
                for rows in result["evaluations"].values():
                    assert len(rows) == 40
                    assert all(
                        0 <= row["success"] <= 1 and np.isfinite(row["return"]) for row in rows
                    )
                resources.extend(result["segments"])
                checked.append(str(path))
        path = Path("runs") / f"v12-{task}-biological-0" / "result.json"
        result = json.loads(path.read_text())
        std = Standardizer.load(ROOT / f"{task}-biological-standardizer.json")
        controller = build(graph, task, 0, std, plan["tasks"][task]["hyperparameters"])
        restore_checkpoint(
            path.with_name("checkpoint.npz"),
            controller,
            {"plan_sha256": digest(planpath), "code_sha256": code_hash()},
        )
        for key in ("actor", "critic"):
            assert array_hash(getattr(controller.motor, key)) == result[f"final_{key}_sha256"]
        rows = evaluate(controller, task, result["evaluation_seeds"], guard=guard)
        assert rows == result["evaluations"]["after"], f"{task}: checkpoint mismatch"
        mask_rng = np.random.default_rng(seed_for(task, "perturb", 0))
        edge_mask = mask_rng.random(graph.e) >= 0.1
        neuron_mask = mask_rng.random(graph.n) >= 0.1
        for key, kwargs in [
            ("sensory_noise", {"noise": 0.1}),
            ("edge_ablation", {"edge_mask": edge_mask}),
            ("neuron_ablation", {"neuron_mask": neuron_mask}),
        ]:
            assert (
                evaluate(controller, task, result["evaluation_seeds"], guard=guard, **kwargs)
                == result["evaluations"][key]
            )
        previous = digest(path)
        subprocess.run(
            [
                sys.executable,
                "scripts/v12_trial.py",
                "--task",
                task,
                "--condition",
                "biological",
                "--seed",
                "0",
            ],
            check=True,
            timeout=10,
        )
        assert digest(path) == previous
        replay.append(
            {
                "task": task,
                "clean_and_three_perturbations": "exact match",
                "completed_rerun_preserved": True,
            }
        )
    # A fresh full-budget repetition of the inexpensive failed Flappy trial.
    result = json.loads(Path("runs/v12-flappy-biological-0/result.json").read_text())
    controller = build(
        graph,
        "flappy",
        0,
        Standardizer.load(ROOT / "flappy-biological-standardizer.json"),
        plan["tasks"]["flappy"]["hyperparameters"],
    )
    training = [
        episode(controller, "flappy", seed, training=True, guard=guard)
        for seed in result["training_seeds"]
    ]
    assert training == result["training"]
    assert array_hash(controller.motor.actor) == result["final_actor_sha256"]
    assert array_hash(controller.motor.critic) == result["final_critic_sha256"]
    for resource in resources:
        assert resource["elapsed_seconds"] < 120 and resource["rss_mb"] < 2048
    report = {
        "status": "PASS",
        "summary_sha256": digest(ROOT / "results_summary.json"),
        "analysis_files": {
            str(p): digest(p)
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
            and (
                p.suffix in (".csv", ".png", ".svg")
                or p.name in ("representations.json", "table.md")
            )
        },
        "trials": len(checked),
        "expected_trials": 36,
        "historical_files_preserved": len(historical),
        "source_graph_sha256": plan["source_graph_sha256"],
        "plan_sha256": digest(planpath),
        "code_sha256": code_hash(),
        "checkpoint_replays": replay,
        "fresh_full_budget_rerun": "flappy biological seed 0: full history, actor and critic exact",
        "controls": controls,
        "seeds_disjoint": True,
        "frozen_and_stage_a_invariants": True,
        "max_segment_seconds": max(r["elapsed_seconds"] for r in resources),
        "max_rss_snapshot_mb": max(r["rss_mb"] for r in resources),
        "result_files": {p: digest(p) for p in checked},
        "resources": guard.check(),
    }
    assert len(checked) == 36
    (ROOT / "scientific_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("result_files", "controls")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
