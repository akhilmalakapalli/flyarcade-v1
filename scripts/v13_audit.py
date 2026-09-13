"""M13: v1.3 scientific audit. Emits artifacts/v13/scientific_audit.json.

Status is PASS only if every invariant holds; any failed check is recorded with its
reason and the status becomes FAIL. Nothing here trains or tunes.
"""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from v13_external import confirmatory_tasks, load_registry

from flyarcade.connectome import load_graph
from flyarcade_v13.controller import FlyFeatures
from flyarcade_v13.environments import TARGETS
from flyarcade_v13.gates import task_gate
from flyarcade_v13.study import (
    CONFIRMATORY_PURPOSES,
    DEVELOPMENT_PURPOSES,
    Trainer,
    code_hash,
    digest,
    graph_hash,
    load_topology,
    params_hash,
    purpose_range,
)

ROOT = Path("artifacts/v13")
PLAN = Path("experiments/v13_run_plan.json")
CONDITIONS = ("biological", "frozen", "rewired", "sensory")


class Audit:
    def __init__(self):
        self.checks = {}

    def check(self, name, condition, detail=None):
        self.checks[name] = {"ok": bool(condition), **({"detail": detail} if detail is not None else {})}
        print(f"{'ok  ' if condition else 'FAIL'} {name}{'' if detail is None else ': ' + str(detail)[:160]}", flush=True)
        return condition


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    a = Audit()
    env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    import os

    os.environ.update(env)
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    a.check("branch", branch == "flyarcade-v1.3-multitask", branch)
    a.check("descends_from_v12", run(["git", "merge-base", "--is-ancestor", "00861e0e5cb7e08584836395c41f2df1d5234da5", "HEAD"]).returncode == 0)
    plan = json.loads(PLAN.read_text())
    plan_sha = digest(PLAN)
    a.check("code_hash_matches_frozen_plan", plan["code_sha256"] == code_hash())
    a.check("graph_source_hash", plan["source_graph_sha256"] == digest("data/malecns-v1.0/graph.npz"))
    for topology, expected in plan["graph_sha256"].items():
        a.check(f"graph_hash_{topology}", graph_hash(load_topology(topology)) == expected)
    for path, expected in plan["standardizer_sha256"].items():
        a.check(f"standardizer {Path(path).name}", digest(path) == expected)
    a.check("environment_source_unchanged", digest("src/flyarcade/v12/environments.py") == plan["environment_source_sha256"])

    snapshot = json.loads((ROOT / "historical_hashes.json").read_text())
    modified = [p for p, h in snapshot["files"].items() if not Path(p).exists() or digest(p) != h]
    a.check("protected_historical_files", not modified, f"{len(snapshot['files'])} files; modified={modified[:5]}")
    a.check("v12_audit_code_hash_still_valid", json.loads(Path("experiments/v12_run_plan.json").read_text())["code_sha256"] == __import__("flyarcade.v12.study", fromlist=["code_hash"]).code_hash())

    # seeds
    blocks = []
    for task in TARGETS:
        for purpose in DEVELOPMENT_PURPOSES + CONFIRMATORY_PURPOSES:
            lo, hi = purpose_range(task, purpose)
            blocks.append((lo, hi))
    blocks.sort()
    a.check("v13_seed_blocks_disjoint", all(h <= l2 for (_, h), (l2, _) in zip(blocks, blocks[1:], strict=False)))
    a.check("v13_seeds_above_historical", blocks[0][0] >= 200_000_000)
    dev_log = [json.loads(line) for line in (ROOT / "development" / "log.jsonl").read_text().splitlines() if line]
    dev_purposes = {e["config"]["train_purpose"] for e in dev_log} | {e["config"]["eval_purpose"] for e in dev_log}
    a.check("development_used_only_development_purposes", dev_purposes <= set(DEVELOPMENT_PURPOSES), sorted(dev_purposes))

    # Stage A core unchanged
    graph = load_graph("data/malecns-v1.0/graph.npz")
    from flyarcade.v12.study import build as v12build

    ref = v12build(graph, "pong", 0, None)
    fly = FlyFeatures(graph, 7, 1, ticks=4)
    a.check("stage_a_core_magnitudes_identical", np.array_equal(ref.core.magnitude, fly.base_magnitude) and np.array_equal(ref.core.base, fly.base_magnitude))
    ref.rng = np.random.default_rng(12345 + 1_000_000_000)
    fly.reset(0, 12345)
    obs = np.full(7, 0.4)
    a.check("stage_a_core_rates_bit_identical_authentic_graph", all(np.array_equal(fly.raw_rates(obs[None], [0])[0], ref.rates(obs)) for _ in range(20)))

    registry = load_registry()  # raises if any externally frozen file changed
    a.check("externally_frozen_task_files_unchanged", True, sorted(registry["tasks"]))
    study_tasks = confirmatory_tasks(TARGETS)

    # trials
    results = {}
    files = {}
    resources = []
    for task in study_tasks:
        for condition in CONDITIONS:
            for seed in plan["training_seeds"]:
                name = f"v13-{task}-{condition}-{seed}"
                path = Path("runs") / name / "result.json"
                if not a.check(f"trial_exists {name}", path.exists()):
                    continue
                r = json.loads(path.read_text())
                results[(task, condition, seed)] = r
                files[str(path)] = digest(path)
                resources.extend(r["segments"])
                ok = (
                    r["status"] == "COMPLETE"
                    and r["code_sha256"] == plan["code_sha256"]
                    and r["finite"]
                    and digest(Path("runs") / name / "policy.npz") == r["policy_npz_sha256"]
                    and r["config"]["train_purpose"] == "conf_train"
                    and r["evaluation"]["purpose"] == "conf_eval"
                    and len(r["evaluation_seeds"]) == plan["evaluation_episodes"]
                    and all(len(v) == plan["evaluation_episodes"] for v in r["evaluations"].values())
                    and r["transitions"] >= r["config"]["transitions"]
                )
                with np.load(Path("runs") / name / "policy.npz", allow_pickle=False) as data:
                    ok &= all(np.isfinite(data[k]).all() for k in data.files)
                if condition == "frozen":
                    ok &= r["initial_params_sha256"] == r["final_params_sha256"]
                    ok &= r["evaluations"]["before"] == r["evaluations"]["after"]
                else:
                    ok &= r["initial_params_sha256"] != r["final_params_sha256"]
                a.check(f"trial_invariants {name}", ok)
    expected = len(study_tasks) * len(CONDITIONS) * len(plan["training_seeds"])
    a.check("expected_trial_count", len(results) == expected, f"{len(results)}/{expected}")
    a.check("resource_guards", all(s["elapsed_seconds"] < 120 and s["rss_mb"] < 2048 for s in resources), {"max_seconds": max(s["elapsed_seconds"] for s in resources), "max_rss_mb": max(s["rss_mb"] for s in resources)})

    # replays: every task seed 0 biological with perturbations; one rewired, frozen, sensory
    replays = []
    for task in study_tasks:
        for condition, extra in (("biological", ["--perturbations", "--baselines"]), ("sensory", []), ("rewired", []), ("frozen", [])):
            out = run([sys.executable, "scripts/v13_replay.py", "--run", f"runs/v13-{task}-{condition}-0", *extra])
            ok = out.returncode == 0
            replays.append({"task": task, "condition": condition, "output": out.stdout.strip()[-400:]})
            a.check(f"replay {task}/{condition}/0", ok, out.stdout.strip()[-200:] or out.stderr[-200:])

    # completed trial not rewritten
    probe_task = study_tasks[0]
    before = digest(f"runs/v13-{probe_task}-biological-0/result.json")
    spec = f"artifacts/v13/confirmatory/specs/v13-{probe_task}-biological-0.json"
    run([sys.executable, "scripts/v13_trial.py", "--spec", spec])
    a.check("completed_result_preserved_on_rerun", digest(f"runs/v13-{probe_task}-biological-0/result.json") == before)

    # fresh full-budget deterministic rerun (cheapest learned fly trial)
    rerun = ROOT / "audit_rerun"
    cheapest = min(
        ((k, r) for k, r in results.items() if k[1] == "biological"),
        key=lambda kv: kv[1]["train_wall_seconds"],
    )
    (task, _, seed), original = cheapest
    spec_data = json.loads(Path(f"artifacts/v13/confirmatory/specs/v13-{task}-biological-{seed}.json").read_text())
    spec_data["output"] = str(Path("runs") / "v13-audit-rerun")
    spec_data["name"] = original["name"]
    Path("runs/v13-audit-rerun").mkdir(exist_ok=True)
    rerun.mkdir(parents=True, exist_ok=True)
    spec_path = rerun / "spec.json"
    spec_path.write_text(json.dumps(spec_data, indent=1))
    result_path = Path("runs/v13-audit-rerun/result.json")
    for _ in range(400):
        if result_path.exists():
            break
        code = subprocess.run([sys.executable, "scripts/v13_trial.py", "--spec", str(spec_path)], env={**os.environ}).returncode
        if code not in (0, 7):
            break
    fresh = json.loads(result_path.read_text()) if result_path.exists() else {}
    strip = lambda rows: [{k: v for k, v in h.items() if k not in ("wall_seconds", "transitions_per_second")} for h in rows]  # noqa: E731
    a.check(
        "fresh_full_budget_rerun_exact",
        fresh
        and fresh["final_params_sha256"] == original["final_params_sha256"]
        and fresh["evaluations"] == original["evaluations"]
        and strip(fresh["history"]) == strip(original["history"])
        and fresh["state_probe"] == original["state_probe"],
        f"{task} biological seed {seed}",
    )

    # tables regenerate from raw results
    tracked = ["primary.csv", "contrasts.csv", "perturbations.csv", "task_specific.csv", "representations.csv", "results_summary.json"]
    before_hashes = {t: digest(ROOT / t) for t in tracked}
    out = run([sys.executable, "scripts/v13_report.py"])
    a.check("report_regenerates", out.returncode == 0, out.stderr[-300:])
    after_hashes = {t: digest(ROOT / t) for t in tracked}
    a.check("tables_identical_after_regeneration", before_hashes == after_hashes, [t for t in tracked if before_hashes[t] != after_hashes[t]])
    summary = json.loads((ROOT / "results_summary.json").read_text())
    a.check("summary_references_all_result_files", summary["result_files"] == files)
    a.check("summary_plan_hash", summary["plan_sha256"] == plan_sha)
    figures = sorted((ROOT / "figures").glob("*.png"))
    a.check("figures_present", len(figures) >= 6, [f.name for f in figures])

    gates = {task: task_gate([results[(task, "biological", s)] for s in plan["training_seeds"]]) for task in study_tasks}
    a.check("gates_match_summary", all(gates[t] == summary["tasks"][t]["criteria"] for t in study_tasks))

    manuscript = Path("paper/manuscript.md")
    if manuscript.exists() and (ROOT / "manuscript_numbers.json").exists():
        numbers = json.loads((ROOT / "manuscript_numbers.json").read_text())
        text = manuscript.read_text()
        a.check("manuscript_numbers_traceable", all(v in text for v in numbers["strings"]), numbers.get("source"))
    a.check("no_credentials_in_artifacts", not any("token" in p.name.lower() for p in ROOT.rglob("*")))
    archive = gzip.compress(json.dumps({k: files[k] for k in sorted(files)}).encode())
    status = "PASS" if all(c["ok"] for c in a.checks.values()) else "FAIL"
    report = {
        "status": status,
        "plan_sha256": plan_sha,
        "code_sha256": code_hash(),
        "graph_sha256": plan["graph_sha256"],
        "trials": len(results),
        "expected_trials": expected,
        "confirmatory_gates": {t: g["passed"] for t, g in gates.items()},
        "replays": replays,
        "result_files": files,
        "result_index_sha256": hashlib.sha256(archive).hexdigest(),
        "checks": a.checks,
        "failed": [k for k, c in a.checks.items() if not c["ok"]],
    }
    (ROOT / "scientific_audit.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"status": status, "failed": report["failed"]}, indent=1))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
