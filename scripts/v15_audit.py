"""Audit v1.5: provenance, seed separation, selection reproducibility, validation-once,
fixed substrate, and preservation of every historical (v1.1-v1.4) artifact."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import v15_path  # noqa: F401
from flyarcade_v15.provenance import code_hash
from flyarcade_v15.seeds import PURPOSES, seed_for
from flyarcade_v15.selection import select, summarize
from flyarcade_v15.trainer import digest, params_hash

from flyarcade_v14 import study as v14

ROOT = Path("artifacts/v15")
PLAN = Path("experiments/v15/development_plan.json")
README_SHA256 = "a46572881f00b6963a0bbddcf2cc439d7acfd1ca46158711ad1a6fe8caaa7769"
V14_COMMIT = "73ea7a1"


def git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def main():
    plan = json.loads(PLAN.read_text())
    checks = {}
    assert git("branch", "--show-current").stdout.strip() == "flyarcade-v1.5-performance"
    # 1. plan and code provenance
    assert plan["code_sha256"] == code_hash(), "code changed after plan"
    assert plan["suite_sha256"] == digest("scripts/v15_suite.py")
    assert plan["trial_sha256"] == digest("scripts/v15_trial.py")
    assert plan["runner_sha256"] == digest("scripts/v15_run.py")
    plan_commit = git("log", "--format=%H", "--diff-filter=A", "--", str(PLAN)).stdout.split()[-1]
    first_spec = git(
        "log", "--format=%H", "--diff-filter=A", "--", "experiments/v15/specs/development"
    ).stdout.split()
    if first_spec:
        assert (
            git("merge-base", "--is-ancestor", plan_commit, first_spec[-1], check=False).returncode
            == 0
        )
    checks["plan_commit"] = plan_commit
    # 2. historical preservation
    assert git("diff", "--quiet", V14_COMMIT, "HEAD", "--", "src", check=False).returncode == 0
    changed = git("diff", "--name-only", V14_COMMIT, "HEAD").stdout.split()
    allowed = (
        "artifacts/v15/",
        "experiments/v15/",
        "src_v15/",
        "scripts/v15_",
        "tests/test_v15.py",
    )
    allowed_files = {"STATE.md", "HANDOFF.md", "CHANGELOG.md", "RUNBOOK.md"}
    unexpected = [p for p in changed if not p.startswith(allowed) and p not in allowed_files]
    assert not unexpected, f"non-v1.5 files changed since v1.4: {unexpected}"
    protected = json.loads(Path("artifacts/v14/historical_hashes.json").read_text())
    for path, expected in protected.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(Path("README.md").read_bytes()).hexdigest() == README_SHA256
    selected_v14 = json.loads(Path("artifacts/v14/selected_configs.json").read_text())
    assert selected_v14["code_sha256"] == v14.code_hash(), "v1.4 provenance no longer reproducible"
    checks["historical_files_preserved"] = len(protected)
    # 3. development runs
    selected = json.loads((ROOT / "selected_configs.json").read_text())
    assert selected["plan_sha256"] == digest(PLAN) and selected["code_sha256"] == code_hash()
    core_hashes, peaks, seconds, n_runs = set(), [], [], 0
    for task, entry in selected["tasks"].items():
        records = entry["configurations"]
        assert len(records) <= plan["max_development_configurations"][task]
        rebuilt = []
        for rec in records:
            results = []
            for spec_path in rec["specs"]:
                spec = json.loads(Path(spec_path).read_text())
                assert spec["plan_sha256"] == digest(PLAN) and spec["code_sha256"] == code_hash()
                run = Path("runs/v15") / spec["name"]
                if not (run / "result.json").exists():
                    results.append(
                        {**json.loads((run / "failure.json").read_text()), "status": "FAILED"}
                    )
                    continue
                r = json.loads((run / "result.json").read_text())
                n_runs += 1
                assert r["spec_sha256"] == digest(spec_path)
                assert r["config"]["train_purpose"] == "train"
                s = r["config"]["seed"]
                assert r["dev_eval_seeds"] == [seed_for(task, "dev_eval", s, i) for i in range(100)]
                assert digest(r["checkpoint_path"]) == r["checkpoint_sha256"]
                with np.load(r["policy_path"]) as data:
                    assert params_hash({k: data[k] for k in data.files}) == r["policy_sha256"]
                if r["fixed_core_sha256"]:
                    core_hashes.add(r["fixed_core_sha256"])
                assert r["finite"] and r["status"] == "COMPLETE"
                assert (r.get("imitation") is not None) == (spec["role"] == "rescue")
                peaks.append(r["peak_memory_mb"])
                seconds.extend(seg["elapsed_seconds"] for seg in r["segments"])
                results.append(r)
            again = summarize(rec["name"], rec["stage"], rec["config"], results, role=rec["role"])
            for key in ("mean", "sd", "utility", "eligible", "scores"):
                assert again[key] == rec[key], (rec["name"], key)
            rebuilt.append(again)
        for role, key in (("headline", "headline_selected"), ("rescue", "rescue_selected")):
            best = select(rebuilt, role)
            if key == "headline_selected":
                assert (best["name"] if best else None) == (
                    entry[key]["name"] if entry[key] else None
                )
        for rec in records:
            assert rec["role"] != "control" or not rec["eligible"]
        if entry["headline_selected"]:
            assert entry["headline_selected"]["role"] == "headline"
            readout = entry["headline_selected"]["config"]["readout"]
            assert readout in ("descending", "nonvisual", "cb_intrinsic", "pooled"), (
                "leaky headline"
            )
    assert len(core_hashes) == 1, "fixed substrate differs between runs"
    assert max(peaks) < 2048 and max(seconds) < 120
    checks.update(
        development_runs=n_runs,
        fixed_core_sha256=core_hashes.pop(),
        max_peak_memory_mb=max(peaks),
        max_process_seconds=max(seconds),
    )
    # 4. seed namespaces
    intervals = []
    for t in plan["tasks"]:
        for p in PURPOSES:
            lo = seed_for(t, p)
            hi = lo + 5_000_000_000 - 1
            assert all(hi < a or lo > b for a, b in intervals)
            intervals.append((lo, hi))
    v14_max = v14.seed_for("flappy", "validation_probe", 2, v14.SEED_SLOT - 1) + 2_000_000_000
    assert min(a for a, _ in intervals) > v14_max
    checks["seed_namespaces_disjoint"] = True
    # 5. validation once, after the committed lock
    summary = json.loads((ROOT / "validation_summary.json").read_text())
    manifest = json.loads((ROOT / "validation_manifest.json").read_text())
    assert summary["selection_sha256"] == digest(ROOT / "selected_configs.json")
    lock_commit = manifest["selection_commit"]
    shown = git("show", f"{lock_commit}:artifacts/v15/selected_configs.json").stdout
    assert hashlib.sha256(shown.encode()).hexdigest() == manifest["selection_sha256"]
    assert (
        git("log", "--format=%H", lock_commit, "--", "artifacts/v15/validation_summary.json").stdout
        == ""
    )
    files = sorted((ROOT / "validation").glob("*.json"))
    expected = sum(
        3
        for e in selected["tasks"].values()
        for k in ("headline_selected", "rescue_selected")
        if e[k]
    )
    assert len(files) == expected
    for f in files:
        v = json.loads(f.read_text())
        s = v["development_seed"]
        assert v["evaluation_seeds"] == [
            seed_for(v["task"], "validation", s, i) for i in range(100)
        ]
        assert not v["learning"] and len(v["rows"]) == 100
        assert v["score"] == float(np.mean([row["success"] for row in v["rows"]]))
    checks.update(validation_files=len(files), selection_lock_commit=lock_commit)
    checks["confirmatory_testing"] = False
    checks["status"] = "PASS"
    (ROOT / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
