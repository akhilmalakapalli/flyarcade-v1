"""Evaluate each locked v1.5 selection ONCE on the fresh validation block. No training.

Refuses to run unless artifacts/v15/selected_configs.json is committed and unmodified.
Existing validation outputs are preserved, never recomputed.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import v15_path  # noqa: F401
from flyarcade_v15.provenance import code_hash
from flyarcade_v15.seeds import seed_for
from flyarcade_v15.trainer import (
    digest,
    evaluate,
    evaluate_baseline,
    params_hash,
    state_probe,
    summarize_rows,
)

from flyarcade.resources import ResourceGuard

ROOT = Path("artifacts/v15")
SELECTION = ROOT / "selected_configs.json"
EPISODES = 100


def committed(path):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)])
    return tracked.returncode == 0 and clean.returncode == 0


def load_policy(result):
    with np.load(result["policy_path"]) as data:
        params = {k: data[k].copy() for k in data.files}
    assert params_hash(params) == result["policy_sha256"], "policy hash mismatch"
    return params


def validate_one(task, role, record, seed):
    name = f"{record['name']}-s{seed}"
    out = ROOT / "validation" / f"{name}.json"
    source = Path("runs/v15") / name / "result.json"
    result = json.loads(source.read_text())
    identity = {
        "selection_sha256": digest(SELECTION),
        "source_result_sha256": digest(source),
        "policy_sha256": result["policy_sha256"],
        "code_sha256": code_hash(),
        "task": task,
        "role": role,
        "configuration": record["name"],
        "development_seed": seed,
    }
    if out.exists():
        existing = json.loads(out.read_text())
        assert all(existing[k] == v for k, v in identity.items())
        return existing
    guard = ResourceGuard()
    params = load_policy(result)
    config = result["config"]
    seeds = [seed_for(task, "validation", seed, i) for i in range(EPISODES)]
    rows = evaluate(config, params, seeds)
    guard.check()
    summary = summarize_rows(rows)
    payload = {
        **identity,
        "purpose": "validation",
        "learning": False,
        "episodes": EPISODES,
        "evaluation_seeds": seeds,
        "score": summary["score"],
        "action_distribution": summary["action_distribution"],
        "dominant_action_fraction": summary["dominant_action_fraction"],
        "policy_entropy": summary["policy_entropy"],
        "state_probe": state_probe(config, params, "validation_probe", seed),
        "random_score": float(
            np.mean([r["success"] for r in evaluate_baseline(task, seeds, "random")])
        ),
        "reference_score": float(
            np.mean([r["success"] for r in evaluate_baseline(task, seeds, "reference")])
        ),
        "training_transitions": result["training_transitions"],
        "imitation_transitions": (result.get("imitation") or {}).get("transitions", 0),
        "wall_clock_training_seconds": result["wall_clock_training_seconds"],
        "rows": rows,
        "status": "COMPLETE",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(name, role, round(payload["score"], 4), flush=True)
    return payload


def main():
    if not committed(SELECTION):
        sys.exit("selected_configs.json must be committed and unmodified before validation")
    selected = json.loads(SELECTION.read_text())
    assert selected["code_sha256"] == code_hash()
    output = ROOT / "validation_summary.json"
    if output.exists():
        print("validation already complete; preserved, not recomputed")
        return
    manifest = ROOT / "validation_manifest.json"
    meta = {
        "selection_sha256": digest(SELECTION),
        "selection_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "validation_script_sha256": digest(__file__),
        "code_sha256": code_hash(),
        "confirmatory_testing": False,
    }
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for key in ("selection_sha256", "validation_script_sha256", "code_sha256"):
            assert old[key] == meta[key]
    else:
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
    summary = {"selection_sha256": digest(SELECTION), "tasks": {}, "confirmatory_testing": False}
    for task, entry in selected["tasks"].items():
        summary["tasks"][task] = {}
        for role in ("headline_selected", "rescue_selected"):
            record = entry[role]
            if record is None:
                continue
            rows = [validate_one(task, role, record, seed) for seed in (0, 1, 2)]
            values = [r["score"] for r in rows]
            summary["tasks"][task][role] = {
                "configuration": record["name"],
                "config": record["config"],
                "seed_values": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "variance": float(np.var(values)),
                "state_probes": [r["state_probe"]["score"] for r in rows],
                "dominant_action_fractions": [r["dominant_action_fraction"] for r in rows],
                "policy_entropies": [r["policy_entropy"] for r in rows],
                "random_mean": float(np.mean([r["random_score"] for r in rows])),
                "reference_mean": float(np.mean([r["reference_score"] for r in rows])),
                "training_transitions": rows[0]["training_transitions"],
                "imitation_transitions": rows[0]["imitation_transitions"],
                "mean_training_seconds": float(
                    np.mean([r["wall_clock_training_seconds"] for r in rows])
                ),
                "status": "COMPLETE",
            }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("v1.5 fresh validation complete. STOP: no confirmatory testing.", flush=True)


if __name__ == "__main__":
    main()
