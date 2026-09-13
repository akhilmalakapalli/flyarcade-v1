"""Evaluate locked development selections once on fresh validation; no training."""

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from v14_trial import peak_mb

from flyarcade.resources import ResourceGuard
from flyarcade_v14.study import (
    build_model,
    code_hash,
    digest,
    evaluate_baseline,
    evaluate_policy,
    make_source,
    mean_success,
    params_hash,
    seed_for,
    state_probe,
)

ROOT = Path("artifacts/v14")


def run_one(selection, task, record, kind, seed):
    guard = ResourceGuard()
    source_name = f"{record['name']}-s{seed}"
    source_path = Path("runs/v14") / source_name / "result.json"
    source_result = json.loads(source_path.read_text())
    assert source_result["code_sha256"] == code_hash()
    checkpoint = Path(source_result["checkpoint_path"])
    assert digest(checkpoint) == source_result["checkpoint_sha256"]
    purpose = "primary_validation" if kind == "primary" else "final_validation"
    output = ROOT / "validation" / f"{source_name}-{purpose}.json"
    output.parent.mkdir(exist_ok=True)
    identity = {
        "selection_sha256": digest(selection),
        "source_result_sha256": digest(source_path),
        "checkpoint_sha256": digest(checkpoint),
        "code_sha256": code_hash(),
        "purpose": purpose,
        "task": task,
        "configuration": record["name"],
        "development_seed": seed,
    }
    if output.exists():
        existing = json.loads(output.read_text())
        assert all(existing[k] == v for k, v in identity.items())
        print(output.name, "preserved", flush=True)
        return existing
    config = source_result["config"]
    source = make_source(config, 16)
    model = build_model(config, source.width)
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    params = state["params"]
    assert params_hash(params) == source_result["final_policy_sha256"]
    seeds = [seed_for(task, purpose, seed, i) for i in range(100)]
    rows = evaluate_policy(model, params, source, task, seeds)
    probe = state_probe(model, params, source, task, "validation_probe", seed)
    random = evaluate_baseline(task, seeds, "random")
    counts = np.sum([r["action_counts"] for r in rows], axis=0)
    assert params_hash(params) == source_result["final_policy_sha256"]
    result = {
        **identity,
        "score": mean_success(rows),
        "rows": rows,
        "random_rows": random,
        "state_probe": probe,
        "action_distribution": (counts / counts.sum()).tolist(),
        "dominant_action_fraction": float(counts.max() / counts.sum()),
        "policy_entropy": float(np.mean([r["policy_entropy"] for r in rows])),
        "evaluation_seeds": seeds,
        "learning": False,
        "status": "COMPLETE",
        "resources": guard.check(),
        "peak_memory_mb": peak_mb(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(source_name, purpose, result["score"], flush=True)
    return result


def main():
    selection = ROOT / "selected_configs.json"
    selected = json.loads(selection.read_text())
    assert selected["code_sha256"] == code_hash()
    output = ROOT / "validation_summary.json"
    if output.exists():
        existing = json.loads(output.read_text())
        assert existing["selection_sha256"] == digest(selection)
        print("Fresh validation already complete; preserved, no rescoring.")
        return
    metadata = {
        "selection_sha256": digest(selection),
        "validation_script_sha256": digest(__file__),
        "selection_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "code_sha256": code_hash(),
        "confirmatory_testing": False,
    }
    manifest = ROOT / "validation_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert all(
            old[k] == metadata[k]
            for k in ("selection_sha256", "validation_script_sha256", "code_sha256")
        )
    else:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {"selection_sha256": digest(selection), "tasks": {}, "confirmatory_testing": False}
    for task, entry in selected["tasks"].items():
        primary, final = entry["primary_selected"], entry["final_selected"]
        summary["tasks"][task] = {}
        for kind, record in (("primary", primary), ("final", final)):
            if record is None:
                summary["tasks"][task][kind] = {
                    "status": "FAILED",
                    "reason": "no stable development configuration",
                }
                continue
            if kind == "final" and primary and final["name"] == primary["name"]:
                summary["tasks"][task][kind] = {
                    **summary["tasks"][task]["primary"],
                    "same_as_primary": True,
                }
                continue
            rows = [run_one(selection, task, record, kind, seed) for seed in (0, 1, 2)]
            values = [r["score"] for r in rows]
            summary["tasks"][task][kind] = {
                "configuration": record["name"],
                "mean": float(np.mean(values)),
                "seed_values": values,
                "sd": float(np.std(values)),
                "status": "COMPLETE",
                "state_probes": [r["state_probe"]["score"] for r in rows],
                "dominant_action_fractions": [r["dominant_action_fraction"] for r in rows],
                "policy_entropies": [r["policy_entropy"] for r in rows],
                "random_mean": float(np.mean([mean_success(r["random_rows"]) for r in rows])),
            }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("Fresh validation complete. STOP: confirmatory testing remains unauthorized.", flush=True)


if __name__ == "__main__":
    main()
