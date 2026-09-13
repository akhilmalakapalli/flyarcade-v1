"""Registry of v1.3 tasks frozen and completed outside the multitask plan.

Deliberately lives in scripts/, not src/flyarcade_v13/: the multitask study hashes
src/flyarcade_v13 and scripts/v13_trial.py into every checkpoint and run plan, and
adding this guard must not invalidate trials that are already running.
"""

import hashlib
import json
from pathlib import Path

REGISTRY = Path("experiments/v13_external_frozen.json")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registry(verify=True):
    registry = json.loads(REGISTRY.read_text())
    if verify:
        for task, entry in registry["tasks"].items():
            for path, expected in entry["file_sha256"].items():
                if not Path(path).exists() or _sha(path) != expected:
                    raise ValueError(f"externally frozen {task} file changed or missing: {path}")
    return registry


def external_tasks():
    return tuple(load_registry()["tasks"])


def spent(spec):
    """True if a spec would touch confirmatory seeds already spent by an external freeze."""
    registry = load_registry(verify=False)
    config = spec.get("config", {})
    entry = registry["tasks"].get(config.get("task"))
    if entry is None:
        return False
    purposes = {
        config.get(k) for k in ("train_purpose", "eval_purpose", "model_purpose", "rollout_purpose")
    }
    evaluation = spec.get("evaluation", {})
    purposes |= {evaluation.get(k) for k in ("purpose", "probe_purpose", "perturb_purpose")}
    purposes.discard(None)
    return bool(purposes & set(entry["spent_confirmatory_purposes"]))


def confirmatory_tasks(targets):
    """Tasks the multitask freeze may still write confirmatory specs for."""
    external = set(external_tasks())
    return tuple(t for t in targets if t not in external)


def external_summary_entry(task, describe):
    """Summary entry for an externally frozen task, read only from its frozen files.

    Shaped like the multitask report's per-task entries so the shared figures can
    plot it, and explicitly flagged so it is never mistaken for a multitask trial.
    """
    registry = load_registry()
    entry = registry["tasks"][task]
    summary = json.loads(Path(entry["summary"]).read_text())
    per = summary["per_seed"]
    seeds = [str(s) for s in entry["spent_confirmatory_seeds"]]
    bio = per["biological"]
    primary = {
        "biological": describe([bio[s]["after"] for s in seeds]),
        "frozen": describe([bio[s]["before"] for s in seeds]),
        "rewired": describe([per["rewired"][s]["after"] for s in seeds]),
        "sensory": describe([per["sensory"][s]["after"] for s in seeds]),
        "random": describe([bio[s]["random"] for s in seeds]),
        "reference": describe([bio[s]["reference"] for s in seeds]),
    }
    return {
        "external_freeze": True,
        "architecture": "mlp (separate Pong freeze)",
        "criteria": {
            "passed": bool(summary["criterion"]["met"]),
            "criterion_source": "experiments/v13_pong_development.json (pre-registered)",
            "components": summary["criterion"],
            "not_rescored_under_multitask_gate": True,
        },
        "primary": primary,
        "perturbations": None,
        "frozen_plan": entry["frozen_plan"],
        "frozen_plan_sha256": entry["file_sha256"][entry["frozen_plan"]],
        "frozen_plan_commit": entry["frozen_plan_commit"],
        "results_commit": entry["results_commit"],
        "protocol_differences_from_v13_multitask_plan": entry[
            "protocol_differences_from_v13_multitask_plan"
        ],
    }
