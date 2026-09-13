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


def _seed_values(entry, summary):
    """Per-seed values for each reported condition, from either frozen summary format."""
    seeds = entry["spent_confirmatory_seeds"]
    per = summary["per_seed"]
    if entry["summary_format"] == "pong_rescue":
        bio = per["biological"]
        pick = lambda cond, key: [per[cond][str(s)][key] for s in seeds]  # noqa: E731
        return {
            "biological": pick("biological", "after"),
            "frozen": [bio[str(s)]["before"] for s in seeds],
            "rewired": pick("rewired", "after"),
            "sensory": pick("sensory", "after"),
            "random": [bio[str(s)]["random"] for s in seeds],
            "reference": [bio[str(s)]["reference"] for s in seeds],
        }
    if entry["summary_format"] == "flappy_rescue":
        row = lambda cond, s: per[cond][f"flappy-confirm-{cond}-s{s}"]  # noqa: E731
        return {
            "biological": [row("biological", s)["after"] for s in seeds],
            "frozen": [row("biological", s)["before"] for s in seeds],
            "rewired": [row("rewired", s)["after"] for s in seeds],
            "sensory": [row("sensory", s)["after"] for s in seeds],
            "random": [row("biological", s)["random"] for s in seeds],
            "reference": [row("biological", s)["reference"] for s in seeds],
        }
    raise ValueError(f"unknown frozen summary format for {entry}")


def _criterion(entry, summary):
    if entry["summary_format"] == "pong_rescue":
        return bool(summary["criterion"]["met"]), summary["criterion"]
    verification = json.loads(Path(entry["verification"]).read_text())
    components = {
        **summary["criterion_without_replay"]["checks"],
        "10_checkpoint_replay_and_provenance_verification": bool(verification["passed"]),
    }
    return bool(
        summary["criterion_without_replay"]["passed"] and verification["passed"]
    ), components


def external_summary_entry(task, describe):
    """Summary entry for an externally frozen task, read only from its frozen files.

    Shaped like the multitask report's per-task entries so the shared figures can
    plot it, and explicitly flagged so it is never mistaken for a multitask trial.
    """
    registry = load_registry()
    entry = registry["tasks"][task]
    summary = json.loads(Path(entry["summary"]).read_text())
    primary = {k: describe(v) for k, v in _seed_values(entry, summary).items()}
    passed, components = _criterion(entry, summary)
    perturbations = summary.get("perturbations_biological") or None
    return {
        "external_freeze": True,
        "architecture": f"{entry['frozen_config'].get('arch', 'mlp')} (separate {task} freeze)",
        "criteria": {
            "passed": passed,
            "criterion_source": entry["frozen_plan"],
            "components": components,
            "not_rescored_under_multitask_gate": True,
        },
        "primary": primary,
        "perturbations": None,
        "perturbation_means_reported_by_frozen_study": perturbations,
        "frozen_plan": entry["frozen_plan"],
        "frozen_plan_sha256": entry["file_sha256"][entry["frozen_plan"]],
        "frozen_plan_commit": entry["frozen_plan_commit"],
        "results_commit": entry["results_commit"],
        "protocol_differences_from_v13_multitask_plan": entry[
            "protocol_differences_from_v13_multitask_plan"
        ],
    }
