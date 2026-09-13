"""Pong rescue sprint: validation gate, freeze, confirmatory specs and report.

  gate    - apply the gate written in experiments/v13_pong_development.json to the
            validation results; exits 1 if it fails.
  freeze  - only if the gate passes: write experiments/v13_pong_run_plan.json and the
            confirmatory specs on confirmatory-only seed purposes.
  report  - aggregate confirmatory results against the pre-registered criterion.

Nothing here trains; trials run through scripts/v13_suite.py.
"""

import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DEV = Path("experiments/v13_pong_development.json")
PLAN = Path("experiments/v13_pong_run_plan.json")
VALIDATE = Path("runs/v13-pong-rescue/validate")
CONFIRM = Path("runs/v13-pong-rescue/confirm")
SPECS = Path("experiments/v13_pong/specs")
SEEDS = (0, 1, 2, 3, 4)


def load(path):
    return json.loads(Path(path).read_text())


def per_seed(result):
    m = result["means"]
    counts = result["state_probe"]["action_counts"]
    return {
        "before": m["before"],
        "after": m["after"],
        "random": m["random"],
        "reference": m["reference"],
        "finite": bool(result["finite"]),
        "probe_action_counts": counts,
        "state_dependent": sum(1 for c in counts if c > 0) > 1,
    }


def seed_passes(row):
    return (
        row["after"] > row["before"]
        and row["after"] > row["random"] + 0.10
        and row["finite"]
        and row["state_dependent"]
    )


def gate():
    rows = {}
    for path in sorted(VALIDATE.glob("*/result.json")):
        rows[path.parent.name] = per_seed(load(path))
    if len(rows) != 3:
        raise SystemExit(f"expected 3 validation results, found {len(rows)}")
    mean_after = float(np.mean([r["after"] for r in rows.values()]))
    checks = {
        "mean_after_at_least_0.70": mean_after >= 0.70,
        "every_seed_after_gt_before": all(r["after"] > r["before"] for r in rows.values()),
        "every_seed_after_gt_random_plus_0.10": all(
            r["after"] > r["random"] + 0.10 for r in rows.values()
        ),
        "every_seed_finite": all(r["finite"] for r in rows.values()),
        "every_seed_state_dependent": all(r["state_dependent"] for r in rows.values()),
    }
    report = {
        "mean_after": mean_after,
        "checks": checks,
        "passed": all(checks.values()),
        "seeds": rows,
    }
    Path("artifacts/v13/pong").mkdir(parents=True, exist_ok=True)
    Path("artifacts/v13/pong/validation_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def freeze():
    report = gate()
    if not report["passed"]:
        raise SystemExit("validation gate failed: refusing to freeze")
    if PLAN.exists():
        raise SystemExit(f"{PLAN} already exists; a frozen plan is never rewritten")
    template = load(SPECS / "validate/pong-validate-biological-s0.json")
    frozen = {k: v for k, v in template["config"].items() if k not in ("seed",)}
    for key in ("train_purpose", "eval_purpose"):
        frozen.pop(key)
    conditions = {
        "biological": {"source": "fly", "topology": "biological"},
        "rewired": {"source": "fly", "topology": "rewired-{seed}"},
        "sensory": {"source": "sensory"},
    }
    written = []
    for name, extra in conditions.items():
        for seed in SEEDS:
            config = {**copy.deepcopy(frozen), **extra, "seed": seed}
            if name == "rewired":
                config["topology"] = f"rewired-{seed}"
            config.update(
                train_purpose="conf_train",
                model_purpose="conf_model",
                rollout_purpose="conf_rollout",
                eval_purpose="conf_eval",
            )
            spec = {
                "name": f"pong-confirm-{name}-s{seed}",
                "stage": "pong_rescue_confirm",
                "condition": name,
                "config": config,
                "evaluation": {
                    "purpose": "conf_eval",
                    "episodes": 40,
                    "probe_purpose": "conf_probe",
                },
                "output": str(CONFIRM / f"pong-confirm-{name}-s{seed}"),
            }
            path = SPECS / "confirm" / f"{spec['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec, indent=1) + "\n")
            written.append(str(path))
    plan = {
        "frozen_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "frozen_after": "validation gate passed; before any confirmatory run",
        "frozen_config": frozen,
        "frozen_config_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest(),
        "learner": "NumPy MLP actor-critic (128 tanh) + clipped PPO + GAE, Adam, grad-norm 0.5",
        "defaults_applied": "flyarcade_v13.study.DEFAULTS for every key not listed",
        "conditions": list(conditions),
        "frozen_condition": (
            "the untrained initial policy of each biological trial, evaluated greedily "
            "on the same confirmatory seeds with learning off (reported as 'before')"
        ),
        "seeds": list(SEEDS),
        "seed_purposes": ["conf_train", "conf_model", "conf_rollout", "conf_eval", "conf_probe"],
        "evaluation": "40 greedy episodes, learning disabled, fresh neural and recurrent state",
        "standardisation": (
            "each topology uses its own feature_fit standardizer "
            "(artifacts/v13/standardizers/pong-<topology>-t8-descending.json)"
        ),
        "success_criterion": load(DEV)["confirmatory_success_criterion"],
        "validation_gate": "artifacts/v13/pong/validation_gate.json",
        "specs": written,
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"froze {PLAN} with {len(written)} confirmatory specs")


def report():
    plan = load(PLAN)
    rows = {}
    for condition in plan["conditions"]:
        rows[condition] = {}
        for seed in plan["seeds"]:
            path = CONFIRM / f"pong-confirm-{condition}-s{seed}" / "result.json"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            result = load(path)
            if result["config_sha256"] is None:
                raise SystemExit("result lacks config hash")
            rows[condition][seed] = per_seed(result)
    bio = rows["biological"]
    per_seed_pass = {s: seed_passes(r) for s, r in bio.items()}
    mean = lambda cond, key: float(np.mean([r[key] for r in rows[cond].values()]))  # noqa: E731
    summary = {
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "biological_before_frozen": mean("biological", "before"),
        "biological_after": mean("biological", "after"),
        "rewired_after": mean("rewired", "after"),
        "sensory_after": mean("sensory", "after"),
        "random": mean("biological", "random"),
        "heuristic_reference": mean("biological", "reference"),
        "per_seed": rows,
        "biological_seed_pass": per_seed_pass,
        "criterion": {
            "every_biological_seed_passes": all(per_seed_pass.values()),
            "mean_biological_after_at_least_0.70": mean("biological", "after") >= 0.70,
        },
    }
    summary["criterion"]["met"] = all(summary["criterion"].values())
    Path("artifacts/v13/pong/confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
    for condition, block in rows.items():
        print(condition, [round(r["after"], 3) for r in block.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate", "freeze", "report"])
    args = parser.parse_args()
    {"gate": gate, "freeze": freeze, "report": report}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
