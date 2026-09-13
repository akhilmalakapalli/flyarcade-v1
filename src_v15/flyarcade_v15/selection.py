"""Pre-registered v1.5 multi-seed selection (development data only)."""

import numpy as np

HEADLINE_READOUTS = ("descending", "nonvisual", "cb_intrinsic", "pooled")
ARCH_ORDER = ("linear", "mlp", "gru")
READOUT_ORDER = ("descending", "pooled", "cb_intrinsic", "nonvisual", "visual_cb", "visual", "all")


def summarize(name, stage, config, results, *, role, seeds=3):
    complete = len(results) == seeds and all(r.get("status") == "COMPLETE" for r in results)
    stable = complete and all(r["finite"] for r in results)
    values = [r["score"] for r in results if r.get("status") == "COMPLETE"]
    mean = float(np.mean(values)) if values else None
    sd = float(np.std(values)) if values else None
    gates = stable and all(
        r["state_probe"]["score"] > 0.05 and r["dominant_action_fraction"] < 0.98 for r in results
    )
    return {
        "name": name,
        "stage": stage,
        "role": role,  # headline | control | rescue
        "config": config,
        "complete": complete,
        "stable": stable,
        "gates_passed": gates,
        "eligible": gates and role != "control",
        "scores": values,
        "mean": mean,
        "sd": sd,
        "utility": mean - 0.25 * sd if stable else None,
        "state_probes": [r.get("state_probe", {}).get("score") for r in results],
        "dominant_action_fractions": [r.get("dominant_action_fraction") for r in results],
        "chosen_checkpoint_transitions": [r.get("chosen_checkpoint_transitions") for r in results],
        "failure_reasons": [r.get("failure_reason") for r in results],
    }


def complexity(record):
    c = record["config"]
    width = c.get("hidden", 0) if c["arch"] != "linear" else 0
    return (
        ARCH_ORDER.index(c["arch"]),
        width,
        c.get("ticks", 0),
        READOUT_ORDER.index(c.get("readout", "descending")),
        c["transitions"],
        record["name"],
    )


def select(records, role="headline"):
    """Best eligible record of the given role by mean - 0.25*SD; ties prefer simpler."""
    pool = [r for r in records if r["role"] == role and r["eligible"]]
    best = None
    for row in pool:
        if (
            best is None
            or row["utility"] > best["utility"] + 1e-12
            or (
                abs(row["utility"] - best["utility"]) <= 1e-12
                and complexity(row) < complexity(best)
            )
        ):
            best = row
    return best


def still_rising(results, budget, fraction=0.75, minimum=2):
    """Learning-curve criterion for the longer-budget arm."""
    late = sum(r["chosen_checkpoint_transitions"] >= fraction * budget for r in results)
    return late >= minimum
