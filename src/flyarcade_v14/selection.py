"""Development-only, multi-seed architecture/configuration selection."""

import numpy as np


def summarize(name, phase, config, results):
    complete = len(results) == 3 and all(r.get("status") == "COMPLETE" for r in results)
    stable = complete and all(r["finite"] for r in results)
    values = [r["score"] for r in results if r.get("status") == "COMPLETE"]
    mean = float(np.mean(values)) if values else None
    sd = float(np.std(values)) if values else None
    eligible = stable and all(
        r["state_probe"]["score"] > 0.05 and r["dominant_action_fraction"] < 0.98 for r in results
    )
    return {
        "name": name,
        "phase": phase,
        "config": config,
        "complete": complete,
        "stable": stable,
        "eligible": eligible,
        "scores": values,
        "mean": mean,
        "sd": sd,
        "variance": sd**2 if sd is not None else None,
        "utility": mean - 0.25 * sd if stable else None,
        "state_probes": [r.get("state_probe", {}).get("score") for r in results],
        "failure_reasons": [r.get("failure_reason") for r in results],
    }


def complexity(record):
    c = record["config"]
    return (
        ("linear", "mlp", "gru").index(c["arch"]),
        c["ticks"],
        ("descending", "all").index(c["readout"]),
        c["transitions"],
        record["name"],
    )


def select(records):
    stable = [r for r in records if r["stable"]]
    candidates = [r for r in stable if r["eligible"]] or stable
    best = None
    for row in candidates:
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
