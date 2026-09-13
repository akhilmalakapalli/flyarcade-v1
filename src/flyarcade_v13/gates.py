"""Predefined development and confirmatory success criteria (identical definitions)."""

import numpy as np

MARGIN = 0.05
GAP_CLOSURE = 0.15
PROBE = 0.05
ENTROPY_FLOOR = 1e-3
CONSTANT_ACTION = 0.98
EPSILON = 1e-6


def seed_summary(result):
    ev = result["evaluations"]
    mean = {k: float(np.mean([r["success"] for r in rows])) for k, rows in ev.items()}
    counts = np.sum([r["action_counts"] for r in ev["after"]], axis=0)
    entropies = [h["entropy"] for h in result["history"] if h.get("entropy") is not None]
    gap = (mean["after"] - mean["random"]) / max(mean["reference"] - mean["random"], EPSILON)
    return {
        "after": mean["after"],
        "before": mean["before"],
        "random": mean["random"],
        "reference": mean["reference"],
        "gap_closure": float(gap),
        "state_probe": result["state_probe"]["score"],
        "final_entropy": float(entropies[-1]) if entropies else None,
        "max_greedy_action_fraction": float(counts.max() / max(counts.sum(), 1)),
        "finite": bool(result["finite"]),
        "beats_random_margin": mean["after"] > mean["random"] + MARGIN,
    }


def task_gate(results, *, min_seeds_passing=4):
    """All criteria are strict; the list order matches the mission specification."""
    seeds = [seed_summary(r) for r in results]
    mean = lambda key: float(np.mean([s[key] for s in seeds]))  # noqa: E731
    passing = sum(s["beats_random_margin"] for s in seeds)
    checks = {
        "after_gt_before": mean("after") > mean("before"),
        "after_gt_random_plus_margin": mean("after") > mean("random") + MARGIN,
        "seeds_beating_random_margin": passing >= min_seeds_passing,
        "state_dependence_gt_0.05": mean("state_probe") > PROBE,
        "finite": all(s["finite"] for s in seeds),
        "nonzero_entropy": all((s["final_entropy"] or 0) > ENTROPY_FLOOR for s in seeds),
        "no_constant_action_collapse": all(
            s["max_greedy_action_fraction"] < CONSTANT_ACTION for s in seeds
        ),
        "gap_closure_ge_0.15": mean("gap_closure") >= GAP_CLOSURE,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "means": {
            k: mean(k)
            for k in ("after", "before", "random", "reference", "gap_closure", "state_probe")
        },
        "seeds_beating_random_margin": passing,
        "seed_count": len(seeds),
        "seeds": seeds,
    }
