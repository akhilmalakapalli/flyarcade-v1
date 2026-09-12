"""Aggregate independent training seeds; export reproducible scientific figures."""

import hashlib
import json
from pathlib import Path

import numpy as np


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent seed values required")
    rng = np.random.default_rng(2026)
    means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "seed_values": values.tolist(),
        "descriptive_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def mean_metric(result, key):
    return float(np.mean([row["success"] for row in result[key]]))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("artifacts")
    results = {p.parent.name: json.loads(p.read_text()) for p in Path("runs").glob("*/result.json")}
    expected = []
    for task in ("catch", "dodge"):
        for condition in ("learning", "frozen", "rewired", "readout_only"):
            expected += [f"{task}-{condition}-{seed}-200" for seed in range(3)]
        expected += [f"{task}-learning-{seed}-100" for seed in range(3)]
        source = "dodge" if task == "catch" else "catch"
        expected += [f"{task}-learning-{seed}-200-from-{source}" for seed in range(3)]
    if any(name not in results or results[name]["status"] != "COMPLETE" for name in expected):
        raise ValueError("required trial grid incomplete")
    results = {name: results[name] for name in expected}
    if len({r["model_code_sha256"] for r in results.values()}) != 1:
        raise ValueError("trials used different source snapshots")
    summary = {
        "trial_count": len(results),
        "independent_seeds_per_condition": 3,
        "scope": "exploratory, not confirmatory; bootstrap resamples training seeds",
        "model_code_sha256": next(iter(results.values()))["model_code_sha256"],
        "primary": {},
        "contrasts": {},
        "robustness": {},
        "transfer": {},
        "source_graph_sha256": next(iter(results.values()))["source_graph_sha256"],
        "result_files": {
            name: hashlib.sha256((Path("runs") / name / "result.json").read_bytes()).hexdigest()
            for name in expected
        },
    }
    for task in ("catch", "dodge"):
        for condition in ("learning", "frozen", "rewired", "readout_only"):
            rows = [results[f"{task}-{condition}-{s}-200"] for s in range(3)]
            summary["primary"][f"{task}/{condition}"] = {
                **{
                    key: describe([mean_metric(r, key) for r in rows])
                    for key in ("before", "after", "random", "heuristic")
                },
                "change": describe(
                    [mean_metric(r, "after") - mean_metric(r, "before") for r in rows]
                ),
                "recurrent_l1_change": [r["recurrent_l1_change"] for r in rows],
                "readout_l1_change": [r["readout_l1_change"] for r in rows],
            }
        full = [results[f"{task}-learning-{s}-200"] for s in range(3)]
        for control in ("frozen", "rewired", "readout_only"):
            other = [results[f"{task}-{control}-{s}-200"] for s in range(3)]
            summary["contrasts"][f"{task}/learning-minus-{control}"] = describe(
                [
                    mean_metric(a, "after") - mean_metric(b, "after")
                    for a, b in zip(full, other, strict=True)
                ]
            )
        summary["robustness"][task] = {
            key: describe([mean_metric(r, key) for r in full])
            for key in ("after", "sensory_noise_0.1", "edge_ablation_0.1", "neuron_ablation_0.1")
        }
        source = "dodge" if task == "catch" else "catch"
        transfer = [results[f"{task}-learning-{s}-200-from-{source}"] for s in range(3)]
        fresh = [results[f"{task}-learning-{s}-100"] for s in range(3)]
        summary["transfer"][f"{source}-to-{task}"] = {
            "transfer": describe([mean_metric(r, "after") for r in transfer]),
            "target200": describe([mean_metric(r, "after") for r in full]),
            "target100": describe([mean_metric(r, "after") for r in fresh]),
            "paired_minus_target200": describe(
                [
                    mean_metric(a, "after") - mean_metric(b, "after")
                    for a, b in zip(transfer, full, strict=True)
                ]
            ),
        }
    summary["resources"] = {
        "trial_seconds_sum": sum(r["resources"]["elapsed_seconds"] for r in results.values()),
        "max_trial_seconds": max(r["resources"]["elapsed_seconds"] for r in results.values()),
        "max_rss_snapshot_mb": max(r["resources"]["rss_mb"] for r in results.values()),
    }
    (root / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Compact per-seed outcome evidence stays in the repository even though runs/ is ignored.
    (root / "trial_metrics.json").write_text(
        json.dumps(
            {
                name: {
                    "config": r["config"],
                    "training_success": [v["success"] for v in r["training"]],
                    "evaluation": {
                        key: [v["success"] for v in r[key]]
                        for key in (
                            "before",
                            "after",
                            "random",
                            "heuristic",
                            "sensory_noise_0.1",
                            "edge_ablation_0.1",
                            "neuron_ablation_0.1",
                        )
                    },
                    "control": r["control"],
                    "resources": r["resources"],
                }
                for name, r in results.items()
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    # Validated categorical slots 1-4, fixed order, never cycled.
    colors = {
        "learning": "#2a78d6",
        "frozen": "#eb6834",
        "rewired": "#1baf7a",
        "readout_only": "#eda100",
    }
    for ax, task in zip(axes, ("catch", "dodge"), strict=True):
        for condition, color in colors.items():
            values = (
                np.array(
                    [
                        [v["success"] for v in results[f"{task}-{condition}-{s}-200"]["training"]]
                        for s in range(3)
                    ]
                )
                .reshape(3, 10, 20)
                .mean(axis=2)
            )
            ax.plot(np.arange(20, 201, 20), values.mean(axis=0), label=condition, color=color, lw=2)
            ax.fill_between(
                np.arange(20, 201, 20),
                values.min(axis=0),
                values.max(axis=0),
                color=color,
                alpha=0.1,
            )
        ax.set(title=task, xlabel="Training episode", ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Training landing success (20-episode blocks)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("MaleCNS arcade pilot: means and seed ranges (n=3)")
    fig.tight_layout()
    fig.savefig(root / "learning_curves.svg")
    fig.savefig(root / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, task in zip(axes, ("catch", "dodge"), strict=True):
        labels = list(colors)
        for x, condition in enumerate(labels):
            item = summary["primary"][f"{task}/{condition}"]["after"]
            ax.bar(x, item["mean"], color=colors[condition], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], item["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            # Direct value labels: identity and magnitude never rely on colour alone.
            ax.text(
                x,
                max(item["mean"], max(item["seed_values"])) + 0.035,
                f"{item['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        ax.axhline(
            summary["primary"][f"{task}/learning"]["random"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random baseline",
        )
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.set(title=task, ylim=(0, 1), xticks=range(4), xticklabels=labels)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Held-out landing success; points = training seeds")
    fig.tight_layout()
    fig.savefig(root / "heldout_results.svg")
    fig.savefig(root / "heldout_results.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {k: summary[k] for k in ("trial_count", "primary", "contrasts", "resources")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
