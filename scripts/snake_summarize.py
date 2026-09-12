"""Aggregate the frozen Snake confirmatory grid into tables and figures."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

COLORS = {
    "eprop": "#2a78d6",
    "frozen": "#eb6834",
    "rewired": "#1baf7a",
    "random": "#eda100",
    "heuristic": "#52514e",
}
PERTURBATIONS = (
    "after",
    "sensory_noise_0.1",
    "edge_ablation_0.1",
    "neuron_ablation_0.05",
    "neuron_ablation_0.10",
    "neuron_ablation_0.25",
)
PERTURBATION_LABELS = (
    "intact",
    "sensory noise",
    "10% edges",
    "5% neurons",
    "10% neurons",
    "25% neurons",
)


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


def metric(result, key, field):
    return float(np.mean([row[field] for row in result[key]]))


def state_dependence(result, key):
    counts = np.sum([row["action_counts"] for row in result[key]], axis=0)
    return float(1 - counts.max() / counts.sum())


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("artifacts")
    plan = json.loads(Path("experiments/snake_run_plan.json").read_text())
    results = {}
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            name = f"snake-{condition}-{seed}"
            path = Path("runs") / name / "result.json"
            if not path.exists():
                raise SystemExit(f"missing trial {name}")
            results[name] = json.loads(path.read_text())
    if len({r["plan_sha256"] for r in results.values()}) != 1:
        raise ValueError("trials ran against different frozen plans")

    summary = {
        "trial_count": len(results),
        "plan_sha256": next(iter(results.values()))["plan_sha256"],
        "v11_code_sha256": next(iter(results.values()))["v11_code_sha256"],
        "hyperparameters": plan["hyperparameters"],
        "training_episodes": plan["training_episodes"],
        "evaluation": plan["conditions"] and "greedy mode, learning disabled, unseen seeds",
        "primary": {},
        "contrasts": {},
        "robustness": {},
        "result_files": {
            name: hashlib.sha256((Path("runs") / name / "result.json").read_bytes()).hexdigest()
            for name in results
        },
    }
    for condition in plan["conditions"]:
        rows = [results[f"snake-{condition}-{s}"] for s in plan["seeds"]]
        block = {}
        for field in ("food", "steps", "return"):
            block[f"after_{field}"] = describe([metric(r, "after", field) for r in rows])
            block[f"before_{field}"] = describe([metric(r, "before", field) for r in rows])
            block[f"random_{field}"] = describe([metric(r, "random", field) for r in rows])
            block[f"heuristic_{field}"] = describe([metric(r, "heuristic", field) for r in rows])
        block["state_dependence"] = describe([state_dependence(r, "after") for r in rows])
        block["learning_curve_auc_food"] = describe(
            [r["metrics"]["learning_curve_auc_food"] for r in rows]
        )
        block["final_entropy"] = describe(
            [r["metrics"]["final_entropy"] for r in rows]
            if condition != "frozen"
            else [0.0, 0.0, 0.0]
        )
        thresholds = [r["metrics"]["episodes_to_one_food"] for r in rows]
        block["episodes_to_one_food"] = thresholds
        summary["primary"][condition] = block

    eprop = [results[f"snake-eprop-{s}"] for s in plan["seeds"]]
    for control in ("frozen", "rewired"):
        other = [results[f"snake-{control}-{s}"] for s in plan["seeds"]]
        summary["contrasts"][f"eprop-minus-{control}-food"] = describe(
            [
                metric(a, "after", "food") - metric(b, "after", "food")
                for a, b in zip(eprop, other, strict=True)
            ]
        )
    summary["contrasts"]["eprop-minus-random-food"] = describe(
        [metric(r, "after", "food") - metric(r, "random", "food") for r in eprop]
    )
    summary["robustness"] = {
        key: describe([metric(r, key, "food") for r in eprop]) for key in PERTURBATIONS
    }
    summary["robustness_steps"] = {
        key: describe([metric(r, key, "steps") for r in eprop]) for key in PERTURBATIONS
    }
    summary["resources"] = {
        "max_trial_seconds": max(r["resources"]["elapsed_seconds"] for r in results.values()),
        "max_rss_snapshot_mb": max(r["resources"]["rss_mb"] for r in results.values()),
    }
    (root / "snake_results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "snake_primary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "before_food",
                "after_food",
                "after_food_lo",
                "after_food_hi",
                "seed0",
                "seed1",
                "seed2",
                "after_steps",
                "after_return",
                "random_food",
                "heuristic_food",
                "state_dependence",
                "auc_food",
                "episodes_to_one_food",
            ]
        )
        for condition in plan["conditions"]:
            b = summary["primary"][condition]
            writer.writerow(
                [
                    condition,
                    f"{b['before_food']['mean']:.3f}",
                    f"{b['after_food']['mean']:.3f}",
                    f"{b['after_food']['descriptive_bootstrap_95'][0]:.3f}",
                    f"{b['after_food']['descriptive_bootstrap_95'][1]:.3f}",
                    *[f"{v:.3f}" for v in b["after_food"]["seed_values"]],
                    f"{b['after_steps']['mean']:.2f}",
                    f"{b['after_return']['mean']:.3f}",
                    f"{b['random_food']['mean']:.3f}",
                    f"{b['heuristic_food']['mean']:.3f}",
                    f"{b['state_dependence']['mean']:.3f}",
                    f"{b['learning_curve_auc_food']['mean']:.3f}",
                    ";".join(str(x) for x in b["episodes_to_one_food"]),
                ]
            )
    with (tables / "snake_robustness.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["perturbation", "food", "food_lo", "food_hi", "steps", "s0", "s1", "s2"])
        for key in PERTURBATIONS:
            item = summary["robustness"][key]
            writer.writerow(
                [
                    key,
                    f"{item['mean']:.3f}",
                    f"{item['descriptive_bootstrap_95'][0]:.3f}",
                    f"{item['descriptive_bootstrap_95'][1]:.3f}",
                    f"{summary['robustness_steps'][key]['mean']:.2f}",
                    *[f"{v:.3f}" for v in item["seed_values"]],
                ]
            )
    with (tables / "snake_contrasts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "mean", "lo", "hi", "seed0", "seed1", "seed2"])
        for name, item in summary["contrasts"].items():
            writer.writerow(
                [
                    name,
                    f"{item['mean']:.3f}",
                    f"{item['descriptive_bootstrap_95'][0]:.3f}",
                    f"{item['descriptive_bootstrap_95'][1]:.3f}",
                    *[f"{v:.3f}" for v in item["seed_values"]],
                ]
            )

    order = [*plan["conditions"], "random", "heuristic"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field, label in zip(
        axes, ("food", "steps"), ("Food eaten per episode", "Steps survived"), strict=True
    ):
        for x, condition in enumerate(order):
            if condition in plan["conditions"]:
                item = summary["primary"][condition][f"after_{field}"]
            else:
                item = summary["primary"]["eprop"][f"{condition}_{field}"]
            ax.bar(x, item["mean"], color=COLORS[condition], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], item["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            top = max(item["mean"], max(item["seed_values"]))
            ax.text(
                x,
                top * 1.03 + 0.02,
                f"{item['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        ax.set(title=label, xticks=range(len(order)), xticklabels=order)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Snake on unseen confirmatory seeds (n=3 training seeds)")
    fig.tight_layout()
    fig.savefig(root / "snake_confirmatory.svg")
    fig.savefig(root / "snake_confirmatory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field, label in zip(
        axes, ("food", "steps"), ("Food eaten", "Steps survived"), strict=True
    ):
        for condition in ("eprop", "rewired"):
            curves = np.array(
                [
                    [r[field] for r in results[f"snake-{condition}-{s}"]["training"]]
                    for s in plan["seeds"]
                ]
            )
            block = curves.shape[1] // 20
            trimmed = curves[:, : block * 20].reshape(len(plan["seeds"]), 20, block).mean(axis=2)
            x = np.arange(1, 21) * block
            ax.plot(x, trimmed.mean(axis=0), color=COLORS[condition], lw=2, label=condition)
            ax.fill_between(
                x, trimmed.min(axis=0), trimmed.max(axis=0), color=COLORS[condition], alpha=0.12
            )
        ax.axhline(
            summary["primary"]["eprop"][f"random_{field}"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random policy",
        )
        ax.set(title=label, xlabel="Training episode")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Training-episode mean")
    fig.suptitle("Snake learning curves: biological vs degree-rewired topology")
    fig.tight_layout()
    fig.savefig(root / "snake_learning_curves.svg")
    fig.savefig(root / "snake_learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, block, label in zip(
        axes,
        (summary["robustness"], summary["robustness_steps"]),
        ("Food eaten", "Steps survived"),
        strict=True,
    ):
        for x, key in enumerate(PERTURBATIONS):
            item = block[key]
            ax.bar(x, item["mean"], color=COLORS["eprop"], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], item["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            top = max(item["mean"], max(item["seed_values"]))
            ax.text(
                x,
                top * 1.03 + 0.02,
                f"{item['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        ax.set(title=label, xticks=range(len(PERTURBATIONS)), xticklabels=PERTURBATION_LABELS)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Snake perturbations applied to frozen learned weights (n=3)")
    fig.tight_layout()
    fig.savefig(root / "snake_robustness.svg")
    fig.savefig(root / "snake_robustness.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("primary", "contrasts", "robustness")}, indent=2))


if __name__ == "__main__":
    main()
