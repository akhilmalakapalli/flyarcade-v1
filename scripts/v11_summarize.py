"""Aggregate the v1.1 confirmatory grid and compare it against the frozen v1 result."""

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


def mean_success(result, key):
    return float(np.mean([row["success"] for row in result[key]]))


def state_dependence(result, key):
    counts = np.sum([row["action_counts"] for row in result[key]], axis=0)
    return float(1 - counts.max() / counts.sum())


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("artifacts")
    plan = json.loads(Path("experiments/v11_run_plan.json").read_text())
    results = {}
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            for seed in plan["seeds"]:
                name = f"v11-{task}-{condition}-{seed}"
                path = Path("runs") / name / "result.json"
                if not path.exists():
                    raise SystemExit(f"missing trial {name}")
                results[name] = json.loads(path.read_text())
    if len({r["plan_sha256"] for r in results.values()}) != 1:
        raise ValueError("trials ran against different frozen plans")
    if len({r["v11_code_sha256"] for r in results.values()}) != 1:
        raise ValueError("trials ran against different v1.1 source snapshots")

    summary = {
        "trial_count": len(results),
        "plan_sha256": next(iter(results.values()))["plan_sha256"],
        "v11_code_sha256": next(iter(results.values()))["v11_code_sha256"],
        "stage": plan["stage"],
        "hyperparameters": plan["hyperparameters"],
        "evaluation": "greedy policy mode on confirmatory seeds never used for tuning",
        "scope": "exploratory-to-confirmatory pilot; bootstrap resamples training seeds",
        "primary": {},
        "contrasts": {},
        "state_dependence": {},
        "robustness": {},
        "result_files": {
            name: hashlib.sha256((Path("runs") / name / "result.json").read_bytes()).hexdigest()
            for name in results
        },
    }
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            rows = [results[f"v11-{task}-{condition}-{s}"] for s in plan["seeds"]]
            summary["primary"][f"{task}/{condition}"] = {
                key: describe([mean_success(r, key) for r in rows])
                for key in ("before", "after", "random", "heuristic")
            }
            summary["primary"][f"{task}/{condition}"]["change"] = describe(
                [mean_success(r, "after") - mean_success(r, "before") for r in rows]
            )
            summary["state_dependence"][f"{task}/{condition}"] = describe(
                [state_dependence(r, "after") for r in rows]
            )
        eprop = [results[f"v11-{task}-eprop-{s}"] for s in plan["seeds"]]
        summary["robustness"][task] = {
            key: describe([mean_success(r, key) for r in eprop])
            for key in (
                "after",
                "sensory_noise_0.1",
                "edge_ablation_0.1",
                "neuron_ablation_0.1",
            )
        }
        for control in ("frozen", "rewired"):
            other = [results[f"v11-{task}-{control}-{s}"] for s in plan["seeds"]]
            summary["contrasts"][f"{task}/eprop-minus-{control}"] = describe(
                [
                    mean_success(a, "after") - mean_success(b, "after")
                    for a, b in zip(eprop, other, strict=True)
                ]
            )
        summary["contrasts"][f"{task}/eprop-minus-random"] = describe(
            [mean_success(r, "after") - mean_success(r, "random") for r in eprop]
        )
    summary["resources"] = {
        "max_trial_seconds": max(r["resources"]["elapsed_seconds"] for r in results.values()),
        "max_rss_snapshot_mb": max(r["resources"]["rss_mb"] for r in results.values()),
    }
    (root / "v11_results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "v11_primary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task",
                "condition",
                "before",
                "after",
                "change",
                "change_lo",
                "change_hi",
                "seed0",
                "seed1",
                "seed2",
                "random",
                "heuristic",
                "state_dependence",
            ]
        )
        for task in plan["tasks"]:
            for condition in plan["conditions"]:
                item = summary["primary"][f"{task}/{condition}"]
                writer.writerow(
                    [
                        task,
                        condition,
                        f"{item['before']['mean']:.4f}",
                        f"{item['after']['mean']:.4f}",
                        f"{item['change']['mean']:.4f}",
                        f"{item['change']['descriptive_bootstrap_95'][0]:.4f}",
                        f"{item['change']['descriptive_bootstrap_95'][1]:.4f}",
                        *[f"{v:.4f}" for v in item["after"]["seed_values"]],
                        f"{item['random']['mean']:.4f}",
                        f"{item['heuristic']['mean']:.4f}",
                        f"{summary['state_dependence'][f'{task}/{condition}']['mean']:.4f}",
                    ]
                )
    with (tables / "v11_contrasts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "mean", "lo", "hi", "seed0", "seed1", "seed2"])
        for name, item in summary["contrasts"].items():
            writer.writerow(
                [
                    name,
                    f"{item['mean']:.4f}",
                    f"{item['descriptive_bootstrap_95'][0]:.4f}",
                    f"{item['descriptive_bootstrap_95'][1]:.4f}",
                    *[f"{v:.4f}" for v in item["seed_values"]],
                ]
            )

    with (tables / "v11_robustness.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "perturbation", "mean", "lo", "hi", "seed0", "seed1", "seed2"])
        for task, block in summary["robustness"].items():
            for key, item in block.items():
                writer.writerow(
                    [
                        task,
                        key,
                        f"{item['mean']:.4f}",
                        f"{item['descriptive_bootstrap_95'][0]:.4f}",
                        f"{item['descriptive_bootstrap_95'][1]:.4f}",
                        *[f"{v:.4f}" for v in item["seed_values"]],
                    ]
                )

    order = [*plan["conditions"], "random", "heuristic"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, plan["tasks"], strict=True):
        for x, condition in enumerate(order):
            if condition in plan["conditions"]:
                item = summary["primary"][f"{task}/{condition}"]["after"]
            else:
                item = summary["primary"][f"{task}/eprop"][condition]
            ax.bar(x, item["mean"], color=COLORS[condition], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], item["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            ax.text(
                x,
                max(item["mean"], max(item["seed_values"])) + 0.035,
                f"{item['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        ax.set(title=task, ylim=(0, 1.18), xticks=range(len(order)), xticklabels=order)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Confirmatory held-out success; points = seeds")
    fig.suptitle("v1.1 actor-critic e-prop on unseen confirmatory seeds (n=3)")
    fig.tight_layout()
    fig.savefig(root / "v11_confirmatory.svg")
    fig.savefig(root / "v11_confirmatory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, plan["tasks"], strict=True):
        for condition in ("eprop", "rewired"):
            curves = np.array(
                [
                    [r["success"] for r in results[f"v11-{task}-{condition}-{s}"]["training"]]
                    for s in plan["seeds"]
                ]
            )
            block = max(curves.shape[1] // 20, 1)
            trimmed = curves[:, : block * 20].reshape(len(plan["seeds"]), 20, block).mean(axis=2)
            x = np.arange(1, 21) * block
            ax.plot(x, trimmed.mean(axis=0), color=COLORS[condition], lw=2, label=condition)
            ax.fill_between(
                x, trimmed.min(axis=0), trimmed.max(axis=0), color=COLORS[condition], alpha=0.12
            )
        ax.axhline(
            summary["primary"][f"{task}/eprop"]["random"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random baseline",
        )
        ax.set(title=task, xlabel="Training episode", ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Training landing success")
    fig.suptitle("v1.1 training curves: biological vs degree-rewired topology")
    fig.tight_layout()
    fig.savefig(root / "v11_learning_curves.svg")
    fig.savefig(root / "v11_learning_curves.png", dpi=180)
    plt.close(fig)
    labels = ["unperturbed", "sensory noise", "edge ablation", "neuron ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, plan["tasks"], strict=True):
        for x, key in enumerate(summary["robustness"][task]):
            item = summary["robustness"][task][key]
            ax.bar(x, item["mean"], color=COLORS["eprop"], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], item["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            ax.text(
                x,
                max(item["mean"], max(item["seed_values"])) + 0.035,
                f"{item['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        ax.axhline(
            summary["primary"][f"{task}/eprop"]["random"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random baseline",
        )
        ax.set(title=task, ylim=(0, 1.18), xticks=range(4), xticklabels=labels)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    axes[0].set_ylabel("Held-out success; points = seeds")
    fig.suptitle("v1.1 lesions applied to a genuinely state-dependent policy (n=3)")
    fig.tight_layout()
    fig.savefig(root / "v11_robustness.svg")
    fig.savefig(root / "v11_robustness.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("primary", "contrasts", "robustness")}, indent=2))


if __name__ == "__main__":
    main()
