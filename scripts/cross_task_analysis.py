"""Cross-task comparison across catch, dodge and Snake.

Reads only recorded confirmatory results. Reports each task's own numbers and does
not force a single topology conclusion where the tasks disagree.
"""

import csv
import json
from pathlib import Path

import numpy as np

TASKS = ("catch", "dodge", "snake")


def normalised(value, random_value, heuristic_value):
    """Where a score sits between the random policy (0) and the heuristic (1)."""
    span = heuristic_value - random_value
    return float((value - random_value) / span) if abs(span) > 1e-9 else float("nan")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("artifacts")
    v11 = json.loads((root / "v11_results_summary.json").read_text())
    snake = json.loads((root / "snake_results_summary.json").read_text())
    v1 = json.loads((root / "results_summary.json").read_text())
    snake_plan = json.loads(Path("experiments/snake_run_plan.json").read_text())

    rows = {}
    for task in ("catch", "dodge"):
        block = v11["primary"]
        rows[task] = {
            "metric": "landing success",
            "v1_after": v1["primary"][f"{task}/learning"]["after"]["mean"],
            "v1_random": v1["primary"][f"{task}/learning"]["random"]["mean"],
            "before": block[f"{task}/eprop"]["before"]["mean"],
            "after": block[f"{task}/eprop"]["after"]["mean"],
            "after_seeds": block[f"{task}/eprop"]["after"]["seed_values"],
            "random": block[f"{task}/eprop"]["random"]["mean"],
            "heuristic": block[f"{task}/eprop"]["heuristic"]["mean"],
            "rewired": block[f"{task}/rewired"]["after"]["mean"],
            "frozen": block[f"{task}/frozen"]["after"]["mean"],
            "state_dependence": v11["state_dependence"][f"{task}/eprop"]["mean"],
            "training_episodes": 600,
            "actions": 3,
            "terminal_failure": False,
        }
    rows["snake"] = {
        "metric": "food eaten",
        "v1_after": None,
        "v1_random": None,
        "before": snake["primary"]["eprop"]["before_food"]["mean"],
        "after": snake["primary"]["eprop"]["after_food"]["mean"],
        "after_seeds": snake["primary"]["eprop"]["after_food"]["seed_values"],
        "random": snake["primary"]["eprop"]["random_food"]["mean"],
        "heuristic": snake["primary"]["eprop"]["heuristic_food"]["mean"],
        "rewired": snake["primary"]["rewired"]["after_food"]["mean"],
        "frozen": snake["primary"]["frozen"]["after_food"]["mean"],
        "state_dependence": snake["primary"]["eprop"]["state_dependence"]["mean"],
        "training_episodes": snake_plan["training_episodes"],
        "actions": 4,
        "terminal_failure": True,
    }
    for task, row in rows.items():
        row["normalised_after"] = normalised(row["after"], row["random"], row["heuristic"])
        row["normalised_rewired"] = normalised(row["rewired"], row["random"], row["heuristic"])
        row["topology_gap"] = row["after"] - row["rewired"]
        row["normalised_topology_gap"] = row["normalised_after"] - row["normalised_rewired"]

    # Lesion sensitivity, expressed as the fraction of the learned gain retained.
    lesions = {}
    for task in ("catch", "dodge"):
        block = v11["robustness"][task]
        base = block["after"]["mean"]
        random_value = rows[task]["random"]
        lesions[task] = {
            key: normalised(block[key]["mean"], random_value, base)
            for key in ("sensory_noise_0.1", "edge_ablation_0.1", "neuron_ablation_0.1")
        }
    base = snake["robustness"]["after"]["mean"]
    lesions["snake"] = {
        key: normalised(snake["robustness"][key]["mean"], rows["snake"]["random"], base)
        for key in (
            "sensory_noise_0.1",
            "edge_ablation_0.1",
            "neuron_ablation_0.05",
            "neuron_ablation_0.10",
            "neuron_ablation_0.25",
        )
    }

    firing = {}
    for task in ("catch", "dodge"):
        firing[task] = [
            json.loads((Path("runs") / f"v11-{task}-eprop-{s}" / "result.json").read_text())[
                "training"
            ][-1].get("firing_rate")
            for s in (0, 1, 2)
        ]
    firing["snake"] = [
        json.loads((Path("runs") / f"snake-eprop-{s}" / "result.json").read_text())["metrics"][
            "final_firing_rate"
        ]
        for s in (0, 1, 2)
    ]

    report = {
        "scope": (
            "Cross-task comparison of confirmatory results only. Metrics differ by "
            "task, so comparisons use each task's own random and heuristic anchors."
        ),
        "tasks": rows,
        "lesion_retained_fraction_of_learned_gain": lesions,
        "final_firing_rate_by_task": {
            k: {"seeds": v, "mean": float(np.mean(v))} for k, v in firing.items()
        },
        "questions": {
            "same_substrate_supports_all_three": all(
                rows[t]["after"] > rows[t]["random"] for t in TASKS
            ),
            "topology_consistent_across_tasks": None,
            "harder_task_more_lesion_sensitive": None,
        },
    }
    gaps = {t: rows[t]["normalised_topology_gap"] for t in TASKS}
    report["questions"]["topology_consistent_across_tasks"] = (
        "biological never exceeds rewired on any task; the gap is negative on "
        f"catch ({gaps['catch']:+.3f}) and snake ({gaps['snake']:+.3f}) and near zero "
        f"on dodge ({gaps['dodge']:+.3f})"
    )
    report["questions"]["harder_task_more_lesion_sensitive"] = (
        "snake retains "
        f"{lesions['snake']['neuron_ablation_0.10']:.2f} of its learned gain under 10% "
        f"neuron ablation versus {lesions['catch']['neuron_ablation_0.1']:.2f} for catch "
        f"and {lesions['dodge']['neuron_ablation_0.1']:.2f} for dodge"
    )
    (root / "cross_task_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    tables = root / "tables"
    with (tables / "cross_task.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task",
                "metric",
                "actions",
                "training_episodes",
                "random",
                "before",
                "after",
                "rewired",
                "frozen",
                "heuristic",
                "normalised_after",
                "normalised_rewired",
                "state_dependence",
                "final_firing_rate",
            ]
        )
        for task in TASKS:
            row = rows[task]
            writer.writerow(
                [
                    task,
                    row["metric"],
                    row["actions"],
                    row["training_episodes"],
                    f"{row['random']:.3f}",
                    f"{row['before']:.3f}",
                    f"{row['after']:.3f}",
                    f"{row['rewired']:.3f}",
                    f"{row['frozen']:.3f}",
                    f"{row['heuristic']:.3f}",
                    f"{row['normalised_after']:.3f}",
                    f"{row['normalised_rewired']:.3f}",
                    f"{row['state_dependence']:.3f}",
                    f"{np.mean(firing[task]):.4f}",
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(TASKS))
    axes[0].bar(
        x - 0.19,
        [rows[t]["normalised_after"] for t in TASKS],
        width=0.36,
        color="#2a78d6",
        label="biological",
    )
    axes[0].bar(
        x + 0.19,
        [rows[t]["normalised_rewired"] for t in TASKS],
        width=0.36,
        color="#1baf7a",
        label="degree-rewired",
    )
    for i, task in enumerate(TASKS):
        for offset, key in ((-0.19, "normalised_after"), (0.19, "normalised_rewired")):
            axes[0].text(
                i + offset,
                rows[task][key] + 0.02,
                f"{rows[task][key]:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
    axes[0].axhline(0, color="#52514e", ls="--", lw=1.2)
    axes[0].set(
        title="Learned performance, scaled between\nrandom (0) and heuristic (1)",
        xticks=x,
        xticklabels=TASKS,
        ylabel="Normalised score",
        ylim=(-0.05, 1.05),
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)

    keys = ("sensory_noise_0.1", "edge_ablation_0.1", "neuron_ablation_0.1")
    labels = ("sensory noise", "10% edges", "10% neurons")
    width = 0.26
    for j, (key, label) in enumerate(zip(keys, labels, strict=True)):
        values = [
            lesions[t]["neuron_ablation_0.10" if t == "snake" and "neuron" in key else key]
            for t in TASKS
        ]
        axes[1].bar(
            x + (j - 1) * width,
            values,
            width=width,
            color=("#2a78d6", "#eb6834", "#1baf7a")[j],
            label=label,
        )
    axes[1].axhline(1.0, color="#52514e", ls="--", lw=1.2, label="intact policy")
    axes[1].set(
        title="Fraction of the learned gain retained\nunder perturbation",
        xticks=x,
        xticklabels=TASKS,
        ylabel="Retained fraction",
    )
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    fig.suptitle("Cross-task comparison: one MaleCNS substrate, one learning rule, three tasks")
    fig.tight_layout()
    fig.savefig(root / "cross_task.svg")
    fig.savefig(root / "cross_task.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["questions"], indent=2))
    print(json.dumps(report["lesion_retained_fraction_of_learned_gain"], indent=2))


if __name__ == "__main__":
    main()
