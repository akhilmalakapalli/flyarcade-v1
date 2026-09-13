"""Generate the six-task tables, figures and a data-derived manuscript extension."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from v11_summarize import describe

ROOT = Path("artifacts/v12")
TASKS = ("catch", "dodge", "flappy", "pong", "breakout", "snake")
CONDITIONS = ("biological", "frozen", "rewired", "random", "heuristic")
COLORS = ("#2878b5", "#b4b4b4", "#e58b28", "#ad587c", "#399473")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(task, condition, seed):
    historical = task in ("catch", "dodge")
    arm = "eprop" if historical and condition == "biological" else condition
    path = Path("runs") / f"{'v11' if historical else 'v12'}-{task}-{arm}-{seed}" / "result.json"
    result = json.loads(path.read_text())
    if historical:
        evaluations = {key: result[key] for key in ("after", "random", "heuristic")}
        evaluations.update(
            {
                key: result[key + "_0.1"]
                for key in ("sensory_noise", "edge_ablation", "neuron_ablation")
            }
        )
    else:
        evaluations = result["evaluations"]
    return result, evaluations, path


def mean(rows):
    return float(np.mean([row["success"] for row in rows]))


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in rows
        )


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "savefig.dpi": 200,
            "svg.hashsalt": "flyarcade-v12",
        }
    )
    summary = {
        "n": 3,
        "uncertainty": "descriptive 95% seed bootstrap; no p-values",
        "plan_sha256": digest("experiments/v12_run_plan.json"),
        "tasks": {},
        "result_files": {},
        "historical": ["catch", "dodge"],
    }
    primary, contrasts, perturbations, raw, archive, curves = [], [], [], [], {}, {}
    for task in TASKS:
        arms = {}
        evaluations = {}
        curves[task] = {}
        for condition in CONDITIONS[:3]:
            arms[condition] = []
            evaluations[condition] = []
            for seed in range(3):
                result, ev, path = read(task, condition, seed)
                arms[condition].append(result)
                evaluations[condition].append(ev)
                summary["result_files"][str(path)] = digest(path)
                archive[str(path)] = result
                if task not in ("catch", "dodge"):
                    for phase, rows in {"before": result["before"], **ev}.items():
                        for index, row in enumerate(rows):
                            raw.append(
                                {
                                    "task": task,
                                    "condition": condition,
                                    "seed": seed,
                                    "phase": phase,
                                    "episode": index,
                                    **{
                                        k: v
                                        for k, v in row.items()
                                        if isinstance(v, (int, float, str))
                                    },
                                    "action_counts": json.dumps(row["action_counts"]),
                                }
                            )
            curves[task][condition] = [
                [r["success"] for r in result["training"]] for result in arms[condition]
            ]
        values = {
            condition: [mean(ev["after"]) for ev in evaluations[condition]]
            for condition in CONDITIONS[:3]
        }
        values.update(
            {
                condition: [mean(ev[condition]) for ev in evaluations["biological"]]
                for condition in CONDITIONS[3:]
            }
        )
        before = [mean(result["before"]) for result in arms["biological"]]
        delta = np.asarray(values["biological"]) - before
        task_summary = {
            "primary": {c: describe(v) for c, v in values.items()},
            "before": describe(before),
            "change": describe(delta),
            "contrasts": {
                f"biological_minus_{c}": describe(np.asarray(values["biological"]) - values[c])
                for c in ("random", "rewired")
            },
            "perturbations": {
                key: describe([mean(ev[key]) for ev in evaluations["biological"]])
                for key in ("after", "sensory_noise", "edge_ablation", "neuron_ablation")
            },
        }
        if task not in ("catch", "dodge"):
            probe = [r["state_probe"]["score"] for r in arms["biological"]]
            checks = [
                {
                    "seed": s,
                    "improved": bool(delta[s] > 0),
                    "beats_random_margin": bool(
                        values["biological"][s] > values["random"][s] + 0.05
                    ),
                    "state_sensitive": bool(probe[s] > 0.05),
                }
                for s in range(3)
            ]
            task_summary["success_checks"] = checks
            task_summary["meets_frozen_success_criteria"] = all(
                row["improved"] and row["beats_random_margin"] and row["state_sensitive"]
                for row in checks
            )
            task_summary["state_probe"] = describe(probe)
            task_summary["action_entropy"] = describe(
                [
                    np.mean([row["action_entropy"] for row in ev["after"]])
                    for ev in evaluations["biological"]
                ]
            )
        task_summary["condition_before"] = {
            c: describe([mean(r["before"]) for r in arms[c]]) for c in CONDITIONS[:3]
        }
        if task not in ("catch", "dodge"):
            numeric_keys = [
                k
                for k, v in evaluations["biological"][0]["after"][0].items()
                if isinstance(v, (int, float))
            ]
            task_summary["raw_biological"] = {
                k: describe(
                    [np.mean([row[k] for row in ev["after"]]) for ev in evaluations["biological"]]
                )
                for k in numeric_keys
            }
        summary["tasks"][task] = task_summary
        for condition, desc in task_summary["primary"].items():
            primary.append(
                {
                    "task": task,
                    "condition": condition,
                    "mean": desc["mean"],
                    "n_training_seeds": 3,
                    "seed_0": desc["seed_values"][0],
                    "seed_1": desc["seed_values"][1],
                    "seed_2": desc["seed_values"][2],
                    "descriptive_lower": desc["descriptive_bootstrap_95"][0],
                    "descriptive_upper": desc["descriptive_bootstrap_95"][1],
                }
            )
        for name, desc in {
            "change_from_before": task_summary["change"],
            **task_summary["contrasts"],
        }.items():
            contrasts.append({"task": task, "contrast": name, **desc})
        for name, desc in task_summary["perturbations"].items():
            perturbations.append(
                {
                    "task": task,
                    "condition": name,
                    **desc,
                    "change_from_intact": describe(
                        np.asarray(desc["seed_values"]) - values["biological"]
                    ),
                }
            )
    representation = [
        row
        for task in TASKS[2:]
        for row in json.loads((ROOT / f"representation-{task}.json").read_text())["rows"]
    ]
    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "results_archive.json.gz").write_bytes(
        gzip.compress(json.dumps(archive, sort_keys=True).encode(), mtime=0)
    )
    (ROOT / "representations.json").write_text(json.dumps(representation, indent=2) + "\n")
    for name, rows in [
        ("primary.csv", primary),
        ("contrasts.csv", contrasts),
        ("perturbations.csv", perturbations),
        ("task_specific.csv", raw),
        ("representations.csv", representation),
    ]:
        write_csv(name, rows)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png")
        fig.savefig(figures / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)

    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, condition in enumerate(CONDITIONS):
        vals = [summary["tasks"][t]["primary"][condition] for t in TASKS]
        positions = x + (i - 2) * 0.15
        ax.bar(positions, [v["mean"] for v in vals], width=0.14, color=COLORS[i], label=condition)
        for pos, v in zip(positions, vals, strict=True):
            ax.scatter(
                pos + np.array([-0.025, 0, 0.025]), v["seed_values"], s=13, color="black", zorder=4
            )
    ax.set(
        xticks=x,
        xticklabels=TASKS,
        ylim=(0, 1.02),
        ylabel="Task-defined normalized success",
        title="Six-task held-out performance (n = 3 independent training seeds)",
    )
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    save(fig, "six_task_performance")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, task in zip(axes.flat, TASKS, strict=True):
        for condition, color in zip(CONDITIONS[:3], COLORS[:3], strict=True):
            data = np.asarray(curves[task][condition])
            if data.shape[1] == 0:
                ax.axhline(
                    summary["tasks"][task]["primary"][condition]["mean"],
                    color=color,
                    linestyle=":",
                    label="frozen (reference)",
                )
                continue
            width = max(1, data.shape[1] // 30)
            limit = data.shape[1] // width * width
            smooth = data[:, :limit].reshape(3, -1, width).mean(axis=2)
            xx = np.arange(1, smooth.shape[1] + 1) * width
            for trace in smooth:
                ax.plot(xx, trace, color=color, alpha=0.18, lw=0.6)
            ax.plot(xx, smooth.mean(axis=0), color=color, label=condition)
        ax.set(title=task, ylim=(0, 1.02), xlabel="Training episode", ylabel="Normalized success")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training behavior; n = 3 seeds (historical Catch/Dodge)")
    save(fig, "learning_curves")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, c in enumerate(("biological", "rewired")):
        v = [summary["tasks"][t]["primary"][c] for t in TASKS]
        pos = x + (i - 0.5) * 0.3
        ax.bar(pos, [d["mean"] for d in v], width=0.28, color=COLORS[i * 2], label=c)
        for p, d in zip(pos, v, strict=True):
            ax.scatter([p] * 3, d["seed_values"], s=15, color="black")
    ax.set(
        xticks=x,
        xticklabels=TASKS,
        ylim=(0, 1.02),
        ylabel="Normalized success",
        title="Topology comparison; n = 3 independent training seeds",
    )
    ax.legend()
    save(fig, "topology")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(("after", "sensory_noise", "edge_ablation", "neuron_ablation")):
        vals = [summary["tasks"][t]["perturbations"][c]["mean"] for t in TASKS]
        ax.bar(x + (i - 1.5) * 0.19, vals, 0.18, label=c)
    ax.set(
        xticks=x,
        xticklabels=TASKS,
        ylim=(0, 1.02),
        ylabel="Normalized success",
        title="Post-training perturbations, no retraining; n = 3 seeds",
    )
    ax.legend(fontsize=8)
    save(fig, "perturbations")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    metrics = (
        "linear_probe_accuracy",
        "participation_ratio",
        "mean_pairwise_correlation",
        "population_activity_variance",
    )
    for ax, metric in zip(axes, metrics, strict=True):
        for i, c in enumerate(("biological", "rewired")):
            means = [
                np.mean(
                    [r[metric] for r in representation if r["task"] == t and r["condition"] == c]
                )
                for t in TASKS[2:]
            ]
            ax.bar(np.arange(4) + (i - 0.5) * 0.35, means, 0.33, label=c, color=COLORS[i * 2])
        ax.set(xticks=np.arange(4), xticklabels=TASKS[2:], title=metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=40)
        if metric == "linear_probe_accuracy":
            ax.set_ylim(0, 1)
            ax.scatter(
                np.arange(4),
                [
                    np.mean([r["majority_accuracy"] for r in representation if r["task"] == t])
                    for t in TASKS[2:]
                ],
                marker="_",
                color="black",
                s=100,
                label="majority baseline",
            )
        else:
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].legend(fontsize=7)
    fig.suptitle("Post-hoc matched states; n = 3 model seeds per topology")
    save(fig, "representations")
    table = [
        (
            "| Task | Biological | Frozen | Rewired | Random | Heuristic | "
            "Change | Bio−random | Bio−rewired |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, d in summary["tasks"].items():
        vals = [d["primary"][c]["mean"] for c in CONDITIONS] + [
            d["change"]["mean"],
            d["contrasts"]["biological_minus_random"]["mean"],
            d["contrasts"]["biological_minus_rewired"]["mean"],
        ]
        table.append("| " + task + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    (ROOT / "table.md").write_text("\n".join(table) + "\n")
    print(
        json.dumps(
            {
                t: {
                    "mean": d["primary"]["biological"]["mean"],
                    "passed": d.get("meets_frozen_success_criteria", "historical"),
                }
                for t, d in summary["tasks"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
