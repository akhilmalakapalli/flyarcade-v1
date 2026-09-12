"""Export manuscript CSV tables and the robustness/transfer and policy figures.

Reads only artifacts already produced by summarize_results.py and diagnostics.py.
Performs no training, no model selection and no re-analysis choices that depend on
held-out outcomes.
"""

import csv
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("learning", "frozen", "rewired", "readout_only")
TASKS = ("catch", "dodge")
# Validated categorical slots 1-4, fixed order, never cycled.
COLORS = {
    "learning": "#2a78d6",
    "frozen": "#eb6834",
    "rewired": "#1baf7a",
    "readout_only": "#eda100",
}
PERTURBATIONS = ("after", "sensory_noise_0.1", "edge_ablation_0.1", "neuron_ablation_0.1")


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path("artifacts")
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    summary = json.loads((root / "results_summary.json").read_text())
    metrics = json.loads((root / "trial_metrics.json").read_text())
    diagnostics = {t: json.loads((root / f"diagnostics_{t}.json").read_text()) for t in TASKS}

    rows = []
    for task in TASKS:
        for condition in CONDITIONS:
            item = summary["primary"][f"{task}/{condition}"]
            low, high = item["change"]["descriptive_bootstrap_95"]
            rows.append(
                [
                    task,
                    condition,
                    f"{item['before']['mean']:.4f}",
                    f"{item['after']['mean']:.4f}",
                    f"{item['change']['mean']:.4f}",
                    f"{low:.4f}",
                    f"{high:.4f}",
                    *[f"{v:.4f}" for v in item["change"]["seed_values"]],
                    f"{item['random']['mean']:.4f}",
                    f"{item['heuristic']['mean']:.4f}",
                ]
            )
    write_csv(
        tables / "primary_heldout.csv",
        [
            "task",
            "condition",
            "heldout_before",
            "heldout_after",
            "change_mean",
            "change_boot_lo",
            "change_boot_hi",
            "change_seed0",
            "change_seed1",
            "change_seed2",
            "random_baseline",
            "heuristic_baseline",
        ],
        rows,
    )

    write_csv(
        tables / "contrasts.csv",
        ["contrast", "mean", "boot_lo", "boot_hi", "seed0", "seed1", "seed2"],
        [
            [
                name,
                f"{item['mean']:.4f}",
                f"{item['descriptive_bootstrap_95'][0]:.4f}",
                f"{item['descriptive_bootstrap_95'][1]:.4f}",
                *[f"{v:.4f}" for v in item["seed_values"]],
            ]
            for name, item in summary["contrasts"].items()
        ],
    )

    write_csv(
        tables / "robustness.csv",
        ["task", "perturbation", "mean", "boot_lo", "boot_hi", "seed0", "seed1", "seed2"],
        [
            [
                task,
                key,
                f"{item['mean']:.4f}",
                f"{item['descriptive_bootstrap_95'][0]:.4f}",
                f"{item['descriptive_bootstrap_95'][1]:.4f}",
                *[f"{v:.4f}" for v in item["seed_values"]],
            ]
            for task, block in summary["robustness"].items()
            for key, item in block.items()
        ],
    )

    write_csv(
        tables / "transfer.csv",
        ["direction", "arm", "mean", "boot_lo", "boot_hi", "seed0", "seed1", "seed2"],
        [
            [
                direction,
                arm,
                f"{item['mean']:.4f}",
                f"{item['descriptive_bootstrap_95'][0]:.4f}",
                f"{item['descriptive_bootstrap_95'][1]:.4f}",
                *[f"{v:.4f}" for v in item["seed_values"]],
            ]
            for direction, block in summary["transfer"].items()
            for arm, item in block.items()
        ],
    )

    write_csv(
        tables / "trial_index.csv",
        [
            "trial",
            "task",
            "condition",
            "seed",
            "episodes",
            "source",
            *PERTURBATIONS,
            "before",
            "random",
            "heuristic",
            "elapsed_seconds",
            "rss_mb",
        ],
        [
            [
                name,
                row["config"]["task"],
                row["config"]["condition"],
                row["config"]["seed"],
                row["config"]["episodes"],
                row["config"]["source"] or "",
                *[f"{np.mean(row['evaluation'][key]):.4f}" for key in PERTURBATIONS],
                f"{np.mean(row['evaluation']['before']):.4f}",
                f"{np.mean(row['evaluation']['random']):.4f}",
                f"{np.mean(row['evaluation']['heuristic']):.4f}",
                f"{row['resources']['elapsed_seconds']:.2f}",
                f"{row['resources']['rss_mb']:.1f}",
            ]
            for name, row in metrics.items()
        ],
    )

    write_csv(
        tables / "diagnostics.csv",
        [
            "task",
            "condition",
            "state_distribution",
            "mean_policy_entropy_nats",
            "max_policy_entropy_nats",
            "mean_max_action_probability",
            "action_left",
            "action_stay",
            "action_right",
            "probe_accuracy",
            "probe_shuffled_control",
            "label_majority_rate",
        ],
        [
            [
                task,
                row["condition"],
                row["state_distribution"],
                f"{row['mean_policy_entropy_nats']:.4f}",
                f"{row['max_policy_entropy_nats']:.4f}",
                f"{row['mean_max_action_probability']:.4f}",
                *row["action_usage"],
                f"{row['probe_accuracy_descending_rates']:.4f}",
                f"{row['probe_accuracy_shuffled_control']:.4f}",
                f"{row['label_majority_rate']:.4f}",
            ]
            for task, report in diagnostics.items()
            for row in report["conditions"]
        ],
    )

    write_csv(
        tables / "constant_action_reference.csv",
        ["task", "policy", "heldout_success"],
        [
            [task, policy, f"{value:.4f}"]
            for task, report in diagnostics.items()
            for policy, value in report["constant_action_reference"].items()
        ],
    )

    # Figure: perturbation robustness and transfer, per training seed.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = ["unperturbed", "sensory noise", "edge ablation", "neuron ablation"]
    for ax, task in zip(axes, TASKS, strict=True):
        for x, key in enumerate(PERTURBATIONS):
            item = summary["robustness"][task][key]
            ax.bar(x, item["mean"], color=COLORS["learning"], width=0.62)
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
            summary["primary"][f"{task}/learning"]["random"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random baseline",
        )
        ax.set(title=task, ylim=(0, 1.18), xticks=range(4), xticklabels=labels)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("Held-out success; points = training seeds")
    fig.suptitle("Perturbation robustness of trained controllers (n=3 seeds)")
    fig.tight_layout()
    fig.savefig(root / "robustness.svg")
    fig.savefig(root / "robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    arms = ["transfer", "target200", "target100"]
    arm_labels = ["100 source + 100 target", "200 target only", "100 target only"]
    for ax, (direction, block) in zip(axes, summary["transfer"].items(), strict=True):
        for x, arm in enumerate(arms):
            ax.bar(x, block[arm]["mean"], color=COLORS["rewired"], width=0.62)
            ax.scatter(
                [x - 0.08, x, x + 0.08], block[arm]["seed_values"], color="#0b0b0b", s=18, zorder=3
            )
            ax.text(
                x,
                max(block[arm]["mean"], max(block[arm]["seed_values"])) + 0.035,
                f"{block[arm]['mean']:.2f}",
                ha="center",
                fontsize=8,
                color="#52514e",
            )
        task = direction.split("-to-")[1]
        ax.axhline(
            summary["primary"][f"{task}/learning"]["random"]["mean"],
            color="#52514e",
            ls="--",
            lw=1.2,
            label="random baseline",
        )
        ax.set(title=direction, ylim=(0, 1.18), xticks=range(3), xticklabels=arm_labels)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("Held-out success; points = training seeds")
    fig.suptitle("Equal-budget transfer comparisons (n=3 seeds)")
    fig.tight_layout()
    fig.savefig(root / "transfer.svg")
    fig.savefig(root / "transfer.png", dpi=180)
    plt.close(fig)

    # Figure: policy collapse and retained decodable signal, matched state distribution.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    matched = {
        task: {row["condition"]: row for row in report["conditions"]}
        for task, report in diagnostics.items()
    }
    keys = ["initial_weights_matched_states", "trained_weights_matched_states"]
    key_labels = ["initial", "after training"]
    for x, task in enumerate(TASKS):
        offsets = np.arange(2) + x * 2.6
        values = [matched[task][k]["mean_policy_entropy_nats"] for k in keys]
        axes[0].bar(offsets, values, color=[COLORS["frozen"], COLORS["learning"]], width=0.62)
        for o, v in zip(offsets, values, strict=True):
            axes[0].text(o, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    axes[0].axhline(float(np.log(3)), color="#52514e", ls="--", lw=1.2, label="uniform policy")
    axes[0].set(
        title="Policy entropy collapses",
        ylabel="Mean action entropy (nats)",
        xticks=[0, 1, 2.6, 3.6],
        xticklabels=[f"{k}\n{t}" for t in TASKS for k in key_labels],
        ylim=(0, 1.35),
    )
    axes[0].legend(fontsize=8, frameon=False)

    for x, task in enumerate(TASKS):
        offsets = np.arange(2) + x * 2.6
        values = [matched[task][k]["mean_max_action_probability"] for k in keys]
        axes[1].bar(offsets, values, color=[COLORS["frozen"], COLORS["learning"]], width=0.62)
        for o, v in zip(offsets, values, strict=True):
            axes[1].text(o, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    axes[1].axhline(1 / 3, color="#52514e", ls="--", lw=1.2, label="uniform policy")
    axes[1].set(
        title="Preferred action dominates",
        ylabel="Mean probability of the most likely action",
        xticks=[0, 1, 2.6, 3.6],
        xticklabels=[f"{k}\n{t}" for t in TASKS for k in key_labels],
        ylim=(0, 1.15),
    )
    axes[1].legend(fontsize=8, frameon=False)

    for x, task in enumerate(TASKS):
        offsets = np.arange(2) + x * 2.6
        values = [matched[task][k]["probe_accuracy_descending_rates"] for k in keys]
        axes[2].bar(offsets, values, color=[COLORS["frozen"], COLORS["learning"]], width=0.62)
        for o, v in zip(offsets, values, strict=True):
            axes[2].text(o, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    axes[2].axhline(
        matched["catch"]["initial_weights_matched_states"]["label_majority_rate"],
        color="#52514e",
        ls="--",
        lw=1.2,
        label="majority-class rate",
    )
    axes[2].set(
        title="Decodable signal survives",
        ylabel="Analyst probe accuracy from descending rates",
        xticks=[0, 1, 2.6, 3.6],
        xticklabels=[f"{k}\n{t}" for t in TASKS for k in key_labels],
        ylim=(0, 1.15),
    )
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Why learning fails: the readout collapses while task information remains "
        "(matched state distribution, seed 0)"
    )
    fig.tight_layout()
    fig.savefig(root / "policy_collapse.svg")
    fig.savefig(root / "policy_collapse.png", dpi=180)
    plt.close(fig)

    # Figure: measured properties of the acquired biological subgraph.
    audit = json.loads((root / "malecns_audit.json").read_text())
    write_csv(
        tables / "connectome_summary.csv",
        ["property", "value"],
        [
            ["neurons", audit["neurons"]],
            ["directed_edges", audit["edges"]],
            ["synapses", audit["synapses"]],
            ["isolates", audit["isolates"]],
            *[[f"superclass_{k}", v] for k, v in audit["classes"].items()],
            *[[f"consensus_nt_{k}", v] for k, v in audit["consensus_nt"].items()],
            ["reachable_descending", audit["reachable_descending_count"]],
            *[[f"descending_at_{k}_hops", v] for k, v in audit["descending_shortest_hops"].items()],
            ["incoming_weight_retained", f"{audit['incoming_weight_retained_fraction']:.4f}"],
            ["outgoing_weight_retained", f"{audit['outgoing_weight_retained_fraction']:.4f}"],
        ],
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    classes = audit["classes"]
    axes[0].bar(
        range(len(classes)),
        list(classes.values()),
        color=[COLORS["learning"], COLORS["frozen"], COLORS["rewired"]],
        width=0.62,
    )
    for i, v in enumerate(classes.values()):
        axes[0].text(i, v + 25, str(v), ha="center", fontsize=8, color="#52514e")
    axes[0].set(
        title="Selected neurons by superclass",
        ylabel="Neurons",
        xticks=range(len(classes)),
        xticklabels=[k.replace("_", "\n") for k in classes],
        ylim=(0, max(classes.values()) * 1.18),
    )
    nt = audit["consensus_nt"]
    axes[1].bar(range(len(nt)), list(nt.values()), color=COLORS["learning"], width=0.62)
    for i, v in enumerate(nt.values()):
        axes[1].text(i, v + 20, str(v), ha="center", fontsize=8, color="#52514e")
    axes[1].set(
        title="Consensus neurotransmitter annotations",
        ylabel="Neurons",
        xticks=range(len(nt)),
        xticklabels=list(nt),
        ylim=(0, max(nt.values()) * 1.18),
    )
    axes[1].tick_params(axis="x", rotation=40)
    hops = audit["descending_shortest_hops"]
    axes[2].bar(range(len(hops)), list(hops.values()), color=COLORS["rewired"], width=0.62)
    for i, v in enumerate(hops.values()):
        axes[2].text(i, v + 12, str(v), ha="center", fontsize=8, color="#52514e")
    axes[2].set(
        title="Descending neurons by shortest path\nfrom any visual-projection neuron",
        ylabel="Descending neurons",
        xlabel="Directed hops",
        xticks=range(len(hops)),
        xticklabels=list(hops),
        ylim=(0, max(hops.values()) * 1.18),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Measured MaleCNS v1.0 subgraph: 2,040 neurons, 126,676 directed edges, no isolates"
    )
    fig.tight_layout()
    fig.savefig(root / "connectome.svg")
    fig.savefig(root / "connectome.png", dpi=180)
    plt.close(fig)

    written = sorted(str(p) for p in tables.glob("*.csv"))
    print(
        json.dumps(
            {"tables": written, "figures": sorted(str(p) for p in root.glob("*.png"))}, indent=2
        )
    )


if __name__ == "__main__":
    main()
