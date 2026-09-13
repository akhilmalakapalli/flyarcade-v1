"""M11/M12: regenerate every v1.3 table, summary and figure from raw result files.

Inputs: experiments/v13_run_plan.json, runs/v13-<task>-<condition>-<seed>/result.json,
artifacts/v13/representation-<task>.json, development stage summaries and the
unchanged historical v1.1/v1.2 summaries. Outputs only under artifacts/v13/.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade_v13.environments import TARGETS
from flyarcade_v13.gates import seed_summary, task_gate

ROOT = Path("artifacts/v13")
FIG = ROOT / "figures"
PLAN = Path("experiments/v13_run_plan.json")
CONDITIONS = ("biological", "frozen", "rewired", "sensory")
COLORS = {
    "biological": "#2878b5",
    "frozen": "#9a9a9a",
    "rewired": "#e58b28",
    "sensory": "#399473",
    "random": "#ad587c",
    "reference": "#5b4b8a",
}
PERTURBATIONS = ("after", "sensory_noise", "edge_ablation", "neuron_ablation")
RAW = {
    "flappy": ("obstacles_passed",),
    "pong": ("returns", "attempted_returns"),
    "breakout": ("bricks_destroyed", "interceptions"),
    "snake": ("food", "steps"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values, seed=2026, draws=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (draws, len(values))).mean(1)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "seed_values": values.tolist(),
        "descriptive_bootstrap_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def success(rows):
    return float(np.mean([r["success"] for r in rows]))


def load(plan):
    results, files = {}, {}
    for task in TARGETS:
        for condition in CONDITIONS:
            for seed in plan["training_seeds"]:
                path = Path("runs") / f"v13-{task}-{condition}-{seed}" / "result.json"
                results[(task, condition, seed)] = json.loads(path.read_text())
                files[str(path)] = digest(path)
    return results, files


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def historical_rows():
    v11 = json.loads(Path("artifacts/v11_results_summary.json").read_text())["primary"]
    v12 = json.loads(Path("artifacts/v12/results_summary.json").read_text())["tasks"]
    rows = {}
    for task in ("catch", "dodge"):
        rows[task] = {
            "biological": v11[f"{task}/eprop"]["after"],
            "rewired": v11.get(f"{task}/rewired", {}).get("after"),
            "random": v11[f"{task}/eprop"]["random"],
            "reference": v11[f"{task}/eprop"]["heuristic"],
            "source": "artifacts/v11_results_summary.json (v1.1 linear actor-critic, unchanged)",
        }
    v12_rows = {t: {k: v12[t]["primary"][k]["mean"] for k in v12[t]["primary"]} for t in TARGETS}
    return rows, v12_rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = json.loads(PLAN.read_text())
    results, files = load(plan)
    seeds = plan["training_seeds"]
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {"plan_sha256": digest(PLAN), "result_files": files, "tasks": {}}
    primary, contrasts, perturb, specific = [], [], [], []
    for task in TARGETS:
        bio = [results[(task, "biological", s)] for s in seeds]
        gate = task_gate(bio)
        entry = {"architecture": plan["tasks"][task]["architecture"], "criteria": gate, "primary": {}}
        for condition in CONDITIONS:
            rows = [results[(task, condition, s)] for s in seeds]
            entry["primary"][condition] = describe([r["means"]["after"] for r in rows])
            entry["primary"][f"{condition}_before"] = describe([r["means"]["before"] for r in rows])
            entry["primary"][f"{condition}_state_probe"] = describe([r["state_probe"]["score"] for r in rows])
            for s, r in zip(seeds, rows, strict=True):
                summ = seed_summary(r)
                primary.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed": s,
                        "before": summ["before"],
                        "after": summ["after"],
                        "random": summ["random"],
                        "reference": summ["reference"],
                        "gap_closure": summ["gap_closure"],
                        "state_probe": summ["state_probe"],
                        "final_entropy": summ["final_entropy"],
                        "max_greedy_action_fraction": summ["max_greedy_action_fraction"],
                        "transitions": r["transitions"],
                        "episodes": r["episodes"],
                        "train_wall_seconds": r["train_wall_seconds"],
                    }
                )
                for key in RAW[task]:
                    specific.append(
                        {
                            "task": task,
                            "condition": condition,
                            "seed": s,
                            "metric": key,
                            "after_mean": float(np.mean([e[key] for e in r["evaluations"]["after"]])),
                            "before_mean": float(np.mean([e[key] for e in r["evaluations"]["before"]])),
                            "random_mean": float(np.mean([e[key] for e in r["evaluations"]["random"]])),
                            "reference_mean": float(np.mean([e[key] for e in r["evaluations"]["reference"]])),
                        }
                    )
        for key in ("random", "reference"):
            entry["primary"][key] = describe([results[(task, "biological", s)]["means"][key] for s in seeds])
        if task == "flappy":
            entry["primary"]["heuristic_v12"] = describe(
                [results[(task, "biological", s)]["means"]["heuristic_v12"] for s in seeds]
            )
        pairs = {
            "biological_minus_random": ("biological", "random"),
            "biological_minus_frozen": ("biological", "frozen"),
            "biological_minus_rewired": ("biological", "rewired"),
            "biological_minus_sensory": ("biological", "sensory"),
            "rewired_minus_random": ("rewired", "random"),
        }
        entry["contrasts"] = {}
        for name, (a, b) in pairs.items():
            values = []
            for s in seeds:
                left = results[(task, a, s)]["means"]["after"]
                right = results[(task, "biological", s)]["means"][b] if b == "random" else results[(task, b, s)]["means"]["after"]
                values.append(left - right)
            entry["contrasts"][name] = describe(values)
            contrasts.append({"task": task, "contrast": name, **entry["contrasts"][name]})
        entry["perturbations"] = {}
        for key in PERTURBATIONS:
            values = [results[(task, "biological", s)]["means"][key] for s in seeds]
            entry["perturbations"][key] = describe(values)
            if key != "after":
                deltas = [results[(task, "biological", s)]["means"][key] - results[(task, "biological", s)]["means"]["after"] for s in seeds]
                entry["perturbations"][f"{key}_minus_clean"] = describe(deltas)
                perturb.append({"task": task, "perturbation": key, "mean": float(np.mean(values)), "delta_mean": float(np.mean(deltas)), "seed_values": values})
        entry["level_transitions"] = [results[(task, "biological", s)]["level_transitions"] for s in seeds]
        summary["tasks"][task] = entry
    write_csv("primary.csv", primary)
    write_csv("contrasts.csv", contrasts)
    write_csv("perturbations.csv", perturb)
    write_csv("task_specific.csv", specific)

    # representations
    rep_rows = []
    for task in TARGETS:
        path = ROOT / f"representation-{task}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for key, row in report["rows"].items():
            rep_rows.append(
                {
                    "task": task,
                    "row": key,
                    **{
                        k: row.get(k)
                        for k in (
                            "topology",
                            "ticks",
                            "participation_ratio",
                            "mean_pairwise_correlation",
                            "mean_activity_variance",
                            "majority_accuracy_test",
                            "majority_balanced_accuracy",
                            "linear_action_balanced_accuracy",
                            "mlp_action_balanced_accuracy",
                            "linear_state_r2_mean",
                            "mlp_state_r2_mean",
                            "class_counts_train",
                            "class_counts_test",
                            "seconds_per_state",
                        )
                    },
                    "state_sha256": report["state_sha256"],
                }
            )
    write_csv("representations.csv", rep_rows)
    historical, v12 = historical_rows()
    summary["historical"] = historical
    summary["v12_final_means"] = v12
    summary["uncertainty"] = "bootstrap over 5 training-seed means, 10,000 draws, seed 2026; descriptive only, no p-values"
    (ROOT / "results_summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=180 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)

    # 1 six-task performance
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.8), sharey=True)
    for ax, task in zip(axes, ("catch", "dodge", *TARGETS), strict=True):
        if task in historical:
            h = historical[task]
            bars = {k: h[k]["mean"] for k in ("biological", "random", "reference") if h.get(k)}
            if h.get("rewired"):
                bars["rewired"] = h["rewired"]["mean"]
            points = {k: h[k]["seed_values"] for k in bars if h.get(k)}
            ax.set_title(f"{task} (v1.1 linear)")
        else:
            p = summary["tasks"][task]["primary"]
            order = ("biological", "frozen", "rewired", "sensory", "random", "reference")
            bars = {k: p[k]["mean"] for k in order}
            points = {k: p[k]["seed_values"] for k in order}
            ax.set_title(f"{task} (v1.3 {summary['tasks'][task]['architecture']})")
        for i, (k, v) in enumerate(bars.items()):
            ax.bar(i, v, color=COLORS[k], label=k)
            ax.scatter(np.full(len(points[k]), i) + np.linspace(-0.2, 0.2, len(points[k])), points[k], s=9, color="black", zorder=3)
        ax.set_xticks(range(len(bars)), list(bars), rotation=60, fontsize=7)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("held-out success (0-1)")
    save(fig, "six_task_performance")

    # 2 learning curves (transitions)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, task in zip(axes, TARGETS, strict=True):
        for condition in ("biological", "rewired", "sensory"):
            for s in seeds:
                curve = results[(task, condition, s)]["curve"]
                if curve:
                    ax.plot([c["transitions"] for c in curve], [c["success"] for c in curve], color=COLORS[condition], alpha=0.5, lw=1, label=condition if s == seeds[0] else None)
        ax.set_title(task)
        ax.set_xlabel("environment transitions")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("curve-seed greedy success")
    axes[0].legend(fontsize=7)
    save(fig, "learning_curves")

    # 3 capacity ladder
    ladder_path = ROOT / "development" / "capacity_ladder.json"
    if ladder_path.exists():
        ladder = json.loads(ladder_path.read_text())
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
        for ax, task in zip(axes, TARGETS, strict=True):
            rungs = ladder[task]
            ax.bar(range(len(rungs)), [r["mean_after"] for r in rungs], color="#2878b5")
            ax.axhline(rungs[0]["random"], color=COLORS["random"], ls="--", lw=1, label="random")
            ax.set_xticks(range(len(rungs)), [r["label"] for r in rungs], rotation=45, fontsize=7)
            ax.set_title(task)
            ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("success (biological core)")
        axes[0].legend(fontsize=7)
        save(fig, "capacity_ladder")

    # 4 biological vs rewired
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, task in enumerate(TARGETS):
        p = summary["tasks"][task]["primary"]
        for j, condition in enumerate(("biological", "rewired")):
            ax.bar(i + (j - 0.5) * 0.35, p[condition]["mean"], 0.35, color=COLORS[condition], label=condition if i == 0 else None)
            ax.scatter(np.full(len(seeds), i + (j - 0.5) * 0.35), p[condition]["seed_values"], s=8, color="black", zorder=3)
    ax.set_xticks(range(4), TARGETS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out success")
    ax.legend(fontsize=8)
    save(fig, "topology")

    # 5 perturbations
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, task in enumerate(TARGETS):
        pr = summary["tasks"][task]["perturbations"]
        for j, key in enumerate(PERTURBATIONS):
            ax.bar(i + (j - 1.5) * 0.2, pr[key]["mean"], 0.2, color=plt.cm.viridis(j / 3), label=key if i == 0 else None)
    ax.set_xticks(range(4), TARGETS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("biological held-out success")
    ax.legend(fontsize=7)
    save(fig, "perturbations")

    # 6 representation diagnostics
    if rep_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        for task in TARGETS:
            rows = sorted((r for r in rep_rows if r["task"] == task and r["topology"] == "biological"), key=lambda r: r["ticks"])
            if not rows:
                continue
            ticks = [r["ticks"] for r in rows]
            axes[0].plot(ticks, [r["linear_state_r2_mean"] for r in rows], marker="o", label=task)
            axes[1].plot(ticks, [r["mlp_action_balanced_accuracy"] - r["majority_balanced_accuracy"] for r in rows], marker="o", label=task)
            axes[2].plot(ticks, [r["participation_ratio"] for r in rows], marker="o", label=task)
        axes[0].set_ylabel("held-out linear state R²")
        axes[1].set_ylabel("MLP balanced action acc. − majority")
        axes[2].set_ylabel("participation ratio")
        for ax in axes:
            ax.set_xlabel("LIF ticks per action")
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, None)
        axes[2].set_ylim(0, None)
        axes[0].legend(fontsize=7)
        save(fig, "representations")
    summary_hash = digest(ROOT / "results_summary.json")
    print(json.dumps({t: {"passed": e["criteria"]["passed"], **{k: round(v["mean"], 3) for k, v in e["primary"].items() if not k.endswith(("_before", "_probe"))}} for t, e in summary["tasks"].items()}, indent=1))
    print("summary", summary_hash)


if __name__ == "__main__":
    main()
