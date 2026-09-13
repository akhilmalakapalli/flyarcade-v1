"""Generate development/validation reports without reading historical trial episodes."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from flyarcade_v14.selection import select
from flyarcade_v14.study import digest

ROOT = Path("artifacts/v14")


def historical():
    files = {
        "lanes": "artifacts/v11_results_summary.json",
        "snake": "artifacts/snake_results_summary.json",
        "pong": "artifacts/v13/pong/confirmatory_summary.json",
        "flappy": "artifacts/v13/flappy-rescue/confirmatory_summary.json",
    }
    data = {k: json.loads(Path(p).read_text()) for k, p in files.items()}
    scores = {
        t: data["lanes"]["primary"][f"{t}/eprop"]["after"]["mean"] for t in ("catch", "dodge")
    }
    scores["snake"] = data["snake"]["primary"]["eprop"]["after_food"]["mean"]
    scores.update({t: data[t]["biological_after"] for t in ("pong", "flappy")})
    return scores, {p: digest(p) for p in files.values()}


def describe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"seed_values": values.tolist(), "mean": None, "descriptive_95": None}
    rng = np.random.default_rng(1400)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "seed_values": values.tolist(),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "descriptive_95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def csv_file(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
            for r in rows
        )


def main():
    plan = json.loads(Path("experiments/v14/development_plan.json").read_text())
    history, history_files = historical()
    configurations = [
        json.loads(p.read_text()) for p in sorted((ROOT / "configurations").glob("*.json"))
    ]
    selected = (
        json.loads((ROOT / "selected_configs.json").read_text())
        if (ROOT / "selected_configs.json").exists()
        else None
    )
    validation = (
        json.loads((ROOT / "validation_summary.json").read_text())
        if (ROOT / "validation_summary.json").exists()
        else None
    )
    results = {
        str(p): json.loads(p.read_text()) for p in sorted(Path("runs/v14").glob("*/result.json"))
    }
    rows = []
    for path, r in results.items():
        c = r["config"]
        rows.append(
            {
                "task": c["task"],
                "phase": r["phase"],
                "learner": c["arch"],
                "seed": c["seed"],
                "hidden": c["hidden"],
                "second_hidden_or_gru": c["core"],
                "gru_size": c["core"] if c["arch"] == "gru" else 0,
                "optimizer": r["optimizer"],
                "learning_rate": c["lr"],
                "lr_schedule": r["lr_schedule"],
                "gamma": c["gamma"],
                "gae_lambda": c["lam"],
                "ppo_clip": c["clip"],
                "entropy_coefficient": c["ent"],
                "value_coefficient": c["vf"],
                "ppo_epochs": c["epochs"],
                "minibatch_count": c["minibatches"],
                "parallel_envs": c["envs"],
                "ticks": c["ticks"],
                "readout": c["readout"],
                "training_transitions": r["training_transitions"],
                "imitation_transitions": r["imitation"]["transitions"] if r["imitation"] else 0,
                "score": r["score"],
                "state_dependence": r["state_probe"]["score"],
                "action_distribution": r["action_distribution"],
                "policy_entropy": r["policy_entropy"],
                "finite": r["finite"],
                "wall_seconds": r["wall_clock_training_seconds"],
                "peak_memory_mb": r["peak_memory_mb"],
                "checkpoint": r["checkpoint_path"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "result_sha256": digest(path),
                "failure_reason": r["failure_reason"],
                "last_update": r["training_history"][-1],
            }
        )
    failures = [json.loads(p.read_text()) for p in sorted(ROOT.glob("*-failure.json"))]
    summary = {
        "status": "VALIDATION_COMPLETE" if validation else "IN_PROGRESS",
        "plan_sha256": digest("experiments/v14/development_plan.json"),
        "historical_scores_reference_only": history,
        "historical_summary_hashes": history_files,
        "configurations": configurations,
        "tasks": {},
        "failures": failures,
        "result_files": {p: digest(p) for p in results},
        "confirmatory_testing": False,
    }
    contrasts = []
    factors = []
    for task in plan["tasks"]:
        primary = {
            r["config"]["arch"]: r
            for r in configurations
            if r["config"]["task"] == task and r["phase"] == "primary"
        }
        entry = {
            "primary": primary,
            "historical_reference": history[task],
            "target": plan["development_targets"][task],
        }
        if len(primary) == 3:
            entry["primary_winner"] = select(list(primary.values()))
            for arch in ("mlp", "gru"):
                a, b = primary[arch], primary["linear"]
                if a["complete"] and b["complete"]:
                    contrast = describe(np.asarray(a["scores"]) - b["scores"])
                    contrasts.append({"task": task, "contrast": f"{arch}-minus-linear", **contrast})
        if selected:
            info = selected["tasks"][task]
            entry["selection"] = info
            previous = []
            for record in info["configurations"]:
                if record["phase"] != "primary":
                    parent = select(previous)
                    if record["phase"] in ("curriculum", "rescue"):
                        # Compare with a matching target-only budget, not a shorter winner.
                        matches = [
                            r
                            for r in previous
                            if r["phase"] != "rescue"
                            and r["config"].get("curriculum", [["target", 1.0]])
                            == [["target", 1.0]]
                            and all(
                                r["config"][k] == record["config"][k]
                                for k in ("arch", "ticks", "readout", "transitions")
                            )
                        ]
                        if matches:
                            parent = matches[0]
                    effect = {
                        "task": task,
                        "factor": record["phase"],
                        "configuration": record["name"],
                        "parent": parent["name"] if parent else None,
                        "eligible": record["eligible"],
                        "mean": record["mean"],
                        "matched_budget": bool(
                            parent
                            and parent["config"]["transitions"] == record["config"]["transitions"]
                        ),
                    }
                    if parent and record["complete"] and parent["complete"]:
                        effect["effect"] = describe(np.asarray(record["scores"]) - parent["scores"])
                    factors.append(effect)
                previous.append(record)
        if validation:
            entry["validation"] = validation["tasks"][task]
            val = entry["validation"]["final"]
            if val["status"] == "COMPLETE":
                entry["target_met"] = bool(val["mean"] >= entry["target"])
                entry["validation_description"] = describe(val["seed_values"])
                entry["state_sensitive_all_seeds"] = all(v > 0.05 for v in val["state_probes"])
                entry["no_constant_collapse"] = all(
                    v < 0.98 for v in val["dominant_action_fractions"]
                )
                entry["status"] = (
                    "TARGET_MET"
                    if entry["target_met"]
                    and entry["state_sensitive_all_seeds"]
                    and entry["no_constant_collapse"]
                    else "TARGET_SCORE_MET_GATE_FAILED"
                    if entry["target_met"]
                    else "BELOW_TARGET"
                )
        summary["tasks"][task] = entry
    if validation:
        summary["interpretation"] = {
            "primary_winners": {
                t: d["primary_winner"]["config"]["arch"]
                for t, d in summary["tasks"].items()
                if d.get("primary_winner")
            },
            "reference_differences": {
                t: d["validation"]["final"]["mean"] - history[t]
                for t, d in summary["tasks"].items()
                if d["validation"]["final"]["status"] == "COMPLETE"
            },
            "meaningful_reference_gains": [
                t
                for t, d in summary["tasks"].items()
                if d["validation"]["final"].get("mean", -1) - history[t]
                >= plan["meaningful_improvement"][t]
            ],
            "reference_caveat": (
                "Unmatched historical differences are descriptive, not causal effects."
            ),
            "gru_consistently_best": all(
                d.get("primary_winner") and d["primary_winner"]["config"]["arch"] == "gru"
                for d in summary["tasks"].values()
            ),
            "rescue_tasks": sorted({r["task"] for r in rows if r["phase"] == "rescue"}),
            "substrate_bottleneck": "Not established: no matched sensory-only comparator; "
            "limited optimization, encoding, integration and readout remain alternatives.",
            "primary_question": "Architecture-dependent development evidence at a common PPO "
            "operating point; not confirmation of a universally best learner.",
        }
    summary["primary_contrasts"] = contrasts
    summary["secondary_effects"] = factors
    (ROOT / "development_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "development_results_archive.json.gz").write_bytes(
        gzip.compress(json.dumps(results, sort_keys=True).encode(), mtime=0)
    )
    csv_file("runs.csv", rows)
    csv_file("failures.csv", failures)
    csv_file(
        "primary_comparison.csv",
        [
            {
                "task": r["config"]["task"],
                "learner": r["config"]["arch"],
                "mean": r["mean"],
                "sd": r["sd"],
                "variance": r["variance"],
                "seed_values": r["scores"],
                "eligible": r["eligible"],
                "utility": r["utility"],
            }
            for r in configurations
            if r["phase"] == "primary"
        ],
    )
    csv_file("primary_contrasts.csv", contrasts)
    csv_file("secondary_factors.csv", factors)
    csv_file("rescue_results.csv", [r for r in rows if r["phase"] == "rescue"])
    if validation:
        csv_file(
            "validation.csv",
            [
                {"task": t, "role": kind, **record}
                for t, d in validation["tasks"].items()
                for kind, record in d.items()
            ],
        )
    table = [
        (
            "| Task | Historical | Linear dev | MLP-PPO dev | GRU-PPO dev | "
            "Selected | Validation | Target | Ticks | Readout | Status |"
        ),
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for task, d in summary["tasks"].items():
        scores = [
            f"{d['primary'][a]['mean']:.3f}"
            if a in d["primary"] and d["primary"][a]["mean"] is not None
            else "pending"
            for a in ("linear", "mlp", "gru")
        ]
        chosen = d.get("selection", {}).get("final_selected")
        val = d.get("validation", {}).get("final", {})
        validation_mean = f"{val['mean']:.3f}" if val.get("mean") is not None else "pending"
        selected_name = f"{chosen['config']['arch']} ({chosen['phase']})" if chosen else "pending"
        table.append(
            "| "
            + " | ".join(
                [
                    task,
                    f"{history[task]:.3f}",
                    *scores,
                    selected_name,
                    validation_mean,
                    str(d["target"]),
                    str(chosen["config"]["ticks"]) if chosen else "—",
                    chosen["config"]["readout"] if chosen else "—",
                    d.get("status", "IN_PROGRESS"),
                ]
            )
            + " |"
        )
    text = "\n".join(table) + "\n"
    (ROOT / "progress.md").write_text(
        "# v1.4 development progress\n\n"
        + text
        + "\nHistorical scores are reference-only and use different budgets/learners. "
        "All v1.4 primary arms share the same transitions and fixed substrate. "
        "Snake uses raw food; other scores are fractions. No confirmatory testing.\n"
    )
    if validation:
        (ROOT / "final_summary.md").write_text(
            "# v1.4 development and fresh validation\n\n"
            + text
            + "\nUncertainty is descriptive across three trained seeds. Primary comparisons "
            "isolate readout architecture under one common PPO configuration; they do not "
            "rank fully optimized architectures. Secondary factors are conditional on "
            "development winners. Historical results are references, not matched controls. "
            "No repeated tuning against validation or new confirmatory testing occurred.\n"
        )
        report_path = ROOT / "final_summary.md"
        extra = [
            "\n## Interpretation\n",
            "Primary winners: "
            + ", ".join(
                t + "=" + a for t, a in summary["interpretation"]["primary_winners"].items()
            )
            + ".\n",
            "Meaningful numerical gains over historical references: "
            + (", ".join(summary["interpretation"]["meaningful_reference_gains"]) or "none")
            + ". These differences are descriptive, not controlled historical contrasts.\n",
            "GRU consistently best across tasks: "
            + str(summary["interpretation"]["gru_consistently_best"])
            + ".\n",
            "Rescue attempts: "
            + (", ".join(summary["interpretation"]["rescue_tasks"]) or "none")
            + ".\n",
            "## Conditional secondary effects\n",
            "| Task | Factor | Compared with | Mean change | Same PPO budget | Eligible |",
            "|---|---|---|---:|---|---|",
        ]
        for factor in factors:
            delta = factor.get("effect", {}).get("mean")
            extra.append(
                "| "
                + " | ".join(
                    [
                        factor["task"],
                        factor["factor"],
                        factor["parent"] or "none",
                        f"{delta:+.3f}" if delta is not None else "unavailable",
                        str(factor["matched_budget"]),
                        str(factor["eligible"]),
                    ]
                )
                + " |"
            )
        extra += [
            "\nThese are predeclared conditional comparisons on development data. "
            "Rescue additionally uses teacher transitions; it is not a primary architecture arm.\n",
            summary["interpretation"]["substrate_bottleneck"] + "\n",
            summary["interpretation"]["primary_question"] + "\n",
        ]
        report_path.write_text(report_path.read_text() + "\n".join(extra))
        figures(summary, results)
    print(text, flush=True)


def figures(summary, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "flyarcade-v14", "savefig.dpi": 180}
    )
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (task, d) in zip(axes, summary["tasks"].items(), strict=True):
        for i, arch in enumerate(("linear", "mlp", "gru")):
            row = d["primary"].get(arch)
            if not row or row["mean"] is None:
                continue
            ax.bar(i, row["mean"], color=("#2878b5", "#e58b28", "#399473")[i])
            ax.scatter(np.full(len(row["scores"]), i), row["scores"], color="black", s=15)
        ax.set(
            title=task,
            xticks=range(3),
            xticklabels=["Linear", "MLP", "GRU"],
            ylabel="Food" if task == "snake" else "Success",
        )
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0, top=max(4.0, ax.get_ylim()[1]) if task == "snake" else 1.02)
    fig.suptitle("Primary development: same fixed SNN, PPO and transition budget; n = 3 seeds")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"primary_comparison.{suffix}",
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (task, d) in zip(axes, summary["tasks"].items(), strict=True):
        val = d["validation"]["final"]
        if val["status"] == "COMPLETE":
            ax.bar(0, val["mean"], color="#2878b5")
            ax.scatter([0] * 3, val["seed_values"], color="black", s=20)
        ax.axhline(d["target"], ls="--", color="#b04a48", label="development target")
        ax.set(
            title=task,
            xticks=[0],
            xticklabels=["Fresh validation"],
            ylabel="Food" if task == "snake" else "Success",
        )
        ax.set_ylim(bottom=0, top=max(4.5, ax.get_ylim()[1]) if task == "snake" else 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle("Locked selections; 100 fresh episodes per trained seed; no confirmatory testing")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"fresh_validation.{suffix}",
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, task in zip(axes, summary["tasks"], strict=True):
        for arch, color in zip(
            ("linear", "mlp", "gru"), ("#2878b5", "#e58b28", "#399473"), strict=True
        ):
            runs = [
                r
                for r in results.values()
                if r["config"]["task"] == task
                and r["phase"] == "primary"
                and r["architecture"] == arch
            ]
            if len(runs) != 3:
                continue
            x = [row["transitions"] for row in runs[0]["training_curve"]]
            curves = np.asarray([[row["score"] for row in r["training_curve"]] for r in runs])
            for curve in curves:
                ax.plot(x, curve, color=color, alpha=0.2, lw=0.6)
            ax.plot(x, curves.mean(axis=0), color=color, label=arch)
        ax.set(
            title=task,
            xlabel="Training transitions",
            ylabel="Food" if task == "snake" else "Success",
        )
        ax.set_ylim(bottom=0, top=max(4.0, ax.get_ylim()[1]) if task == "snake" else 1.02)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=8)
    fig.suptitle("Primary learning efficiency: fixed development curve seeds; n = 3 trained seeds")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            directory / f"primary_learning_curves.{suffix}",
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
