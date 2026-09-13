"""v1.5 reports: runs.csv, development/validation summaries, markdown tables and figures."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/v15")
PLAN = json.loads(Path("experiments/v15/development_plan.json").read_text())
TASKS = PLAN["tasks"]
V14 = PLAN["v14_validation_reference"]


def load_results():
    out = {}
    for path in sorted(Path("runs/v15").glob("*/result.json")):
        if path.parent.name.startswith(("diag-", "smoke-")):
            continue
        out[path.parent.name] = json.loads(path.read_text())
    return out


def quarter(history, key, last=True):
    q = max(1, len(history) // 4)
    rows = history[-q:] if last else history[:q]
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def run_row(name, r, spec):
    c, h = r["config"], r["training_history"]
    return {
        "run": name,
        "task": c["task"],
        "stage": r["phase"],
        "role": spec.get("role"),
        "seed": c["seed"],
        "source": c["source"],
        "architecture": c["arch"],
        "hidden_size": c["hidden"] if c["arch"] != "linear" else 0,
        "recurrent_size": c["core"] if c["arch"] == "gru" else 0,
        "second_layer": c["core"] if c["arch"] == "mlp" else 0,
        "bptt_length": c["chunk"] if c["arch"] == "gru" else 0,
        "optimizer": r["optimizer"],
        "learning_rate": c["lr"],
        "lr_schedule": f"{c['lr_schedule']} (floor {c['lr_floor']})",
        "entropy_coefficient": c["ent"],
        "entropy_schedule": "constant" if c["ent_final"] is None else f"linear to {c['ent_final']}",
        "gamma": c["gamma"],
        "gae_lambda": c["lam"],
        "ppo_clip": c["clip"],
        "value_loss_coefficient": c["vf"],
        "max_grad_norm": c["max_grad_norm"],
        "epochs": c["epochs"],
        "minibatches": c["minibatches"],
        "parallel_envs": c["envs"],
        "rollout_steps": c["steps"],
        "transitions": r["training_transitions"],
        "ticks_per_action": c["ticks"] if c["source"] == "fly" else None,
        "readout": c["readout"] if c["source"] == "fly" else "sensory encoding (no SNN)",
        "normalization": f"standardize, floor {c['std_floor']}, clip 4"
        if c["source"] == "fly"
        else "encoding mapped to [-1,1]",
        "curriculum": "none"
        if c["curriculum"] == [["target", 1.0]]
        else json.dumps(c["curriculum"]),
        "shaping": "none",
        "imitation_transitions": (r.get("imitation") or {}).get("transitions", 0),
        "score": r["score"],
        "last_checkpoint_score": r["last_checkpoint_score"],
        "chosen_checkpoint_transitions": r["chosen_checkpoint_transitions"],
        "state_dependence": r["state_probe"]["score"],
        "action_distribution": json.dumps([round(x, 4) for x in r["action_distribution"]]),
        "dominant_action_fraction": r["dominant_action_fraction"],
        "policy_entropy": r["policy_entropy"],
        "train_entropy_last_quarter": quarter(h, "entropy"),
        "value_loss_last_quarter": quarter(h, "value_loss"),
        "explained_variance_last_quarter": quarter(h, "explained_variance"),
        "approx_kl_last_quarter": quarter(h, "approx_kl"),
        "clip_fraction_last_quarter": quarter(h, "clip_fraction"),
        "grad_norm_first_quarter": quarter(h, "grad_norm", last=False),
        "grad_norm_last_quarter": quarter(h, "grad_norm"),
        "random_score": r["random_score"],
        "reference_score": r["reference_score"],
        "training_seconds": r["wall_clock_training_seconds"],
        "transitions_per_second": r["training_transitions"] / r["wall_clock_training_seconds"],
        "peak_memory_mb": r["peak_memory_mb"],
        "checkpoint_sha256": r["checkpoint_sha256"],
        "policy_sha256": r["policy_sha256"],
        "failure_notes": r["failure_reason"] or "",
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("status\nno rows\n")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, task):
    if x is None:
        return "—"
    return f"{x:.2f}" if task == "snake" else f"{x:.3f}"


def main():
    results = load_results()
    specs = {
        p.stem: json.loads(p.read_text())
        for p in Path("experiments/v15/specs/development").glob("*.json")
    }
    rows = [run_row(n, r, specs.get(n, {})) for n, r in results.items()]
    write_csv(ROOT / "runs.csv", rows)
    failures = [
        json.loads(p.read_text())
        for p in sorted(Path("runs/v15").glob("*/failure.json"))
        if not p.parent.name.startswith(("diag-", "smoke-"))
    ]
    write_csv(ROOT / "failures.csv", failures)
    configs = [json.loads(p.read_text()) for p in sorted((ROOT / "configurations").glob("*.json"))]
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
    by_task = {t: [c for c in configs if c["config"]["task"] == t] for t in TASKS}
    config_rows = [
        {
            "task": c["config"]["task"],
            "configuration": c["name"],
            "stage": c["stage"],
            "role": c["role"],
            "arch": c["config"]["arch"],
            "hidden": c["config"]["hidden"],
            "ticks": c["config"]["ticks"],
            "readout": c["config"]["readout"] if c["config"]["source"] == "fly" else "sensory",
            "transitions": c["config"]["transitions"],
            "mean": c["mean"],
            "sd": c["sd"],
            "seed_scores": json.dumps(c["scores"]),
            "state_probes": json.dumps(c["state_probes"]),
            "gates_passed": c["gates_passed"],
            "eligible": c["eligible"],
            "utility": c["utility"],
        }
        for c in configs
    ]
    write_csv(ROOT / "development_configurations.csv", config_rows)
    write_csv(ROOT / "rescue_results.csv", [r for r in config_rows if r["role"] == "rescue"])
    write_csv(ROOT / "leakage_controls.csv", [r for r in config_rows if r["stage"] == "B"])

    lines = ["# v1.5 development progress\n"]
    for t in TASKS:
        lines.append(f"\n## {t}\n")
        lines.append("| Configuration | Stage | Role | Mean | Seeds | Probes | Eligible |")
        lines.append("|---|---|---|---:|---|---|---|")
        for c in by_task[t]:
            lines.append(
                f"| {c['name']} | {c['stage']} | {c['role']} | {fmt(c['mean'], t)} | "
                f"{[round(s, 3) for s in c['scores']]} | "
                f"{[round(p, 2) for p in c['state_probes'] if p is not None]} | {c['eligible']} |"
            )
    (ROOT / "progress.md").write_text("\n".join(lines) + "\n")

    summary = {
        "plan_sha256": selected["plan_sha256"] if selected else None,
        "status": "VALIDATED" if validation else "LOCKED" if selected else "DEVELOPMENT",
        "tasks": {},
        "confirmatory_testing": False,
    }
    for t in TASKS:
        entry = {"configurations": by_task[t], "v14_validation": V14[t]}
        if selected:
            entry["selection"] = selected["tasks"][t]
        if validation:
            entry["validation"] = validation["tasks"][t]
        summary["tasks"][t] = entry
    (ROOT / "development_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if selected:
        final_table(selected, validation, by_task)
    if validation:
        figures(selected, validation, by_task, results)
    print((ROOT / "progress.md").read_text()[:4000])


def final_table(selected, validation, by_task):
    out = [
        "# v1.5 development and fresh validation\n",
        "| Task | v1.4 validation | v1.5 validation | target | absolute gain | relative gain | "
        "selected architecture | ticks | readout | transitions | status |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for t in TASKS:
        rec = selected["tasks"][t]["headline_selected"]
        val = (validation or {}).get("tasks", {}).get(t, {}).get("headline_selected")
        if rec is None:
            out.append(
                f"| {t} | {fmt(V14[t], t)} | — | {PLAN['targets'][t]} | — | — | none "
                "| | | | FAILED |"
            )
            continue
        c = rec["config"]
        arch = c["arch"] if c["arch"] == "linear" else f"{c['arch']} {c['hidden']}/{c['core']}"
        if val:
            gain = val["mean"] - V14[t]
            status = "TARGET MET" if val["mean"] >= PLAN["targets"][t] else "BELOW TARGET"
            if not all(p > 0.05 for p in val["state_probes"]) or not all(
                d < 0.98 for d in val["dominant_action_fractions"]
            ):
                status += " (gate failed)"
            cells = [fmt(val["mean"], t), f"{gain:+.3f}", f"{gain / V14[t] * 100:+.1f}%"]
        else:
            status, cells = "LOCKED (not yet validated)", ["pending", "—", "—"]
        out.append(
            f"| {t} | {fmt(V14[t], t)} | {cells[0]} | {PLAN['targets'][t]} | {cells[1]} | "
            f"{cells[2]} | {arch} | {c['ticks']} | {c['readout']} | {c['transitions']} | {status} |"
        )
    out.append("\nv1.4 validation used a different seed block; differences are descriptive.")
    if validation:
        out.append("\n## Per-seed fresh validation (100 episodes per trained seed)\n")
        out.append(
            "| Task | Role | Configuration | Seeds | Mean | SD | Probes | "
            "Dominant action | Entropy |"
        )
        out.append("|---|---|---|---|---:|---:|---|---|---|")
        for t in TASKS:
            for role, v in validation["tasks"][t].items():
                out.append(
                    f"| {t} | {role} | {v['configuration']} | "
                    f"{[round(x, 3) for x in v['seed_values']]} | {fmt(v['mean'], t)} | "
                    f"{v['sd']:.3f} | {[round(x, 2) for x in v['state_probes']]} | "
                    f"{[round(x, 2) for x in v['dominant_action_fractions']]} | "
                    f"{[round(x, 2) for x in v['policy_entropies']]} |"
                )
    (ROOT / "final_table.md").write_text("\n".join(out) + "\n")


def figures(selected, validation, by_task, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "flyarcade-v15", "savefig.dpi": 160}
    )
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    ink, muted = "#222222", "#777777"
    colors = {"linear": "#4C78A8", "mlp": "#F58518", "gru": "#54A24B", "control": "#B0B0B0"}

    def save(fig, stem):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(
                fig_dir / f"{stem}.{suffix}", metadata={"Date": None} if suffix == "svg" else None
            )
        plt.close(fig)

    # Stage A: learner x integration time
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    for ax, t in zip(axes, TASKS, strict=True):
        recs = [c for c in by_task[t] if c["stage"] == "A"]
        for i, c in enumerate(recs):
            ax.bar(i, c["mean"], color=colors[c["config"]["arch"]], width=0.7)
            ax.scatter([i] * len(c["scores"]), c["scores"], color=ink, s=10, zorder=3)
        ax.set_xticks(range(len(recs)))
        ax.set_xticklabels(
            [f"{c['config']['arch']}\n{c['config']['ticks']}t" for c in recs], fontsize=7
        )
        ax.set_title(t)
        ax.set_ylabel("food/episode" if t == "snake" else "success")
    fig.suptitle(
        "Stage A: learner × integration time, descending readout "
        "(bars = mean of 3 seeds, dots = seeds)"
    )
    save(fig, "stage_A_learner_ticks")

    # Stage B: readout and leakage controls
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    for ax, t in zip(axes, TASKS, strict=True):
        winner = selected["tasks"][t]["stage_A_architecture_winner"]
        recs = [winner] + [c for c in by_task[t] if c["stage"] == "B"]
        labels = []
        for i, c in enumerate(recs):
            control = c["role"] == "control"
            ax.bar(
                i,
                c["mean"],
                color=colors["control"] if control else colors[c["config"]["arch"]],
                hatch="//" if control else None,
                edgecolor="white",
                width=0.7,
            )
            ax.scatter([i] * len(c["scores"]), c["scores"], color=ink, s=10, zorder=3)
            labels.append(
                "sensory" if c["config"]["source"] == "sensory" else c["config"]["readout"]
            )
        ax.set_xticks(range(len(recs)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(t)
    fig.suptitle("Stage B: readout scope; hatched grey = input-leakage controls (not selectable)")
    save(fig, "stage_B_readout_leakage")

    # Validation vs v1.4
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
    for ax, t in zip(axes, TASKS, strict=True):
        v = validation["tasks"][t].get("headline_selected")
        ax.bar(0, V14[t], color=muted, width=0.6)
        if v:
            ax.bar(1, v["mean"], color=colors["gru"], width=0.6)
            ax.scatter([1] * 3, v["seed_values"], color=ink, s=14, zorder=3)
        ax.axhline(PLAN["targets"][t], ls="--", color="#B04A48", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v1.4", "v1.5"])
        ax.set_title(t)
        ax.set_ylabel("food/episode" if t == "snake" else "success")
    fig.suptitle(
        "Fresh validation of locked selections (dashed = v1.5 target; dots = trained seeds)"
    )
    save(fig, "validation_v14_vs_v15")

    # Learning curves of the headline selection
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, t in zip(axes, TASKS, strict=True):
        rec = selected["tasks"][t]["headline_selected"]
        if not rec:
            continue
        for s in (0, 1, 2):
            r = results.get(f"{rec['name']}-s{s}")
            if r:
                ax.plot(
                    [p["transitions"] for p in r["training_curve"]],
                    [p["score"] for p in r["training_curve"]],
                    color=colors[rec["config"]["arch"]],
                    lw=1.5,
                )
        ax.set_title(t)
        ax.set_xlabel("transitions")
    fig.suptitle(
        "Checkpoint-selection curves of the selected configuration (checkpoint_select seeds)"
    )
    save(fig, "selected_learning_curves")


if __name__ == "__main__":
    main()
