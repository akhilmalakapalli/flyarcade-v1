"""Phase-1 learner diagnostics from recorded v1.4 development histories (read-only)."""

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("artifacts/v15/diagnosis")


def summarize(r):
    h = r["training_history"]
    curve = [row["score"] for row in r["training_curve"]]
    q = max(1, len(h) // 4)

    def avg(rows, key):
        vals = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    counts = np.asarray([row["rollout_action_fractions"] for row in h], dtype=float)
    frac = counts / counts.sum(1, keepdims=True)
    return {
        "run": r["name"],
        "task": r["config"]["task"],
        "phase": r["phase"],
        "arch": r["config"]["arch"],
        "ticks": r["config"]["ticks"],
        "readout": r["config"]["readout"],
        "transitions": r["config"]["transitions"],
        "dev_score": r["score"],
        "state_probe": r["state_probe"]["score"],
        "greedy_dominant_action": r["dominant_action_fraction"],
        "entropy_first_quarter": avg(h[:q], "entropy"),
        "entropy_last_quarter": avg(h[-q:], "entropy"),
        "value_loss_first_quarter": avg(h[:q], "value_loss"),
        "value_loss_last_quarter": avg(h[-q:], "value_loss"),
        "clip_fraction_last_quarter": avg(h[-q:], "clip_fraction"),
        "approx_kl_last_quarter": avg(h[-q:], "approx_kl"),
        "grad_norm_first_quarter": avg(h[:q], "grad_norm"),
        "grad_norm_last_quarter": avg(h[-q:], "grad_norm"),
        "train_success_last_quarter": avg(h[-q:], "train_success"),
        "rollout_dominant_action_last_quarter": float(frac[-q:].max(1).mean()),
        "curve": curve,
        "curve_peak": max(curve),
        "curve_final": curve[-1],
        "late_drop": max(curve) - curve[-1],
        "still_rising": len(curve) >= 2 and curve[-1] >= max(curve[:-1]),
        "wall_seconds": r["wall_clock_training_seconds"],
        "transitions_per_second": r["training_transitions"] / r["wall_clock_training_seconds"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize(json.loads(p.read_text())) for p in sorted(Path("runs/v14").glob("*/result.json"))
    ]
    with (OUT / "v14_learner_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(
            {k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows
        )
    configs = {}
    for r in rows:
        configs.setdefault(r["run"].rsplit("-s", 1)[0], []).append(r)
    summary = {}
    for name, rs in configs.items():
        keys = [
            k for k, v in rs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        summary[name] = {
            k: float(np.mean([r[k] for r in rs if r[k] is not None]))
            if any(r[k] is not None for r in rs)
            else None
            for k in keys
        }
        summary[name]["curves"] = [r["curve"] for r in rs]
        summary[name]["still_rising_seeds"] = sum(r["still_rising"] for r in rs)
    (OUT / "v14_learner_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    keys = ("dev_score", "entropy_last_quarter", "value_loss_last_quarter")
    keys += ("clip_fraction_last_quarter", "approx_kl_last_quarter", "grad_norm_last_quarter")
    keys += ("rollout_dominant_action_last_quarter", "late_drop")
    for name, s in summary.items():
        print(f"{name:28s}", " ".join(f"{k.split('_')[0]}={s[k]:.4f}" for k in keys))


if __name__ == "__main__":
    main()
