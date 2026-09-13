"""Phase 1 Flappy representation diagnostic (development seeds only).

For each (encoding, ticks) the same matched development states are replayed through
fresh frozen MaleCNS columns. Descending-neuron rates are decoded, held out by
episode, into the control-relevant state variables and the oracle action.
Continuous targets: linear ridge and tanh-MLP R2 plus held-out Pearson r.
Oracle action: class counts, majority baseline, accuracy and balanced accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flyarcade_v13.representations import (
    balanced_accuracy,
    collect_states,
    mlp_probe,
    ridge_classifier,
    split,
    zscore,
)
from flyarcade_v13.study import load_topology
from flyarcade_v13_flappy import TransformedFlyFeatures
from flyarcade_v13_flappy.derived import control_quantities, encoding

TARGET_NAMES = (
    "bird_y",
    "vertical_velocity",
    "time_to_pipe",
    "gap_center",
    "signed_gap_error",
    "abs_gap_error",
    "projected_no_flap_crossing_error",
    "velocity_error",
)


def features(graph, states, *, ticks, encoding_name, batch=16):
    transform, width = encoding(encoding_name, 6)
    if transform is None:
        from flyarcade_v13.controller import FlyFeatures

        source = FlyFeatures(graph, width, batch, ticks=ticks)
    else:
        source = TransformedFlyFeatures(graph, width, batch, ticks=ticks, transform=transform)
    obs, episodes = states["observations"], states["episodes"]
    raw = np.zeros((len(obs), source.width))
    ids = np.unique(episodes)
    for start in range(0, len(ids), batch):
        block = ids[start : start + batch]
        index = {e: np.flatnonzero(episodes == e) for e in block}
        for c, e in enumerate(block):
            source.reset(c, states["seeds"][e])
        for t in range(max(len(v) for v in index.values())):
            live = [c for c, e in enumerate(block) if t < len(index[e])]
            rows = [index[block[c]][t] for c in live]
            raw[rows] = source.raw_rates(obs[rows], live)
    return raw


def pearson(truth, pred):
    t, p = truth - truth.mean(0), pred - pred.mean(0)
    denom = np.sqrt((t**2).sum(0) * (p**2).sum(0))
    return np.where(denom > 1e-12, (t * p).sum(0) / np.maximum(denom, 1e-12), 0.0)


def r2(truth, pred):
    total = ((truth - truth.mean(0)) ** 2).sum(0)
    return 1 - ((truth - pred) ** 2).sum(0) / np.maximum(total, 1e-12)


def ridge_predict(xtr, ytr, xte, penalty=10.0):
    x = np.hstack([xtr, np.ones((len(xtr), 1))])
    beta = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ ytr)
    return np.hstack([xte, np.ones((len(xte), 1))]) @ beta


def analyse(raw, states, targets):
    train, test = split(states["episodes"])
    xtr, xte = zscore(raw[train], raw[test])
    ytr, yte = targets[train], targets[test]
    mean, sd = ytr.mean(0), ytr.std(0)
    ytr_n, yte_n = (ytr - mean) / sd, (yte - mean) / sd
    lin = ridge_predict(xtr, ytr_n, xte)
    mlp = mlp_probe(xtr, ytr_n, xte, kind="regress")
    labels = states["labels"]
    ltr, lte = labels[train], labels[test]
    counts = np.bincount(lte, minlength=2)
    majority = int(np.bincount(ltr, minlength=2).argmax())
    action = {}
    for name, pred in (
        ("linear", ridge_classifier(xtr, ltr, xte, 2)),
        ("mlp", mlp_probe(xtr, ltr, xte, kind="classify", classes=2)),
    ):
        action[name] = {
            "accuracy": float(np.mean(pred == lte)),
            "balanced_accuracy": balanced_accuracy(lte, pred, 2),
            "flap_recall": float(np.mean(pred[lte == 1] == 1)) if (lte == 1).any() else None,
        }
    return {
        "continuous": {
            name: {
                "linear_r2": float(r2(yte_n[:, i], lin[:, i])),
                "linear_pearson": float(pearson(yte_n[:, i], lin[:, i])),
                "mlp_r2": float(r2(yte_n[:, i], mlp[:, i])),
                "mlp_pearson": float(pearson(yte_n[:, i], mlp[:, i])),
            }
            for i, name in enumerate(TARGET_NAMES)
        },
        "oracle_action": {
            "class_counts_test": counts.tolist(),
            "majority_accuracy": float(np.mean(lte == majority)),
            "majority_balanced_accuracy": 0.5,
            **action,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--encodings", nargs="+", default=["v12", "flappy_augmented"])
    parser.add_argument("--out", default="artifacts/v13/flappy-rescue/representation.json")
    args = parser.parse_args()
    started = time.monotonic()
    states = collect_states("flappy", min_states=6000)
    targets = np.array(
        [[control_quantities(o)[k] for k in TARGET_NAMES] for o in states["observations"]]
    )
    graph = load_topology("biological")
    report = {
        "purpose": "development-only representation diagnostic (representation seed purpose)",
        "states": int(len(targets)),
        "state_sha256": states["state_sha256"],
        "behaviour": "alternating oracle / uniform-random actions (collect_states)",
        "readout": "descending",
        "rows": {},
    }
    sensory = analyse(
        np.stack([encoding("flappy_augmented", 6)[0](o) for o in states["observations"]]),
        states,
        targets,
    )
    report["rows"]["sensory-augmented-observation"] = sensory
    for name in args.encodings:
        for ticks in args.ticks:
            t0 = time.monotonic()
            raw = features(graph, states, ticks=ticks, encoding_name=name)
            row = analyse(raw, states, targets)
            row["feature_seconds"] = time.monotonic() - t0
            key = f"biological-{name}-t{ticks}"
            report["rows"][key] = row
            c = row["continuous"]
            print(
                key,
                "linR2",
                {k[:14]: round(v["linear_r2"], 2) for k, v in c.items()},
                "action bal lin/mlp",
                round(row["oracle_action"]["linear"]["balanced_accuracy"], 3),
                round(row["oracle_action"]["mlp"]["balanced_accuracy"], 3),
                flush=True,
            )
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    report["wall_seconds"] = time.monotonic() - started
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
