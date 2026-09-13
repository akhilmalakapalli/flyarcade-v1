"""Post-hoc decoding under matched exogenous behavior, held out by episode."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v12.environments import TASKS
from flyarcade.v12.study import array_hash, build, episode, graph_hash, seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    guard = ResourceGuard()
    output = Path(f"artifacts/v12/representation-{args.task}.json")
    if output.exists():
        print(output, "preserved")
        return
    rows = []
    matched = None
    for seed in range(3):
        for condition in ("biological", "rewired"):
            topology = f"rewired-{seed}" if condition == "rewired" else "biological"
            graph = load_graph(
                f"data/v12/{topology}.npz"
                if condition == "rewired"
                else "data/malecns-v1.0/graph.npz"
            )
            controller = build(graph, args.task, seed)
            stdpath = Path(f"artifacts/v12/{args.task}-{topology}-standardizer.json")
            standardizer = Standardizer.load(stdpath)
            samples, labels, groups, observations = [], [], [], []
            seeds = [seed_for(args.task, "representation", index=i) for i in range(20)]
            for group, env_seed in enumerate(seeds):
                _, activity, obs, y = episode(
                    controller, args.task, env_seed, collect=True, guard=guard
                )
                take = np.linspace(0, len(activity) - 1, min(24, len(activity)), dtype=int)
                samples.extend(activity[take])
                labels.extend(y[take])
                observations.extend(obs[take])
                groups.extend([group] * len(take))
            x, y, groups = np.asarray(samples), np.asarray(labels), np.asarray(groups)
            state_hash = array_hash(np.asarray(observations))
            signature = (state_hash, array_hash(y), tuple(groups))
            if matched is None:
                matched = signature
            assert signature == matched, "unmatched behavior invalidates comparison"
            centered = x - x.mean(axis=0)
            gram = centered @ centered.T / max(len(x) - 1, 1)
            eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0)
            ratio = float(eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-15))
            active = np.flatnonzero(x.std(axis=0) > 1e-9)
            subset = active[np.linspace(0, len(active) - 1, min(128, len(active)), dtype=int)]
            corr = np.corrcoef(x[:, subset].T)
            mean_corr = float(corr[np.triu_indices(len(subset), 1)].mean())
            features = np.asarray([standardizer(row) for row in x])
            features = np.column_stack([features, np.ones(len(features))])
            train, test = groups < 10, groups >= 10
            a, b = features[train], features[test]
            targets = np.eye(TASKS[args.task].actions)[y[train]]
            dual = np.linalg.solve(a @ a.T + np.eye(len(a)), targets)
            predicted = (b @ a.T @ dual).argmax(axis=1)
            majority = int(np.bincount(y[train], minlength=TASKS[args.task].actions).argmax())
            rows.append(
                {
                    "task": args.task,
                    "condition": condition,
                    "seed": seed,
                    "graph_sha256": graph_hash(graph),
                    "state_sha256": state_hash,
                    "standardizer_sha256": hashlib.sha256(stdpath.read_bytes()).hexdigest(),
                    "activity_sha256": array_hash(x),
                    "samples": len(x),
                    "training_samples": int(train.sum()),
                    "test_samples": int(test.sum()),
                    "participation_ratio": ratio,
                    "mean_pairwise_correlation": mean_corr,
                    "population_activity_variance": float(x.var(axis=0).mean()),
                    "linear_probe_accuracy": float(np.mean(predicted == y[test])),
                    "majority_accuracy": float(np.mean(y[test] == majority)),
                    "correlation_neuron_indices": subset.tolist(),
                    "episode_seeds": seeds,
                }
            )
    payload = {
        "post_hoc": True,
        "task": args.task,
        "rows": rows,
        "method": "Same alternating heuristic/random behavior and states across all graphs; "
        "up to 24 evenly spaced states per episode; first ten episodes train, "
        "last ten test; ridge 1 with intercept; heuristic-action labels; "
        "raw-rate participation ratio/variance; up to 128 active neurons correlation; "
        "probe uses frozen development standardizer; no policy tuning",
        "resources": guard.check(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        args.task, [(r["condition"], r["seed"], round(r["linear_probe_accuracy"], 3)) for r in rows]
    )


if __name__ == "__main__":
    main()
