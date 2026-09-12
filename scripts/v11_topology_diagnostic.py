"""Why does the degree-rewired graph learn at least as well as the biological one?

Measures properties of the descending representation that any linear readout must
work with, on both graphs, with the recurrent core frozen and under an identical
uniform-random behaviour policy. This is post-hoc analysis: it trains nothing and
feeds nothing back into the model or the frozen plan.
"""

import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph
from flyarcade.games import LaneGame
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.features import fit_for_graph

SEEDS = tuple(4_300_000 + i for i in range(30))


def probe(features, labels, classes, penalty=1.0):
    """Shared-covariance linear discriminant, the same instrument v1 settled on."""
    half = len(features) // 2
    train_x, train_y = features[:half], labels[:half]
    test_x, test_y = features[half:], labels[half:]
    centre = train_x.mean(axis=0)
    centred = train_x - centre
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    covariance += penalty * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
    scores = []
    for label in classes:
        members = centred[train_y == label]
        if not len(members):
            scores.append(np.full(len(test_x), -np.inf))
            continue
        mean = members.mean(axis=0)
        weights = np.linalg.solve(covariance, mean)
        scores.append(
            (test_x - centre) @ weights - 0.5 * mean @ weights + np.log(len(members) / len(centred))
        )
    predicted = classes[np.column_stack(scores).argmax(axis=1)]
    return float((predicted == test_y).mean())


def collect(graph, seed, guard):
    controller = V11Controller(graph, seed, stage="readout")
    controller.standardizer = fit_for_graph(
        graph, lambda g: V11Controller(g, seed, stage="readout"), SEEDS[:10], guard=guard
    )
    behaviour = np.random.default_rng(4_300_000)
    rows, labels = [], []
    for task in ("catch", "dodge"):
        for env_seed in SEEDS:
            guard.check()
            game = LaneGame(task, env_seed)
            controller.reset()
            controller.rng = np.random.default_rng(env_seed + 11)
            while not game.done:
                observation = game.observe()
                rows.append(controller.rates(observation))
                labels.append(int(np.sign(round((observation[1] - observation[0]) * 4))))
                game.step(int(behaviour.integers(0, 3)))
    return np.asarray(rows), np.asarray(labels)


def _mean_abs_correlation(centred):
    """Silent neurons have no correlation to report; exclude them rather than divide by zero."""
    active = centred[:, centred.std(axis=0) > 1e-9]
    correlation = np.corrcoef(active.T)
    return float(np.abs(correlation[np.triu_indices(active.shape[1], 1)]).mean())


def describe(features, labels):
    centred = features - features.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    power = singular**2
    return {
        "probe_accuracy": probe(features, labels, np.array([-1, 0, 1])),
        "majority_class_rate": float(np.bincount(labels + 1).max() / len(labels)),
        # Participation ratio: how many descending directions actually carry variance.
        "participation_ratio": float(power.sum() ** 2 / (power**2).sum()),
        "mean_pairwise_correlation": float(
            np.nanmean(np.abs(np.corrcoef(centred.T)[np.triu_indices(features.shape[1], 1)]))
        ),
        "mean_rate_sd": float(features.std(axis=0).mean()),
    }


def main():
    guard = ResourceGuard()
    biological = load_graph("data/malecns-v1.0/graph.npz")
    report = {
        "scope": (
            "Post-hoc comparison of the descending representation on the biological "
            "and degree-preserving rewired graphs, frozen cores, matched uniform-random "
            "behaviour, seeds in the 4,300,000 development block."
        ),
        "graphs": {},
    }
    for name, graph in (
        ("biological", biological),
        ("rewired", degree_preserving_control(biological, seed=700, swaps_per_edge=10)),
    ):
        features, labels = collect(graph, 0, guard)
        report["graphs"][name] = {**describe(features, labels), "samples": int(len(features))}
        print(name, json.dumps(report["graphs"][name], indent=2), flush=True)
    report["resources"] = guard.check()
    Path("artifacts/v11_topology_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
