"""Fit the frozen Snake feature standardisation on Snake development seeds only.

Identical procedure to the catch-dodge fit: frozen core, uniform-random behaviour so
the statistics do not depend on any policy that will later be trained.
"""

import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import degree_preserving_control, load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.features import Standardizer
from flyarcade.v11.snake import CHANNELS, SnakeGame, heuristic
from flyarcade.v11.snake import features as snake_features
from flyarcade.v11.snake_experiment import FEATURE_FIT_SEEDS


def snake_controller(graph, seed=0):
    return V11Controller(
        graph,
        seed,
        stage="readout",
        encoder=snake_features,
        channels=CHANNELS,
        observation_width=CHANNELS,
        actions=4,
    )


def fit_snake_standardizer(graph, seeds, *, seed=0, behaviour_seed=7_200_000, guard=None):
    controller = snake_controller(graph, seed)
    behaviour = np.random.default_rng(behaviour_seed)
    samples = []
    # A uniform-random snake dies in about eleven steps, so random rollouts alone
    # never visit the long-body states the trained policy spends most of its time
    # in. The fixed heuristic supplies that coverage. Both behaviour policies are
    # fixed and policy-independent; neither is ever trained.
    for mode in ("random", "heuristic"):
        for env_seed in seeds:
            if guard is not None:
                guard.check()
            game = SnakeGame(env_seed)
            controller.reset()
            controller.rng = np.random.default_rng(env_seed + 11)
            while not game.done:
                samples.append(controller.rates(game.observe()))
                action = int(behaviour.integers(0, 4)) if mode == "random" else heuristic(game)
                game.step(action)
    return Standardizer.fit(np.asarray(samples)), len(samples)


def main():
    guard = ResourceGuard()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    standardizer, samples = fit_snake_standardizer(graph, FEATURE_FIT_SEEDS, guard=guard)
    standardizer.save("artifacts/snake_standardizer.json")
    # The rewired topology control must be standardised by the same procedure applied
    # to its own rates. Reusing the biological statistics would give the biological
    # arm a calibrated readout and the control an uncalibrated one, which would
    # manufacture a topology difference out of a preprocessing mismatch.
    rewired = {}
    for seed in (0, 1, 2):
        control = degree_preserving_control(graph, seed=700 + seed, swaps_per_edge=10)
        fitted, count = fit_snake_standardizer(
            control, FEATURE_FIT_SEEDS, seed=seed, guard=ResourceGuard()
        )
        path = f"artifacts/snake_standardizer_rewired_{seed}.json"
        fitted.save(path)
        rewired[path] = {"samples": count, "mean_abs": float(np.abs(fitted.mean).mean())}
    report = {
        "rewired": rewired,
        "width": standardizer.width,
        "samples": samples,
        "channels": CHANNELS,
        "seeds": "7,200,000 block, frozen core, fixed random and heuristic behaviour policies",
        "mean_abs": float(np.abs(standardizer.mean).mean()),
        "scale_mean": float(standardizer.scale.mean()),
        "scale_floor_hits": int((standardizer.scale <= 0.02).sum()),
        "gain": standardizer.gain,
        "resources": guard.check(),
    }
    Path("artifacts/snake_standardizer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
