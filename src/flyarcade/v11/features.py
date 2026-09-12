"""Frozen per-neuron standardisation of descending population rates.

v1 fed raw centred rates straight into the readout. Because every descending
neuron shares a large common-mode component, the state-independent direction
dominated the policy gradient and the readout saturated on one action. Removing
that common mode per neuron is the minimal fix, and it is a fixed preprocessing
step: statistics are estimated once on development rollouts of the frozen core and
never re-estimated from confirmatory data.
"""

import json
from pathlib import Path

import numpy as np

FIT_TASKS = ("catch", "dodge")


class Standardizer:
    def __init__(self, mean, scale, *, clip=4.0, normalise=True):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        if self.mean.shape != self.scale.shape or self.mean.ndim != 1:
            raise ValueError("mean and scale must be one-dimensional and equal length")
        if not np.isfinite(self.mean).all() or not (self.scale > 0).all():
            raise ValueError("finite mean and strictly positive scale required")
        self.clip = float(clip)
        # Divide by sqrt(width) so ||phi|| is order 1 regardless of population size.
        # Without this, a 1,293-neuron readout makes every learning rate about a
        # thousand times too large and the critic saturates inside one episode.
        self.normalise = bool(normalise)
        self.gain = 1.0 / np.sqrt(len(self.mean)) if self.normalise else 1.0

    @property
    def width(self):
        return int(len(self.mean))

    def __call__(self, rates):
        rates = np.asarray(rates, dtype=float)
        if rates.shape != self.mean.shape:
            raise ValueError("rate vector must match the fitted width")
        standardised = np.clip((rates - self.mean) / self.scale, -self.clip, self.clip)
        return standardised * self.gain

    @classmethod
    def fit(cls, samples, *, floor=0.02):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or len(samples) < 2:
            raise ValueError("at least two sampled rate vectors required")
        # A floor keeps silent or saturated neurons from being amplified into noise.
        return cls(samples.mean(axis=0), np.maximum(samples.std(axis=0), floor))

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "clip": self.clip,
                    "normalise": self.normalise,
                }
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["mean"],
            payload["scale"],
            clip=payload["clip"],
            normalise=payload.get("normalise", True),
        )


def fit_for_graph(graph, controller_factory, seeds, *, behaviour_seed=4_200_000, guard=None):
    """Fit the frozen standardisation procedure against one specific graph.

    The rewired topology control must be standardised by the same procedure applied
    to its own rates, not by statistics borrowed from the biological graph, or the
    topology comparison would hand the biological arm a calibrated readout and the
    control an uncalibrated one.
    """
    from flyarcade.games import LaneGame

    controller = controller_factory(graph)
    behaviour = np.random.default_rng(behaviour_seed)
    samples = []
    for task in FIT_TASKS:
        for seed in seeds:
            if guard is not None:
                guard.check()
            game = LaneGame(task, seed)
            controller.reset()
            controller.rng = np.random.default_rng(seed + 11)
            while not game.done:
                samples.append(controller.rates(game.observe()))
                game.step(int(behaviour.integers(0, 3)))
    return Standardizer.fit(np.asarray(samples))
