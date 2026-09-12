"""Connectome LIF core with engineered visual injection and descending readout."""

import numpy as np

from flyarcade.motor import Motor
from flyarcade.neural.lif import LIF
from flyarcade.sensory import features


class Controller:
    def __init__(self, graph, seed=0, *, recurrent_learning=True):
        self.graph = graph
        self.rng = np.random.default_rng(seed)
        neurons = {n["bodyId"]: n for n in graph.provenance.get("neurons", [])}
        # A topology control keeps biological metadata via its parent record.
        if not neurons:
            neurons = {n["bodyId"]: n for n in graph.provenance["parent"]["neurons"]}
        ordered = [neurons[int(i)] for i in graph.neuron_ids]
        self.inputs = np.array(
            [i for i, n in enumerate(ordered) if n.get("superclass") == "visual_projection"],
            dtype=int,
        )
        self.outputs = np.array(
            [i for i, n in enumerate(ordered) if n.get("superclass") == "descending_neuron"],
            dtype=int,
        )
        if not len(self.inputs) or not len(self.outputs):
            raise ValueError("visual inputs and descending outputs required")
        signs = [
            1
            if n.get("consensusNt") == "acetylcholine"
            else -1
            if n.get("consensusNt") == "gaba"
            else 0
            for n in ordered
        ]
        self.core = LIF(graph, signs)
        self.recurrent_learning = recurrent_learning
        # Fixed electrode assignment by sorted body ID, not a biological mapping.
        self.input_channels = np.arange(len(self.inputs)) % 8
        self.motor = Motor(len(self.outputs), self.rng)
        self.baseline = 0.0
        self.reset()

    def reset(self):
        self.core.reset()
        self.motor.reset()
        self.spike_count = 0
        self.tick_count = 0

    def act(self, observation, *, training=False, noise=0, edge_mask=None, neuron_mask=None):
        obs = np.asarray(observation)
        if noise:
            obs = np.clip(obs + self.rng.normal(0, noise, 3), 0, 1)
        f = features(obs)
        counts = np.zeros(len(self.outputs))
        for _ in range(4):
            drive = np.full(self.graph.n, 0.18)
            drive[self.inputs] += 1.5 * (self.rng.random(len(self.inputs)) < f[self.input_channels])
            spikes = self.core.tick(
                drive,
                plastic=training and self.recurrent_learning,
                edge_mask=edge_mask,
                neuron_mask=neuron_mask,
            )
            counts += spikes[self.outputs]
            self.spike_count += int(spikes.sum())
            self.tick_count += 1
        # Keep single-neuron differences; centering removes common-mode activity.
        return self.motor.choose(counts / 4 - 0.25, self.rng, training)

    def reward(self, value):
        advantage = float(np.clip(value - self.baseline, -2, 2))
        self.baseline = 0.98 * self.baseline + 0.02 * value
        self.motor.reward(advantage)
        if self.recurrent_learning:
            self.core.reward(advantage)
