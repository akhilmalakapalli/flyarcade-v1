"""Sparse discrete LIF model; time is in simulation ticks, not calibrated biology."""

import numpy as np


class LIF:
    def __init__(self, graph, signs, *, learning_rate=0.0002):
        self.graph = graph
        self.signs = np.asarray(signs, dtype=float)
        if self.signs.shape != (graph.n,) or not np.isin(self.signs, [-1, 0, 1]).all():
            raise ValueError("one -1/0/+1 sign per neuron required")
        self.learning_rate = learning_rate
        counts = graph.counts.astype(float)
        incoming = np.bincount(graph.post, weights=counts, minlength=graph.n)
        self.base = 2.5 * counts / np.maximum(incoming[graph.post], 1)
        self.magnitude = self.base.copy()
        self.reset()

    def reset(self):
        self.voltage = np.zeros(self.graph.n)
        self.spikes = np.zeros(self.graph.n)
        self.refractory = np.zeros(self.graph.n, dtype=int)
        self.trace = np.zeros(self.graph.n)
        self.eligibility = np.zeros(self.graph.e)

    def tick(self, drive, *, plastic=True, edge_mask=None, neuron_mask=None):
        drive = np.asarray(drive)
        if drive.shape != (self.graph.n,) or not np.isfinite(drive).all():
            raise ValueError("finite input per neuron required")
        g = self.graph
        weights = self.magnitude * self.signs[g.pre]
        if edge_mask is not None:
            weights = weights * edge_mask
        current = np.bincount(g.post, weights=weights * self.spikes[g.pre], minlength=g.n)
        ready = self.refractory == 0
        self.refractory = np.maximum(self.refractory - 1, 0)
        self.voltage = np.where(ready, 0.85 * self.voltage + drive + current, 0)
        self.voltage = np.clip(self.voltage, -3, 4)
        spikes = (self.voltage >= 1) & ready
        if neuron_mask is not None:
            spikes &= neuron_mask
            self.voltage[~neuron_mask] = 0
        self.voltage[spikes] = 0
        self.refractory[spikes] = 1
        if plastic:
            # Old traces implement strict causal/anti-causal pairing, excluding
            # synchronous pairs. Eligibility is bounded and decays between ticks.
            pair = self.trace[g.pre] * spikes[g.post] - self.trace[g.post] * spikes[g.pre]
            self.eligibility = np.clip(0.95 * self.eligibility + pair, -5, 5)
            self.trace = 0.8 * self.trace + spikes
        self.spikes = spikes.astype(float)
        return self.spikes.copy()

    def reward(self, advantage):
        if not np.isfinite(advantage):
            raise ValueError("finite reward required")
        self.magnitude = np.clip(
            self.magnitude + self.learning_rate * np.clip(advantage, -2, 2) * self.eligibility,
            0.25 * self.base,
            2 * self.base,
        )
