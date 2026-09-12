"""LIF core with e-prop eligibility traces and neuron-specific learning signals.

The membrane dynamics are copied exactly from `flyarcade.neural.lif.LIF` so that a
frozen v1.1 core is bit-identical to the frozen v1 core; `tests/test_v11.py`
asserts this. Only the credit-assignment machinery is new: v1 accumulated a single
symmetric STDP pair term per synapse and multiplied every synapse by one global
scalar advantage, which is the update rule the v1 results falsified. Here each
synapse keeps an e-prop eligibility trace and is driven by a learning signal
belonging to its own postsynaptic neuron.
"""

import numpy as np


class EpropCore:
    """Sparse LIF core. Weight magnitudes stay bounded and signs never change."""

    def __init__(
        self,
        graph,
        signs,
        *,
        plastic_edges=None,
        trace_decay=0.8,
        eligibility_decay=0.9,
        pseudo_derivative_scale=0.3,
        weight_bound=2.0,
    ):
        self.graph = graph
        self.signs = np.asarray(signs, dtype=float)
        if self.signs.shape != (graph.n,) or not np.isin(self.signs, [-1, 0, 1]).all():
            raise ValueError("one -1/0/+1 sign per neuron required")
        counts = graph.counts.astype(float)
        incoming = np.bincount(graph.post, weights=counts, minlength=graph.n)
        # Identical initial magnitudes to v1: anatomical counts set relative
        # strength; they are not conductances.
        self.base = 2.5 * counts / np.maximum(incoming[graph.post], 1)
        self.magnitude = self.base.copy()
        self.trace_decay = float(trace_decay)
        self.eligibility_decay = float(eligibility_decay)
        self.pseudo_derivative_scale = float(pseudo_derivative_scale)
        self.weight_bound = float(weight_bound)
        if plastic_edges is None:
            self.plastic_edges = np.zeros(graph.e, dtype=bool)
        else:
            self.plastic_edges = np.asarray(plastic_edges, dtype=bool)
            if self.plastic_edges.shape != (graph.e,):
                raise ValueError("plastic edge mask must cover every edge")
        self.plastic_index = np.flatnonzero(self.plastic_edges)
        self.reset()

    @property
    def plastic_count(self):
        return int(len(self.plastic_index))

    def reset(self):
        self.voltage = np.zeros(self.graph.n)
        self.spikes = np.zeros(self.graph.n)
        self.refractory = np.zeros(self.graph.n, dtype=int)
        self.pre_trace = np.zeros(self.graph.n)
        self.eligibility = np.zeros(len(self.plastic_index))

    def tick(self, drive, *, plastic=False, edge_mask=None, neuron_mask=None):
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
        if plastic and len(self.plastic_index):
            # e-prop eligibility: presynaptic trace gated by the postsynaptic
            # surrogate derivative, accumulated locally per synapse. The voltage
            # read here is the pre-reset value, which is what the surrogate needs.
            psi = self.pseudo_derivative_scale * np.maximum(0.0, 1.0 - np.abs(self.voltage - 1.0))
            index = self.plastic_index
            self.eligibility = (
                self.eligibility_decay * self.eligibility
                + psi[g.post[index]] * self.pre_trace[g.pre[index]]
            )
        self.voltage[spikes] = 0
        self.refractory[spikes] = 1
        self.pre_trace = self.trace_decay * self.pre_trace + spikes
        self.spikes = spikes.astype(float)
        return self.spikes.copy()

    def learn(self, learning_signal, *, learning_rate, update_clip=0.02):
        """Apply neuron-specific learning signals through the eligibility traces.

        `learning_signal` carries one value per neuron. Each synapse is driven by
        the signal of its own postsynaptic neuron, never by a single global scalar.
        """
        if not len(self.plastic_index):
            return 0.0
        signal = np.asarray(learning_signal, dtype=float)
        if signal.shape != (self.graph.n,) or not np.isfinite(signal).all():
            raise ValueError("one finite learning signal per neuron required")
        index = self.plastic_index
        update = learning_rate * signal[self.graph.post[index]] * self.eligibility
        update = np.clip(update, -update_clip, update_clip)
        magnitude = self.magnitude.copy()
        magnitude[index] = np.clip(
            magnitude[index] + update,
            self.base[index] / self.weight_bound,
            self.base[index] * self.weight_bound,
        )
        self.magnitude = magnitude
        return float(np.abs(update).mean())
