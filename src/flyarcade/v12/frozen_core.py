"""Stage A sparse-matrix execution of the unchanged LIF equations.

Legacy v1/v1.1 code is untouched. Tests require bit-identical spikes/voltage on
sorted-edge graphs, including the authentic MaleCNS graph and perturbations.
"""

import numpy as np
from scipy.sparse import csr_matrix


class FrozenCore:
    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        self.plastic_count = 0
        self._mask = None
        self._masked = None
        g = self.graph
        # Stable presynaptic summation order matches the historical edge-list loop.
        if np.any(g.pre[1:] < g.pre[:-1]):
            raise ValueError("frozen CSR execution requires canonical presynaptic edge order")
        self._weights = self.magnitude * self.signs[g.pre]
        self._matrix = csr_matrix((self._weights, (g.post, g.pre)), shape=(g.n, g.n))
        self.reset()

    def reset(self):
        self.voltage = np.zeros(self.graph.n)
        self.spikes = np.zeros(self.graph.n)
        self.refractory = np.zeros(self.graph.n, dtype=int)
        self.pre_trace = np.zeros(self.graph.n)
        self.eligibility = np.zeros(0)

    def tick(self, drive, *, plastic=False, edge_mask=None, neuron_mask=None):
        if plastic:
            raise ValueError("v1.2 Stage A core cannot learn")
        drive = np.asarray(drive)
        if drive.shape != (self.graph.n,) or not np.isfinite(drive).all():
            raise ValueError("finite input per neuron required")
        matrix = self._matrix
        if edge_mask is not None:
            if self._mask is not edge_mask:
                g = self.graph
                self._masked = csr_matrix(
                    (self._weights * edge_mask, (g.post, g.pre)), shape=(g.n, g.n)
                )
                self._mask = edge_mask
            matrix = self._masked
        current = matrix @ self.spikes
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
        self.pre_trace = 0.8 * self.pre_trace + spikes
        self.spikes = spikes.astype(float)
        return self.spikes.copy()
