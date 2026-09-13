"""v1.4 feature source adapted from frozen v1.3 with historical task encoders.

``FlyFeatures`` runs the unchanged Stage A MaleCNS LIF core for a batch of
independent environments. Each column is one environment with its own neural state
and its own Bernoulli-input RNG, so an episode's features do not depend on which
other episodes share the batch. With ``ticks=4`` and the descending readout, a
column reproduces ``V11Controller.rates`` of the frozen v1.2 controller bit for bit
(tested), because the same CSR matrix, update order and RNG calls are used.

``SensoryFeatures`` is the no-SNN control: the engineered observation itself.
"""

import numpy as np

from flyarcade.v11.controller import V11Controller
from flyarcade.v12.frozen_core import FrozenCore
from flyarcade_v14.environments import encoder_for

READOUTS = ("descending", "descending+cx", "all")
RATE_OFFSET = 0.25  # inherited from V11Controller.rates


def rng_for_episode(env_seed):
    """Same derivation as v1.2 ``episode``: the neural RNG is tied to the episode seed."""
    return np.random.default_rng(env_seed + 1_000_000_000)


class FlyFeatures:
    kind = "fly"

    def __init__(
        self,
        graph,
        observation_width,
        batch,
        *,
        task,
        ticks=4,
        readout="descending",
        standardizer=None,
        background=0.18,
        stimulus=1.5,
    ):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        self.encoder = encoder_for(task)
        reference = V11Controller(
            graph,
            0,
            stage="readout",
            core_lr=0,
            encoder=self.encoder,
            channels=len(self.encoder(np.zeros(observation_width))),
            observation_width=observation_width,
            background=background,
            stimulus=stimulus,
        )
        frozen = FrozenCore(reference.core)
        self.graph = graph
        self.n = graph.n
        self.matrix = frozen._matrix
        self.weights = frozen._weights
        self.base_magnitude = reference.core.magnitude.copy()
        self.inputs = reference.inputs
        self.outputs = reference.outputs
        self.intrinsic = reference.intrinsic
        self.input_channels = reference.input_channels
        self.background, self.stimulus = float(background), float(stimulus)
        self.ticks, self.readout = int(ticks), readout
        if self.ticks < 1:
            raise ValueError("at least one tick per action required")
        if readout == "descending":
            self.readout_index = self.outputs
        elif readout == "descending+cx":
            self.readout_index = np.concatenate([self.outputs, self.intrinsic])
        else:
            self.readout_index = np.arange(self.n)
        self.width = len(self.readout_index)
        self.standardizer = standardizer
        if standardizer is not None and standardizer.width != self.width:
            raise ValueError("standardizer width does not match the readout")
        self.observation_width = int(observation_width)
        self.batch = int(batch)
        self._masked_key, self._masked = None, None
        self.voltage = np.zeros((self.n, self.batch))
        self.spikes = np.zeros((self.n, self.batch))
        self.refractory = np.zeros((self.n, self.batch), dtype=int)
        self.rngs = [np.random.default_rng(0) for _ in range(self.batch)]

    # State is explicit so trainers can checkpoint and restore it exactly.
    def state(self):
        return {
            "voltage": self.voltage.copy(),
            "spikes": self.spikes.copy(),
            "refractory": self.refractory.copy(),
            "rngs": [r.bit_generator.state for r in self.rngs],
        }

    def load(self, state):
        self.voltage = state["voltage"].copy()
        self.spikes = state["spikes"].copy()
        self.refractory = state["refractory"].copy()
        for rng, bit_state in zip(self.rngs, state["rngs"], strict=True):
            rng.bit_generator.state = bit_state

    def reset(self, column, env_seed):
        self.voltage[:, column] = 0
        self.spikes[:, column] = 0
        self.refractory[:, column] = 0
        self.rngs[column] = rng_for_episode(env_seed)

    def reset_with_rng(self, column, rng):
        self.voltage[:, column] = 0
        self.spikes[:, column] = 0
        self.refractory[:, column] = 0
        self.rngs[column] = rng

    def _matrix_for(self, edge_mask):
        if edge_mask is None:
            return self.matrix
        if self._masked_key is not edge_mask:
            from scipy.sparse import csr_matrix

            g = self.graph
            self._masked = csr_matrix((self.weights * edge_mask, (g.post, g.pre)), shape=(g.n, g.n))
            self._masked_key = edge_mask
        return self._masked

    def raw_rates(self, observations, columns=None, *, noise=0, edge_mask=None, neuron_mask=None):
        """Advance the selected columns by ``ticks`` and return mean-offset rates."""
        columns = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        observations = np.asarray(observations, dtype=float).reshape(len(columns), -1)
        if not np.isfinite(observations).all():
            raise ValueError("finite observations required")
        full = len(columns) == self.batch and np.array_equal(columns, np.arange(self.batch))
        if noise:
            observations = np.stack(
                [
                    np.clip(o + self.rngs[c].normal(0, noise, len(o)), 0, 1)
                    for o, c in zip(observations, columns, strict=True)
                ]
            )
        encoded = np.stack([self.encoder(o) for o in observations])[:, self.input_channels]
        voltage = self.voltage if full else self.voltage[:, columns]
        spikes = self.spikes if full else self.spikes[:, columns]
        refractory = self.refractory if full else self.refractory[:, columns]
        matrix = self._matrix_for(edge_mask)
        counts = np.zeros((self.width, len(columns)))
        for _ in range(self.ticks):
            drive = np.full((self.n, len(columns)), self.background)
            draws = np.stack([self.rngs[c].random(len(self.inputs)) for c in columns], axis=1)
            drive[self.inputs] += self.stimulus * (draws < encoded.T)
            current = matrix @ spikes
            ready = refractory == 0
            refractory = np.maximum(refractory - 1, 0)
            voltage = np.where(ready, 0.85 * voltage + drive + current, 0)
            voltage = np.clip(voltage, -3, 4)
            fired = (voltage >= 1) & ready
            if neuron_mask is not None:
                fired &= neuron_mask[:, None]
                voltage[~neuron_mask] = 0
            voltage[fired] = 0
            refractory[fired] = 1
            spikes = fired.astype(float)
            counts += spikes[self.readout_index]
        if full:
            self.voltage, self.spikes, self.refractory = voltage, spikes, refractory
        else:
            self.voltage[:, columns] = voltage
            self.spikes[:, columns] = spikes
            self.refractory[:, columns] = refractory
        return (counts / self.ticks - RATE_OFFSET).T

    def __call__(self, observations, columns=None, **kwargs):
        raw = self.raw_rates(observations, columns, **kwargs)
        if self.standardizer is None:
            return raw
        s = self.standardizer
        return np.clip((raw - s.mean) / s.scale, -s.clip, s.clip) * s.gain


class SensoryFeatures:
    """Engineered observations centred to [-1, 1]; stateless, no spiking network."""

    kind = "sensory"

    def __init__(self, observation_width, batch):
        self.width = self.observation_width = int(observation_width)
        self.batch = int(batch)
        self.rngs = [np.random.default_rng(0) for _ in range(self.batch)]

    def state(self):
        return {"rngs": [r.bit_generator.state for r in self.rngs]}

    def load(self, state):
        for rng, bit_state in zip(self.rngs, state["rngs"], strict=True):
            rng.bit_generator.state = bit_state

    def reset(self, column, env_seed):
        self.rngs[column] = rng_for_episode(env_seed)

    def reset_with_rng(self, column, rng):
        self.rngs[column] = rng

    def __call__(self, observations, columns=None, *, noise=0, edge_mask=None, neuron_mask=None):
        if edge_mask is not None or neuron_mask is not None:
            raise ValueError("sensory-only control has no neural core to lesion")
        columns = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        obs = np.asarray(observations, dtype=float).reshape(len(columns), -1)
        if not np.isfinite(obs).all():
            raise ValueError("finite observations required")
        if noise:
            obs = np.stack(
                [
                    np.clip(o + self.rngs[c].normal(0, noise, len(o)), 0, 1)
                    for o, c in zip(obs, columns, strict=True)
                ]
            )
        return 2 * np.clip(obs, 0, 1) - 1
