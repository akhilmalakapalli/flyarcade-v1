"""v1.5 feature sources: readout populations, sensory control, stacking, standardizers.

``Fly15`` reuses the frozen v1.4 ``FlyFeatures`` LIF core unchanged (same CSR matrix,
update order and RNG calls) and only changes which neurons are read out:

* ``descending``   1293 descending neurons (historical default)
* ``all``          all 2040 neurons, including the 218 directly stimulated inputs
* ``visual``       the 218 visual-projection input neurons only (leakage control)
* ``nonvisual``    descending + central-brain intrinsic (all 2040 minus inputs)
* ``cb_intrinsic`` 529 central-brain intrinsic neurons (largely CX/LAL types)
* ``visual_cb``    visual inputs + central-brain intrinsic
* ``pooled``       mean rate per cell type over the non-visual neurons

``Sensory15`` is the no-SNN control: the same task encoding that drives the input
neurons, centred to [-1, 1]. ``Stacked`` concatenates the last k feature vectors of
each column (temporal feature stacking) and resets with the episode.
"""

import json
from pathlib import Path

import numpy as np

from flyarcade.v11.features import Standardizer
from flyarcade_v14.environments import TARGETS, encoder_for
from flyarcade_v14.features import FlyFeatures

READOUTS = ("descending", "all", "visual", "nonvisual", "cb_intrinsic", "visual_cb", "pooled")


def rng_for_episode(env_seed):
    return np.random.default_rng(env_seed + 1_000_000_000)


def neuron_types(graph):
    p = graph.provenance
    neurons = {n["bodyId"]: n for n in (p.get("neurons") or p["parent"]["neurons"])}
    return [neurons[int(i)] for i in graph.neuron_ids]


class Fly15(FlyFeatures):
    kind = "fly"

    def __init__(self, graph, task, batch, *, ticks, readout, standardizer=None):
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        super().__init__(graph, TARGETS[task].width, batch, task=task, ticks=ticks, readout="all")
        self.task = task
        self.readout = readout
        n = self.n
        visual = np.asarray(self.inputs)
        nonvisual = np.sort(np.concatenate([self.outputs, self.intrinsic]))
        self.pool = None
        index = {
            "descending": np.asarray(self.outputs),
            "all": np.arange(n),
            "visual": visual,
            "nonvisual": nonvisual,
            "cb_intrinsic": np.asarray(self.intrinsic),
            "visual_cb": np.sort(np.concatenate([visual, self.intrinsic])),
            "pooled": nonvisual,
        }[readout]
        # raw_rates in the parent accumulates spikes of readout_index; keep "all" there
        # and subset afterwards so the simulated core is identical for every readout.
        self.select = index
        if readout == "pooled":
            types = neuron_types(graph)
            labels = [types[i].get("type") or f"untyped-{i}" for i in nonvisual]
            groups = sorted(set(labels))
            pool = np.zeros((len(groups), len(nonvisual)))
            for j, label in enumerate(labels):
                pool[groups.index(label), j] = 1.0
            self.pool = pool / pool.sum(axis=1, keepdims=True)
            self.pool_groups = groups
        self.width = len(self.pool) if self.pool is not None else len(index)
        self.standardizer = standardizer
        if standardizer is not None and standardizer.width != self.width:
            raise ValueError("standardizer width does not match the readout")

    def all_rates(self, observations, columns=None, **kwargs):
        """Mean-offset rates of all 2040 neurons (advances the core)."""
        width, self.width = self.width, self.n  # parent sizes its spike counter by width
        try:
            return FlyFeatures.raw_rates(self, observations, columns, **kwargs)
        finally:
            self.width = width

    def project(self, all_rates):
        x = all_rates[:, self.select]
        if self.pool is not None:
            x = x @ self.pool.T
        return x

    def raw_rates(self, observations, columns=None, **kwargs):
        return self.project(self.all_rates(observations, columns, **kwargs))

    def standardize(self, raw):
        s = self.standardizer
        if s is None:
            return raw
        return np.clip((raw - s.mean) / s.scale, -s.clip, s.clip) * s.gain

    def __call__(self, observations, columns=None, **kwargs):
        return self.standardize(self.raw_rates(observations, columns, **kwargs))


class Sensory15:
    """Task encoding (the input-neuron drive probabilities) with no spiking network."""

    kind = "sensory"

    def __init__(self, task, batch, **_):
        self.task = task
        self.encoder = encoder_for(task)
        self.width = len(self.encoder(np.zeros(TARGETS[task].width)))
        self.batch = int(batch)
        self.rngs = [np.random.default_rng(0) for _ in range(self.batch)]
        self.standardizer = None

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
            raise ValueError("sensory control has no neural core to lesion")
        columns = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        obs = np.asarray(observations, dtype=float).reshape(len(columns), -1)
        if noise:
            obs = np.stack(
                [
                    np.clip(o + self.rngs[c].normal(0, noise, len(o)), 0, 1)
                    for o, c in zip(obs, columns, strict=True)
                ]
            )
        return 2 * np.stack([np.clip(self.encoder(o), 0, 1) for o in obs]) - 1


class Stacked:
    """Concatenate the last ``k`` feature vectors per column; zero-padded after reset."""

    def __init__(self, base, k):
        self.base, self.k = base, int(k)
        self.batch = base.batch
        self.inner = base.width
        self.width = self.inner * self.k
        self.kind = base.kind
        self.history = np.zeros((self.batch, self.k, self.inner))

    @property
    def base_magnitude(self):
        return self.base.base_magnitude

    def state(self):
        return {"base": self.base.state(), "history": self.history.copy()}

    def load(self, state):
        self.base.load(state["base"])
        self.history = state["history"].copy()

    def reset(self, column, env_seed):
        self.base.reset(column, env_seed)
        self.history[column] = 0

    def reset_with_rng(self, column, rng):
        self.base.reset_with_rng(column, rng)
        self.history[column] = 0

    def __call__(self, observations, columns=None, **kwargs):
        columns = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        phi = self.base(observations, columns, **kwargs)
        h = self.history[columns]
        h = np.concatenate([phi[:, None, :], h[:, :-1, :]], axis=1)
        self.history[columns] = h
        return h.reshape(len(columns), -1)


def standardizer_path(task, ticks, readout, floor):
    tag = f"f{floor:g}".replace(".", "p")
    return Path(f"artifacts/v15/standardizers/{task}-t{ticks}-{readout}-{tag}.json")


def load_standardizer(path):
    data = json.loads(Path(path).read_text())
    return Standardizer(
        np.asarray(data["mean"]), np.asarray(data["scale"]), clip=data["clip"], normalise=False
    )


def save_standardizer(path, standardizer, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": standardizer.mean.tolist(),
        "scale": standardizer.scale.tolist(),
        "clip": standardizer.clip,
        "normalise": False,
    }
    path.write_text(json.dumps(payload) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
