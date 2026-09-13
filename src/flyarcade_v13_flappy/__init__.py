"""FlyArcade v1.3 Flappy rescue extensions, kept outside the hashed flyarcade_v13 code.

The multitask study hashes ``src/flyarcade_v13/**`` and ``scripts/v13_trial.py`` into
every checkpoint and frozen plan, so this package never edits those files. Instead
``install()`` swaps three names inside ``flyarcade_v13.study`` for config-aware
dispatchers that fall back to the originals whenever a config does not opt in:

* ``make_source``  - adds the optional ``encoding`` key (augmented Flappy sensory input);
* ``potential``    - adds the optional arrival potential for Flappy shaping;
* ``code_hash``    - extends provenance to this package and the rescue trial wrapper.

With no opt-in key, training and evaluation are bit-identical to the unpatched
pipeline (tested). Patching is process-local and done only by rescue scripts/tests.
"""

import hashlib
from pathlib import Path

import numpy as np

from flyarcade_v13 import study as _study
from flyarcade_v13.controller import FlyFeatures, SensoryFeatures
from flyarcade_v13.environments import TARGETS, make_env, reference_action
from flyarcade_v13_flappy.derived import arrival_potential, encoding

_ORIGINAL = {
    "make_source": _study.make_source,
    "potential": _study.potential,
    "code_hash": _study.code_hash,
}
ACTIVE = {"potential": "default"}
RESCUE_FILES = ("scripts/v13_flappy_trial.py",)


class TransformedFlyFeatures(FlyFeatures):
    """Frozen-core features with a deterministic transform of the current observation.

    Sensory noise, when requested, is applied to the raw observation first (same RNG
    stream as the base class) and the transform is applied afterwards.
    """

    def __init__(self, *args, transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def raw_rates(self, observations, columns=None, *, noise=0, **kwargs):
        cols = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        obs = np.asarray(observations, dtype=float).reshape(len(cols), -1)
        if noise:
            obs = np.stack(
                [
                    np.clip(o + self.rngs[c].normal(0, noise, len(o)), 0, 1)
                    for o, c in zip(obs, cols, strict=True)
                ]
            )
        transformed = np.stack([self.transform(o) for o in obs])
        return super().raw_rates(transformed, columns, noise=0, **kwargs)


class TransformedSensoryFeatures(SensoryFeatures):
    def __init__(self, observation_width, batch, *, transform):
        super().__init__(observation_width, batch)
        self.transform = transform

    def __call__(self, observations, columns=None, *, noise=0, **kwargs):
        cols = np.arange(self.batch) if columns is None else np.asarray(columns, dtype=int)
        obs = np.asarray(observations, dtype=float).reshape(len(cols), -1)
        if noise:
            obs = np.stack(
                [
                    np.clip(o + self.rngs[c].normal(0, noise, len(o)), 0, 1)
                    for o, c in zip(obs, cols, strict=True)
                ]
            )
        transformed = np.stack([self.transform(o) for o in obs])
        return super().__call__(transformed, columns, noise=0, **kwargs)


def make_source(config, batch, graph=None):
    name = config.get("encoding", "v12")
    if name == "v12":
        return _ORIGINAL["make_source"](config, batch, graph)
    task = config["task"]
    transform, width = encoding(name, TARGETS[task].width)
    if config["source"] == "sensory":
        return TransformedSensoryFeatures(width, batch, transform=transform)
    graph = graph if graph is not None else _study.load_topology(config["topology"])
    std = _study.Standardizer.load(config["standardizer"])
    return TransformedFlyFeatures(
        graph,
        width,
        batch,
        ticks=config["ticks"],
        readout=config["readout"],
        standardizer=std,
        transform=transform,
    )


def potential(env):
    if ACTIVE["potential"] == "arrival" and isinstance(env, TARGETS["flappy"]):
        return arrival_potential(env)
    return _ORIGINAL["potential"](env)


def rescue_code_files():
    return sorted(Path("src/flyarcade_v13_flappy").rglob("*.py")) + [Path(p) for p in RESCUE_FILES]


def code_hash():
    """Base multitask code hash extended with every rescue source file."""
    h = hashlib.sha256(_ORIGINAL["code_hash"]().encode())
    for path in rescue_code_files():
        if path.exists():
            h.update(str(path).encode() + path.read_bytes())
    return h.hexdigest()


def install():
    _study.make_source = make_source
    _study.potential = potential
    _study.code_hash = code_hash


def uninstall():
    for name, value in _ORIGINAL.items():
        setattr(_study, name, value)
    ACTIVE["potential"] = "default"


def activate(config):
    """Configure process-local options from a trial config (one trial per process)."""
    name = config.get("potential", "default")
    if name not in ("default", "arrival"):
        raise ValueError(f"unknown potential {name}")
    ACTIVE["potential"] = name


def fit_standardizer(
    graph,
    task,
    *,
    ticks,
    encoding_name,
    readout="descending",
    min_samples=8000,
    max_episodes=2000,
    batch=16,
):
    """The frozen v1.3 feature_fit procedure, applied to a (possibly augmented) encoding."""
    transform, width = encoding(encoding_name, TARGETS[task].width)
    if transform is None:
        return _study.fit_standardizer(graph, task, ticks=ticks, readout=readout)
    source = TransformedFlyFeatures(
        graph, width, batch, ticks=ticks, readout=readout, transform=transform
    )
    seeds, samples = [], []
    while len(samples) < min_samples and len(seeds) < max_episodes:
        block = [_study.seed_for(task, "feature_fit", index=len(seeds) + i) for i in range(batch)]
        seeds.extend(block)
        envs = [make_env(task, s) for s in block]
        behaviour = [np.random.default_rng(s + _study.BEHAVIOUR_OFFSET) for s in block]
        for c, s in enumerate(block):
            source.reset(c, s)
        while not all(e.done for e in envs):
            live = [c for c, e in enumerate(envs) if not e.done]
            rates = source.raw_rates(np.stack([envs[c].observe() for c in live]), live)
            samples.extend(rates)
            for c in live:
                env = envs[c]
                action = (
                    int(behaviour[c].integers(env.actions))
                    if env.time % 2
                    else reference_action(env)
                )
                env.step(action)
    std = _study.Standardizer(
        np.mean(samples, axis=0),
        np.maximum(np.std(samples, axis=0), 0.02),
        clip=4.0,
        normalise=False,
    )
    return std, {
        "seeds": seeds,
        "purpose": "feature_fit (development only)",
        "samples": len(samples),
        "ticks": ticks,
        "readout": readout,
        "policy": "alternating reference-policy and uniform-random actions",
        "graph_sha256": _study.graph_hash(graph),
        "normalise": False,
        "encoding": encoding_name,
    }
