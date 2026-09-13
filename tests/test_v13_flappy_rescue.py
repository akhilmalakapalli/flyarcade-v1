"""Flappy rescue extensions: opt-in only, exact where they claim to be, state-only."""

import sys
from pathlib import Path

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.v11.features import Standardizer
from flyarcade_v13 import study
from flyarcade_v13.environments import (
    _flappy_predicted_height,
    _steps_to_crossing,
    flappy_oracle,
    make_env,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flyarcade_v13_flappy as rescue  # noqa: E402
from flyarcade_v13_flappy import derived as enc  # noqa: E402

BASE_CODE_HASH = "504a571e76244f199713ffb732eedd645d4504aba13a7fd4088f6d7e2206ae59"


@pytest.fixture
def graph():
    g = synthetic_graph(n=64, edges=256, seed=4)
    g.provenance["neurons"] = [
        {
            "bodyId": int(body),
            "superclass": "visual_projection"
            if i < 16
            else "descending_neuron"
            if i < 40
            else "cb_intrinsic",
            "consensusNt": "acetylcholine" if i % 3 else "gaba",
        }
        for i, body in enumerate(g.neuron_ids)
    ]
    return g


@pytest.fixture
def installed():
    rescue.install()
    yield
    rescue.uninstall()


def _config(path, **extra):
    return {
        "task": "flappy",
        "envs": 4,
        "steps": 8,
        "minibatches": 2,
        "epochs": 2,
        "chunk": 4,
        "hidden": 16,
        "core": 8,
        "transitions": 64,
        "standardizer": str(path),
        **extra,
    }


def test_rescue_never_changes_the_hashed_multitask_code(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert rescue._ORIGINAL["code_hash"]() == BASE_CODE_HASH


def test_derived_quantities_match_the_environment_exactly():
    rng = np.random.default_rng(0)
    for seed in range(60):
        env = make_env("flappy", 90_000 + seed)
        while not env.done:
            s = enc.state_from_observation(env.observe())
            assert np.isclose(s["y"], env.y) and np.isclose(s["vy"], env.vy)
            assert int(enc.steps_to_crossing(env.x)) == _steps_to_crossing(env.x)
            height = enc.projected_no_flap_height(env.y, env.vy, _steps_to_crossing(env.x))
            assert abs(float(height) - _flappy_predicted_height(env)) < 1e-12
            augmented = enc.flappy_augmented(env.observe())
            assert augmented.shape == (13,) and (augmented >= 0).all() and (augmented <= 1).all()
            np.testing.assert_array_equal(augmented[:6], np.clip(env.observe(), 0, 1))
            env.step(flappy_oracle(env) if env.time % 3 else int(rng.integers(2)))


def test_arrival_potential_is_a_state_function_that_ignores_rng_and_future_gaps():
    a, b = make_env("flappy", 1), make_env("flappy", 2)
    for attribute in ("y", "vy", "x", "gap", "time"):
        setattr(b, attribute, getattr(a, attribute))
    assert enc.arrival_potential(a) == enc.arrival_potential(b)
    b.rng = np.random.default_rng(999)  # different future gaps, same current state
    assert enc.arrival_potential(a) == enc.arrival_potential(b)
    a.y, a.vy = a.gap, 0.0
    assert enc.arrival_potential(a) > -0.3


def test_default_configs_train_bit_identically_with_extensions_installed(graph, tmp_path):
    path = tmp_path / "std.json"
    Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False).save(path)
    reference = study.Trainer(_config(path, shaping=1.0), graph)
    reference.train_step()
    rescue.install()
    try:
        rescue.activate(_config(path, shaping=1.0))
        patched = study.Trainer(_config(path, shaping=1.0), graph)
        patched.train_step()
    finally:
        rescue.uninstall()
    assert study.params_hash(reference.model.params) == study.params_hash(patched.model.params)


def test_arrival_potential_changes_training_only_when_requested(graph, tmp_path, installed):
    path = tmp_path / "std.json"
    Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False).save(path)
    rescue.activate(_config(path, shaping=1.0))
    default = study.Trainer(_config(path, shaping=1.0), graph)
    batch_default = default.collect()
    rescue.activate(_config(path, shaping=1.0, potential="arrival"))
    arrival = study.Trainer(_config(path, shaping=1.0, potential="arrival"), graph)
    batch_arrival = arrival.collect()
    np.testing.assert_array_equal(batch_default["actions"], batch_arrival["actions"])
    assert not np.array_equal(batch_default["rewards"], batch_arrival["rewards"])


def test_augmented_source_feeds_the_frozen_core_and_keeps_the_readout(graph, tmp_path, installed):
    path = tmp_path / "std.json"
    Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False).save(path)
    trainer = study.Trainer(_config(path, encoding="flappy_augmented"), graph)
    assert isinstance(trainer.source, rescue.TransformedFlyFeatures)
    assert trainer.source.observation_width == 13 and trainer.model.input_width == 24
    core = trainer.source.base_magnitude.copy()
    start = study.params_hash(trainer.model.params)
    while not trainer.done():
        trainer.train_step()
    np.testing.assert_array_equal(trainer.source.base_magnitude, core)
    assert study.params_hash(trainer.model.params) != start
    assert trainer.model.finite()


def test_extended_code_hash_covers_rescue_files(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert rescue.code_hash() != BASE_CODE_HASH
    assert Path("src/flyarcade_v13_flappy/derived.py") in rescue.rescue_code_files()
