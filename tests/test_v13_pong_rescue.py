"""Guards for the v1.3 Pong rescue sprint: frozen core, separate calibration, seed purposes."""

import json
from pathlib import Path

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.v11.features import Standardizer
from flyarcade_v13.study import (
    CONFIRMATORY_PURPOSES,
    DEVELOPMENT_PURPOSES,
    Trainer,
    graph_hash,
    params_hash,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _config(path, **extra):
    return {
        "task": "pong",
        "envs": 4,
        "steps": 8,
        "minibatches": 2,
        "epochs": 2,
        "chunk": 4,
        "hidden": 16,
        "transitions": 96,
        "ticks": 8,
        "ent": 0.003,
        "shaping": 1.0,
        "standardizer": str(path),
        **extra,
    }


def test_training_updates_the_mlp_but_never_the_frozen_connectome_core(graph, tmp_path):
    path = tmp_path / "std.json"
    Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False).save(path)
    trainer = Trainer(_config(path), graph)
    core_before = trainer.source.base_magnitude.copy()
    weights_before = np.array(trainer.source.weights, copy=True)
    graph_before = graph_hash(graph)
    start = params_hash(trainer.model.params)
    while not trainer.done():
        trainer.train_step()
    assert params_hash(trainer.model.params) != start, "the MLP actor-critic must learn"
    np.testing.assert_array_equal(trainer.source.base_magnitude, core_before)
    np.testing.assert_array_equal(np.asarray(trainer.source.weights), weights_before)
    assert graph_hash(graph) == graph_before
    assert all(np.isfinite(v).all() for v in trainer.model.params.values())


@pytest.mark.parametrize("seed", range(5))
def test_rewired_pong_standardizers_are_fitted_on_their_own_graph(seed):
    """Never standardise a rewired graph with biological statistics (the Snake artefact)."""
    directory = ROOT / "artifacts/v13/standardizers"
    biological = Standardizer.load(directory / "pong-biological-t8-descending.json")
    path = directory / f"pong-rewired-{seed}-t8-descending.json"
    if not path.exists():
        pytest.skip("rewired Pong standardizers not generated in this checkout")
    rewired = Standardizer.load(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    bio_meta = json.loads((directory / "pong-biological-t8-descending.meta.json").read_text())
    assert not np.array_equal(rewired.mean, biological.mean)
    assert meta["topology"] == f"rewired-{seed}"
    assert meta["graph_sha256"] != bio_meta["graph_sha256"]
    # Identical procedure: same development-only seeds, same sample count and policy.
    assert meta["seeds"] == bio_meta["seeds"]
    assert meta["policy"] == bio_meta["policy"] and meta["ticks"] == 8


def test_pong_rescue_specs_keep_development_and_confirmatory_purposes_apart():
    specs = sorted((ROOT / "experiments/v13_pong/specs").rglob("*.json"))
    if not specs:
        pytest.skip("no Pong rescue specs in this checkout")
    for path in specs:
        spec = json.loads(path.read_text())
        purposes = [
            spec["config"][key]
            for key in ("train_purpose", "eval_purpose", "model_purpose", "rollout_purpose")
            if key in spec["config"]
        ] + [spec["evaluation"][k] for k in ("purpose", "probe_purpose") if k in spec["evaluation"]]
        if "confirm" in path.parts:
            assert all(p in CONFIRMATORY_PURPOSES for p in purposes), path
        else:
            assert all(p in DEVELOPMENT_PURPOSES for p in purposes), path
