import json
from pathlib import Path

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.v11.features import Standardizer
from flyarcade_v13.study import Trainer, params_hash


@pytest.fixture
def config(tmp_path):
    std = Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False)
    path = tmp_path / "std.json"
    std.save(path)
    return lambda **kw: {
        "task": "snake",
        "envs": 3,
        "steps": 8,
        "minibatches": 2,
        "epochs": 2,
        "chunk": 4,
        "hidden": 16,
        "core": 8,
        "transitions": 24 * 4,
        "shaping": 0.5,
        "standardizer": str(path),
        **kw,
    }


@pytest.fixture
def graph():
    g = synthetic_graph(n=64, edges=256, seed=4)
    g.provenance["neurons"] = [
        {
            "bodyId": int(body),
            "superclass": "visual_projection" if i < 16 else "descending_neuron" if i < 40 else "x",
            "consensusNt": "acetylcholine" if i % 3 else "gaba",
        }
        for i, body in enumerate(g.neuron_ids)
    ]
    return g


@pytest.mark.parametrize("arch", ["mlp", "gru"])
def test_resume_from_checkpoint_is_exact(graph, config, tmp_path, arch):
    straight = Trainer(config(arch=arch), graph)
    while not straight.done():
        straight.train_step()
    resumed = Trainer(config(arch=arch), graph)
    resumed.train_step()
    resumed.train_step()
    resumed.save(tmp_path / "ckpt.pkl")
    fresh = Trainer(config(arch=arch), graph)
    fresh.load(tmp_path / "ckpt.pkl")
    while not fresh.done():
        fresh.train_step()
    assert params_hash(fresh.model.params) == params_hash(straight.model.params)
    assert params_hash(fresh.optimizer.m) == params_hash(straight.optimizer.m)
    strip = lambda rows: [
        {k: v for k, v in r.items() if "wall" not in k and "per_second" not in k} for r in rows
    ]  # noqa: E731
    assert strip(fresh.history) == strip(straight.history)
    assert fresh.completed == straight.completed


def test_checkpoint_rejects_other_configuration(graph, config, tmp_path):
    trainer = Trainer(config(), graph)
    trainer.save(tmp_path / "ckpt.pkl")
    other = Trainer(config(lr=1e-3), graph)
    with pytest.raises(ValueError):
        other.load(tmp_path / "ckpt.pkl")


def test_protected_historical_files_unchanged():
    snapshot = json.loads(Path("artifacts/v13/historical_hashes.json").read_text())
    import hashlib

    tracked_sources = [
        p for p in snapshot["files"] if p.startswith(("src/", "scripts/", "experiments/", "tests/"))
    ]
    assert tracked_sources
    for path in tracked_sources:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == snapshot["files"][path], path
