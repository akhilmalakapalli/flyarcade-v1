import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.v11.core import EpropCore
from flyarcade.v11.features import Standardizer
from flyarcade.v12.frozen_core import FrozenCore
from flyarcade.v12.study import (
    array_hash,
    build,
    episode,
    evaluate,
    fit_standardizer,
    graph_hash,
    restore_checkpoint,
    save_checkpoint,
    seed_for,
    state_probe,
)


@pytest.fixture
def graph():
    g = synthetic_graph(n=64, edges=256, seed=4)
    g.provenance["neurons"] = [
        {
            "bodyId": int(body),
            "superclass": "visual_projection" if i < 16 else "descending_neuron",
            "consensusNt": "acetylcholine",
        }
        for i, body in enumerate(g.neuron_ids)
    ]
    return g


def test_sparse_optimization_bit_identical_with_masks(graph):
    rng = np.random.default_rng(123)
    signs = rng.choice([-1, 0, 1], graph.n)
    a, b = EpropCore(graph, signs), FrozenCore(EpropCore(graph, signs))
    edges = rng.random(graph.e) > 0.1
    neurons = rng.random(graph.n) > 0.1
    for i in range(100):
        kwargs = {} if i < 30 else {"edge_mask": edges} if i < 60 else {"neuron_mask": neurons}
        drive = rng.uniform(0, 1.5, graph.n)
        np.testing.assert_array_equal(a.tick(drive, **kwargs), b.tick(drive, **kwargs))
        np.testing.assert_array_equal(a.voltage, b.voltage)
    with pytest.raises(ValueError):
        b.tick(np.zeros(graph.n), plastic=True)


def test_stage_a_learns_only_readout_and_eval_frozen(graph):
    std = Standardizer(np.zeros(48), np.ones(48) * 0.1)
    c = build(graph, "pong", standardizer=std, config={"actor_lr": 0.4, "critic_lr": 0.2})
    original = array_hash(c.core.magnitude), array_hash(c.motor.actor)
    episode(c, "pong", seed_for("pong", "dev_train"), training=True)
    assert array_hash(c.core.magnitude) == original[0]
    assert array_hash(c.motor.actor) != original[1]
    before = [array_hash(x) for x in (c.core.magnitude, c.motor.actor, c.motor.critic)]
    evaluate(c, "pong", [seed_for("pong", "dev_eval")])
    state_probe(c, "pong", 0)
    assert [array_hash(x) for x in (c.core.magnitude, c.motor.actor, c.motor.critic)] == before
    frozen = build(graph, "pong", standardizer=std)
    episode(frozen, "pong", seed_for("pong", "dev_train"), training=False)
    assert not frozen.motor.actor.any() and not frozen.motor.critic.any()


def test_standardizer_uses_only_feature_fit_seeds(graph):
    a, meta = fit_standardizer(graph, "snake", count=4)
    b, other = fit_standardizer(graph, "snake", count=4)
    np.testing.assert_array_equal(a.mean, b.mean)
    assert meta == other and meta["purpose"] == "development_feature_fit"
    assert meta["seeds"] == [seed_for("snake", "fit", index=i) for i in range(4)]
    assert not set(meta["seeds"]) & {seed_for("snake", "eval", index=i) for i in range(100)}


def test_checkpoint_reproduces_next_training_episode_and_eval(graph, tmp_path):
    std = Standardizer(np.zeros(48), np.ones(48) * 0.1)
    c = build(graph, "snake", standardizer=std)
    episode(c, "snake", seed_for("snake", "dev_train"), training=True)
    path = tmp_path / "checkpoint.npz"
    expected = {
        "graph_sha256": graph_hash(graph),
        "plan_sha256": "fixture",
        "completed_episodes": 1,
    }
    save_checkpoint(path, c, expected)
    clone = build(graph, "snake", standardizer=std)
    assert restore_checkpoint(path, clone, expected) == expected
    seed = seed_for("snake", "dev_train", index=1)
    assert episode(c, "snake", seed, training=True) == episode(clone, "snake", seed, training=True)
    np.testing.assert_array_equal(c.motor.actor, clone.motor.actor)
    seeds = [seed_for("snake", "dev_eval", index=i) for i in range(3)]
    assert evaluate(c, "snake", seeds) == evaluate(clone, "snake", seeds)
    with pytest.raises(ValueError):
        restore_checkpoint(path, clone, {"plan_sha256": "changed"})


def test_historical_tracked_files_unchanged():
    path = Path("artifacts/v12/historical_hashes.json")
    if not path.exists():
        pytest.skip("historical snapshot unavailable")
    snapshot = json.loads(path.read_text())
    for filename, digest in snapshot.items():
        if filename.startswith(("runs/", "data/")):
            continue
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename


@pytest.mark.parametrize("mismatch", [False, True])
def test_completed_trial_is_preserved(tmp_path, monkeypatch, mismatch):
    import runpy
    import sys

    import flyarcade.v12.study as study

    script = Path("scripts/v12_trial.py").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, "code_hash", lambda: "fixture-code")
    monkeypatch.setattr(
        sys,
        "argv",
        ["v12_trial.py", "--task", "flappy", "--condition", "biological", "--seed", "0"],
    )
    plan = tmp_path / "experiments/v12_run_plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"code_sha256": "fixture-code"}))
    result = tmp_path / "runs/v12-flappy-biological-0/result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "code_sha256": "different" if mismatch else "fixture-code",
                "task": "flappy",
                "condition": "biological",
                "seed": 0,
            }
        )
    )
    original = result.read_bytes()
    main = runpy.run_path(str(script))["main"]
    if mismatch:
        with pytest.raises(ValueError, match="identity mismatch"):
            main()
    else:
        assert main() == 0
    assert result.read_bytes() == original
