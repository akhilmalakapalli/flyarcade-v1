"""New-namespace regression checks; no historical confirmatory episodes execute."""

import json
from pathlib import Path

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.games import LaneGame
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.features import Standardizer
from flyarcade.v11.snake import SnakeGame
from flyarcade.v12.environments import Flappy, Pong
from flyarcade_v14 import study
from flyarcade_v14.environments import TARGETS, encoder_for, make_env
from flyarcade_v14.features import FlyFeatures
from flyarcade_v14.policy import ActorCritic


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


@pytest.mark.parametrize("task", TARGETS)
def test_adapters_leave_frozen_target_trajectories_unchanged(task):
    seed = study.seed_for(task, "software")
    old = (
        LaneGame(task, seed)
        if task in ("catch", "dodge")
        else SnakeGame(seed)
        if task == "snake"
        else {"pong": Pong, "flappy": Flappy}[task](seed)
    )
    new = make_env(task, seed)
    rng = np.random.default_rng(seed)
    while not old.done:
        np.testing.assert_array_equal(old.observe(), new.observe())
        action = int(rng.integers(new.actions))
        a = old.step(action)
        b = new.step(action)
        assert a[1:3] == b[1:3]
        assert all(b[3][k] == v for k, v in a[3].items())
    assert new.done


@pytest.mark.parametrize("task", TARGETS)
def test_feature_mapping_and_core_match_historical_controller(graph, task):
    seed = study.seed_for(task, "software")
    env = make_env(task, seed)
    encoder = encoder_for(task)
    reference = V11Controller(
        graph,
        stage="readout",
        encoder=encoder,
        channels=len(encoder(env.observe())),
        observation_width=env.width,
        actions=env.actions,
    )
    reference.rng = np.random.default_rng(seed + 1000000000)
    source = FlyFeatures(graph, env.width, 1, task=task, ticks=4)
    source.reset(0, seed)
    for _ in range(6):
        np.testing.assert_array_equal(
            reference.rates(env.observe()), source(env.observe()[None])[0]
        )
    np.testing.assert_array_equal(reference.core.magnitude, source.base_magnitude)


def test_seed_blocks_disjoint_including_derived_rng_and_confirm_forbidden():
    intervals = []
    for task in TARGETS:
        for purpose in study.PURPOSES:
            for offset in (0, 1000000000, 2000000000):
                start = study.seed_for(task, purpose) + offset
                end = study.seed_for(task, purpose, 2, 999999) + offset
                assert start > 12000000000
                assert all(end < a or start > b for a, b in intervals)
                intervals.append((start, end))
    for purpose in ("conf_train", "conf_eval", "future_confirmatory"):
        with pytest.raises(ValueError):
            study.seed_for("snake", purpose)
    with pytest.raises(ValueError):
        study.complete_config({"task": "snake", "train_purpose": "final_validation"})


def test_linear_gradient_finite_difference():
    rng = np.random.default_rng(1)
    m = ActorCritic(5, 3, arch="linear", seed=4)
    x = rng.normal(size=(4, 5))
    dl = rng.normal(size=(4, 3))
    dv = rng.normal(size=4)
    _, _, cache = m.forward(x)
    grads = m.backward(cache, dl, dv)

    def loss():
        logits, v, _ = m.forward(x)
        return (logits * dl).sum() + (v * dv).sum()

    for key, array in m.params.items():
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + 1e-6
            plus = loss()
            array[index] = original - 1e-6
            minus = loss()
            array[index] = original
            assert grads[key][index] == pytest.approx((plus - minus) / 2e-6, abs=1e-7)


@pytest.mark.parametrize("arch", ["linear", "mlp", "gru"])
def test_resume_next_update_exact_and_evaluation_frozen(graph, tmp_path, monkeypatch, arch):
    std = Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False)

    def source(config, batch, graph_arg=None):
        return FlyFeatures(graph, 3, batch, task="catch", standardizer=std)

    monkeypatch.setattr(study, "make_source", source)
    config = {
        "task": "catch",
        "arch": arch,
        "hidden": 8,
        "core": 4,
        "envs": 2,
        "steps": 4,
        "chunk": 2,
        "epochs": 1,
        "minibatches": 2,
        "transitions": 32,
    }
    trainer = study.Trainer(config, graph)
    original = study.array_hash(trainer.source.base_magnitude)
    trainer.train_step()
    path = tmp_path / "checkpoint.pkl"
    trainer.save(path)
    clone = study.Trainer(config, graph)
    clone.load(path)
    trainer.train_step()
    clone.train_step()
    assert study.params_hash(trainer.model.params) == study.params_hash(clone.model.params)
    assert study.array_hash(trainer.source.base_magnitude) == original
    assert study.params_hash(trainer.model.params) != study.params_hash(trainer.initial_params)
    before = study.params_hash(trainer.model.params)
    seeds = [study.seed_for("catch", "software", index=i) for i in range(3)]
    a = study.evaluate_policy(
        trainer.model, trainer.model.params, source(config, 2), "catch", seeds
    )
    b = study.evaluate_policy(clone.model, clone.model.params, source(config, 2), "catch", seeds)
    assert a == b and study.params_hash(trainer.model.params) == before


def test_registered_budget_balanced_and_fixed_substrate():
    plan = json.loads(Path("experiments/v14/development_plan.json").read_text())
    assert plan["training_seeds"] == [0, 1, 2] and plan["primary_transitions"] % 2048 == 0
    assert not plan["confirmatory_testing"]
    assert plan["primary_learners"] == ["linear", "mlp", "gru"]
    with pytest.raises(ValueError):
        study.complete_config({"task": "pong", "source": "sensory"})
    with pytest.raises(ValueError):
        study.complete_config({"task": "pong", "shaping": 1.0})


def test_selection_rejects_lucky_collapsed_seed_and_prefers_consistency():
    from flyarcade_v14.selection import select, summarize

    def record(arch, scores, probe=0.2):
        c = {"arch": arch, "ticks": 4, "readout": "descending", "transitions": 65536}
        rows = [
            {
                "status": "COMPLETE",
                "finite": True,
                "score": s,
                "state_probe": {"score": probe},
                "dominant_action_fraction": 0.7,
            }
            for s in scores
        ]
        return summarize(arch, "primary", c, rows)

    linear = record("linear", [0.7, 0.7, 0.7])
    mlp = record("mlp", [0.5, 0.6, 1.0])
    collapsed = record("gru", [0.99, 0.99, 0.99], probe=0)
    assert select([linear, mlp, collapsed])["name"] == "linear"
    assert select([record("gru", [0.7] * 3), record("mlp", [0.7] * 3), linear])["name"] == "linear"


@pytest.mark.parametrize("arch", ["linear", "mlp", "gru"])
def test_imitation_changes_only_readout_on_fresh_development_data(graph, monkeypatch, arch):
    from flyarcade.resources import ResourceGuard
    from flyarcade_v14 import imitation

    std = Standardizer(np.zeros(24), np.full(24, 0.1), normalise=False)

    def source(config, batch, graph_arg=None):
        return FlyFeatures(graph, 3, batch, task="catch", standardizer=std)

    monkeypatch.setattr(study, "make_source", source)
    monkeypatch.setattr(imitation, "make_source", source)
    trainer = study.Trainer({"task": "catch", "arch": arch, "hidden": 8, "core": 4, "envs": 2})
    before = study.params_hash(trainer.model.params)
    core = study.array_hash(trainer.source.base_magnitude)
    result = imitation.warm_start(trainer, ResourceGuard(), count=256, passes=1)
    assert result["transitions"] == 256
    assert study.params_hash(trainer.model.params) != before
    assert study.array_hash(trainer.source.base_magnitude) == core
    assert trainer.transitions == 0 and trainer.optimizer.t == 0
