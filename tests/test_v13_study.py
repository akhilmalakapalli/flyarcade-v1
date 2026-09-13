import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.v11.features import Standardizer
from flyarcade.v12 import study as v12study
from flyarcade_v13 import study
from flyarcade_v13.controller import FlyFeatures, SensoryFeatures
from flyarcade_v13.environments import TARGETS, make_env
from flyarcade_v13.study import (
    CONFIRMATORY_PURPOSES,
    DEVELOPMENT_PURPOSES,
    Trainer,
    evaluate_baseline,
    evaluate_policy,
    fit_standardizer,
    params_hash,
    purpose_range,
    seed_for,
    state_probe,
)


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


def _std(width):
    return Standardizer(np.zeros(width), np.full(width, 0.1), normalise=False)


def test_batched_core_reproduces_frozen_v12_controller_bit_for_bit(graph):
    task = "pong"
    reference = v12study.build(graph, task, 0, _std(24))
    batched = FlyFeatures(graph, TARGETS[task].width, 3, ticks=4, standardizer=_std(24))
    seeds = [11, 12, 13]
    envs = [make_env(task, s) for s in seeds]
    for c, s in enumerate(seeds):
        batched.reset(c, s)
    controllers = []
    for s in seeds:
        c = v12study.build(graph, task, 0, _std(24))
        c.reset()
        c.rng = np.random.default_rng(s + 1_000_000_000)
        controllers.append(c)
    np.testing.assert_array_equal(reference.core.magnitude, batched.base_magnitude)
    for step in range(30):
        obs = np.stack([e.observe() for e in envs])
        columns = [0, 2] if step % 4 == 3 else [0, 1, 2]
        phi = batched(obs[columns], columns)
        for k, c in enumerate(columns):
            np.testing.assert_array_equal(phi[k], controllers[c].rates(obs[c]))
        for c in columns:
            envs[c].step(1)


def test_lesions_and_noise_match_frozen_v12_controller(graph):
    task = "snake"
    width = 34
    rng = np.random.default_rng(0)
    edge_mask, neuron_mask = rng.random(graph.e) >= 0.1, rng.random(graph.n) >= 0.1
    for kwargs in ({"noise": 0.1}, {"edge_mask": edge_mask}, {"neuron_mask": neuron_mask}):
        controller = v12study.build(graph, task, 0, _std(24))
        controller.rng = np.random.default_rng(99 + 1_000_000_000)
        batched = FlyFeatures(graph, TARGETS[task].width, 1, standardizer=_std(24))
        batched.reset(0, 99)
        env = make_env(task, 99)
        for _ in range(10):
            obs = env.observe()
            expected = controller.rates(obs, **kwargs)
            np.testing.assert_array_equal(batched(obs[None], [0], **kwargs)[0], expected)
        assert width == 2 * TARGETS[task].width


@pytest.mark.parametrize("ticks", [4, 8, 12, 16])
def test_ticks_deterministic_and_dimensions(graph, ticks):
    outputs = []
    for _ in range(2):
        source = FlyFeatures(graph, 7, 2, ticks=ticks)
        source.reset(0, 5)
        source.reset(1, 6)
        outputs.append(source(np.full((2, 7), 0.5)))
    np.testing.assert_array_equal(*outputs)
    assert outputs[0].shape == (2, 24)
    counts = (outputs[0] + 0.25) * ticks
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    wide = FlyFeatures(graph, 7, 1, readout="descending+cx")
    assert wide.width == 24 + 24 and FlyFeatures(graph, 7, 1, readout="all").width == 64


def test_seed_blocks_disjoint_from_history_and_confirmatory():
    v12_low = 100_000_000
    v12_high = 100_000_000 + 4 * 10_000_000
    ranges = []
    for task in TARGETS:
        for purpose in DEVELOPMENT_PURPOSES + CONFIRMATORY_PURPOSES:
            lo, hi = purpose_range(task, purpose)
            assert lo >= 200_000_000 and (hi <= v12_low or lo >= v12_high)
            ranges.append((lo, hi, purpose))
    ranges.sort()
    for (lo1, hi1, _), (lo2, _, _) in zip(ranges, ranges[1:], strict=False):
        assert hi1 <= lo2
    dev = {p for p in DEVELOPMENT_PURPOSES}
    assert not dev & set(CONFIRMATORY_PURPOSES)
    with pytest.raises(ValueError):
        seed_for("pong", "screen_train", 20)
    with pytest.raises(ValueError):
        seed_for("pong", "screen_train", 0, 250_000)


def test_standardizer_uses_only_feature_fit_seeds(graph):
    a, meta = fit_standardizer(graph, "pong", ticks=4, min_samples=10, batch=2)
    b, _ = fit_standardizer(graph, "pong", ticks=4, min_samples=10, batch=2)
    np.testing.assert_array_equal(a.mean, b.mean)
    assert meta["seeds"] == [
        seed_for("pong", "feature_fit", index=i) for i in range(len(meta["seeds"]))
    ]
    assert all(
        purpose_range("pong", "feature_fit")[0] <= s < purpose_range("pong", "feature_fit")[1]
        for s in meta["seeds"]
    )


def _config(task="pong", **extra):
    return {
        "task": task,
        "envs": 4,
        "steps": 8,
        "minibatches": 2,
        "epochs": 2,
        "chunk": 4,
        "hidden": 16,
        "core": 8,
        "transitions": 96,
        **extra,
    }


@pytest.fixture
def fly_config(graph, tmp_path):
    std = _std(24)
    path = tmp_path / "std.json"
    std.save(path)
    return lambda **kw: _config(standardizer=str(path), **kw)


def test_no_raw_observation_reaches_primary_fly_policy(graph, fly_config):
    trainer = Trainer(fly_config(), graph)
    assert isinstance(trainer.source, FlyFeatures)
    assert trainer.model.input_width == len(trainer.source.outputs) == 24
    assert trainer.features.shape == (4, 24)
    sensory = Trainer(_config(source="sensory"))
    assert isinstance(sensory.source, SensoryFeatures) and sensory.model.input_width == 7


@pytest.mark.parametrize("arch", ["mlp", "gru"])
def test_evaluation_never_updates_and_uses_fresh_state(graph, fly_config, arch):
    trainer = Trainer(fly_config(arch=arch), graph)
    trainer.train_step()
    before_params = params_hash(trainer.model.params)
    before_opt = params_hash(trainer.optimizer.m)
    state = trainer.source.state()
    source = study.make_source(trainer.config, 4, graph)
    seeds = [seed_for("pong", "screen_eval", index=i) for i in range(5)]
    rows = evaluate_policy(trainer.model, trainer.model.params, source, "pong", seeds)
    again = evaluate_policy(trainer.model, trainer.model.params, source, "pong", seeds)
    assert rows == again and len(rows) == 5
    probe = state_probe(trainer.model, trainer.model.params, source, "pong", "dev_probe", 0)
    assert probe["states"] > 0
    assert params_hash(trainer.model.params) == before_params
    assert params_hash(trainer.optimizer.m) == before_opt
    np.testing.assert_array_equal(trainer.source.state()["voltage"], state["voltage"])


def test_evaluation_independent_of_batch_composition(graph, fly_config):
    trainer = Trainer(fly_config(arch="gru"), graph)
    seeds = [seed_for("pong", "screen_eval", index=i) for i in range(5)]
    wide = evaluate_policy(
        trainer.model,
        trainer.model.params,
        study.make_source(trainer.config, 4, graph),
        "pong",
        seeds,
    )
    narrow = evaluate_policy(
        trainer.model,
        trainer.model.params,
        study.make_source(trainer.config, 1, graph),
        "pong",
        seeds,
    )
    assert wide == narrow


def test_baselines_deterministic():
    seeds = [seed_for("snake", "screen_eval", index=i) for i in range(3)]
    for kind in ("random", "reference", "heuristic_v12"):
        assert evaluate_baseline("snake", seeds, kind) == evaluate_baseline("snake", seeds, kind)


def test_frozen_learner_does_not_change(graph, fly_config):
    trainer = Trainer(fly_config(learn=False), graph)
    start = params_hash(trainer.model.params)
    while not trainer.done():
        trainer.train_step()
    assert params_hash(trainer.model.params) == start == params_hash(trainer.initial_params)


def test_curriculum_schedule_and_transition_accounting(graph, fly_config):
    trainer = Trainer(fly_config(task="snake", curriculum=[["size4", 0.5], ["target", 0.5]]), graph)
    levels = []
    while not trainer.done():
        levels.append(trainer.train_step()["level"])
    assert trainer.transitions == 96 and sum(trainer.level_transitions.values()) == 96
    assert trainer.level_transitions["size4"] >= 48
    with pytest.raises(ValueError):
        Trainer(fly_config(curriculum=[["target", 0.5], ["size4", 0.5]]), graph)


def _fake_result(after, random, reference=1.0, before=0.1, probe=0.3, counts=(10, 10), entropy=0.5):
    rows = lambda v: [{"success": v, "action_counts": list(counts)}]  # noqa: E731
    return {
        "evaluations": {
            "after": rows(after),
            "before": rows(before),
            "random": rows(random),
            "reference": rows(reference),
        },
        "history": [{"entropy": entropy}],
        "state_probe": {"score": probe},
        "finite": True,
    }


def test_task_gate_criteria():
    from flyarcade_v13.gates import task_gate

    passing = [_fake_result(0.5, 0.2) for _ in range(5)]
    assert task_gate(passing)["passed"]
    three = [_fake_result(0.5, 0.2)] * 3 + [_fake_result(0.24, 0.2)] * 2
    assert not task_gate(three)["checks"]["seeds_beating_random_margin"]
    exact = [_fake_result(0.25, 0.2)] * 5  # margin must be strictly exceeded
    assert not task_gate(exact)["checks"]["after_gt_random_plus_margin"]
    small_gap = [_fake_result(0.30, 0.2, reference=0.9)] * 5
    assert not task_gate(small_gap)["checks"]["gap_closure_ge_0.15"]
    collapsed = [_fake_result(0.5, 0.2, counts=(99, 1))] * 5
    assert not task_gate(collapsed)["checks"]["no_constant_action_collapse"]
    no_probe = [_fake_result(0.5, 0.2, probe=0.0)] * 5
    assert not task_gate(no_probe)["checks"]["state_dependence_gt_0.05"]
    entropy_dead = [_fake_result(0.5, 0.2, entropy=0.0)] * 5
    assert not task_gate(entropy_dead)["checks"]["nonzero_entropy"]
