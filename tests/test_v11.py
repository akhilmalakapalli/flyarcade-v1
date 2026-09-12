"""Regression tests for the v1.1 actor-critic e-prop rebuild."""

import numpy as np
import pytest

from flyarcade.connectome import synthetic_graph
from flyarcade.neural.lif import LIF
from flyarcade.v11.actor_critic import ActorCritic
from flyarcade.v11.core import EpropCore
from flyarcade.v11.experiment import (
    FEATURE_FIT_SEEDS,
    confirm_evaluation_seeds,
    confirm_training_seed,
    dev_evaluation_seeds,
    dev_training_seed,
)
from flyarcade.v11.features import Standardizer


@pytest.fixture
def graph():
    return synthetic_graph(n=64, edges=256, seed=3)


def test_frozen_eprop_core_matches_v1_dynamics_exactly(graph):
    """v1.1 must change the learning rule only, never the circuit's dynamics."""
    rng = np.random.default_rng(0)
    signs = rng.choice([-1, 0, 1], graph.n)
    v1, v11 = LIF(graph, signs), EpropCore(graph, signs)
    assert np.array_equal(v1.base, v11.base)
    drive_rng = np.random.default_rng(1)
    for _ in range(50):
        drive = drive_rng.normal(0.2, 0.5, graph.n)
        assert np.array_equal(v1.tick(drive, plastic=False), v11.tick(drive, plastic=False))


def test_learning_signal_is_per_neuron_not_one_global_scalar(graph):
    """The v1 failure was one scalar applied to every synapse; reject that shape."""
    rng = np.random.default_rng(0)
    signs = rng.choice([-1, 0, 1], graph.n)
    core = EpropCore(graph, signs, plastic_edges=np.ones(graph.e, dtype=bool))
    for _ in range(6):
        core.tick(rng.normal(0.5, 0.5, graph.n), plastic=True)
    core.eligibility = np.ones_like(core.eligibility)
    signal = np.zeros(graph.n)
    signal[graph.post[0]] = 1.0
    before = core.magnitude.copy()
    core.learn(signal, learning_rate=0.01)
    changed = np.flatnonzero(before != core.magnitude)
    assert len(changed) and set(graph.post[changed]) == {graph.post[0]}
    with pytest.raises(ValueError):
        core.learn(1.0, learning_rate=0.01)


def test_core_weights_stay_bounded_and_signs_never_flip(graph):
    rng = np.random.default_rng(2)
    signs = rng.choice([-1, 0, 1], graph.n)
    core = EpropCore(graph, signs, plastic_edges=np.ones(graph.e, dtype=bool))
    for _ in range(30):
        core.tick(rng.normal(1.0, 0.5, graph.n), plastic=True)
        core.learn(rng.normal(0, 50, graph.n), learning_rate=1.0)
    assert (core.magnitude > 0).all()
    assert (core.magnitude <= core.base * core.weight_bound + 1e-9).all()
    assert (core.magnitude >= core.base / core.weight_bound - 1e-9).all()
    assert np.array_equal(core.signs, signs)


def test_exploration_floor_bounds_every_action_probability():
    motor = ActorCritic(4, np.random.default_rng(0), exploration_floor=0.09)
    motor.actor[0] = 50.0  # force a saturated preference for action 0
    _, _, mixed = motor.policy(np.ones(4))
    assert mixed.min() >= 0.09 / 3 - 1e-12
    assert mixed.sum() == pytest.approx(1.0)


def test_entropy_term_pushes_a_collapsed_policy_back_toward_uniform():
    """The v1 collapse is the failure v1.1 exists to prevent."""
    motor = ActorCritic(
        4, np.random.default_rng(0), entropy_coefficient=1.0, actor_lr=0.5, exploration_floor=0.0
    )
    motor.actor[0] = 5.0
    features = np.ones(4)
    _, softmax, _ = motor.policy(features)
    start = float(-(softmax * np.log(softmax + 1e-12)).sum())
    for _ in range(60):
        motor.choose(features, np.random.default_rng(1), training=True)
        motor.observe(0.0, 0.0, done=True)  # zero reward: only entropy acts
    _, softmax, _ = motor.policy(features)
    assert float(-(softmax * np.log(softmax + 1e-12)).sum()) > start


def test_critic_learns_a_constant_return():
    motor = ActorCritic(3, np.random.default_rng(0), critic_lr=0.2, discount=0.0)
    features = np.array([0.3, -0.2, 0.5])
    for _ in range(300):
        motor.reset()
        motor.choose(features, np.random.default_rng(2), training=True)
        motor.observe(1.0, 0.0, done=True)
    assert motor.value_of(features) == pytest.approx(1.0, abs=0.1)


def test_standardizer_removes_common_mode_and_normalises_scale():
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 2.0, (200, 50))
    standardizer = Standardizer.fit(samples)
    out = np.array([standardizer(row) for row in samples])
    assert abs(out.mean()) < 0.02
    assert np.linalg.norm(out, axis=1).mean() == pytest.approx(1.0, abs=0.25)


def test_v11_seed_blocks_are_disjoint_from_v1(tmp_path):
    """No v1.1 run may tune or test on a seed v1 already used."""
    del tmp_path
    v1_train = {1000 + 10_000 * s + e for s in range(3) for e in range(200)}
    v1_eval = {3_000_000 + 100 * s + i for s in range(3) for i in range(20)}
    v1_dev = set(range(1_900_000, 1_900_500)) | set(range(2_000_000, 2_000_500))
    v1_all = v1_train | v1_eval | v1_dev
    v11_dev = {dev_training_seed(s, e) for s in range(4) for e in range(2000)}
    v11_dev |= {s for d in range(4) for s in dev_evaluation_seeds(d)}
    v11_dev |= set(FEATURE_FIT_SEEDS)
    v11_confirm = {confirm_training_seed(s, e) for s in range(5) for e in range(2000)}
    v11_confirm |= {s for c in range(5) for s in confirm_evaluation_seeds(c)}
    assert not v11_dev & v1_all
    assert not v11_confirm & v1_all
    assert not v11_dev & v11_confirm
