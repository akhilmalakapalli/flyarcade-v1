import numpy as np
import pytest

from flyarcade.connectome import Graph
from flyarcade.controller import Controller
from flyarcade.games import LaneGame, heuristic
from flyarcade.neural.lif import LIF


def tiny_graph():
    return Graph(
        np.array([1, 2, 3]),
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([2, 3]),
        {
            "kind": "synthetic",
            "neurons": [
                {"bodyId": 1, "superclass": "visual_projection", "consensusNt": "acetylcholine"},
                {"bodyId": 2, "superclass": "cb_intrinsic", "consensusNt": "gaba"},
                {"bodyId": 3, "superclass": "descending_neuron", "consensusNt": "unclear"},
            ],
        },
    )


def test_lif_delay_reset_refractory_and_inhibition():
    lif = LIF(tiny_graph(), [1, -1, 0])
    np.testing.assert_array_equal(lif.tick([1, 0, 0]), [1, 0, 0])
    np.testing.assert_array_equal(lif.tick([1, 0, 0]), [0, 1, 0])
    assert lif.voltage[0] == 0
    lif.tick([0, 0, 0])
    assert lif.voltage[2] < 0
    lif.reset()
    assert not lif.spikes.any() and not lif.eligibility.any()


def test_causal_eligibility_reward_and_bounds():
    lif = LIF(tiny_graph(), [0, 0, 0])
    lif.tick([1, 0, 0])
    lif.tick([0, 1, 0])
    assert lif.eligibility[0] > 0
    before = lif.magnitude.copy()
    lif.reward(1)
    assert lif.magnitude[0] > before[0]
    for _ in range(1000):
        lif.reward(-2)
    assert np.all(lif.magnitude >= 0.25 * lif.base)
    with pytest.raises(ValueError):
        lif.reward(float("nan"))


@pytest.mark.parametrize("task", ["catch", "dodge"])
def test_game_determinism_and_heuristic(task):
    a, b = LaneGame(task, seed=91), LaneGame(task, seed=91)
    score = 0
    while not a.done:
        action = heuristic(task, a.observe())
        left, right = a.step(action), b.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
        score += left[3]["score"]
    assert score == 6
    with pytest.raises(ValueError):
        a.step(1)


def test_invalid_actions_and_observation_bounds():
    game = LaneGame()
    for action in (-1, 3, True, 1.0):
        with pytest.raises(ValueError):
            game.step(action)
    for _ in range(24):
        obs, _, _, _ = game.step(0)
        assert ((obs >= 0) & (obs <= 1)).all()


def test_evaluation_frozen_and_rng_reproducibility():
    # More descending neurons avoid empty artificial readout groups in this test.
    graph = tiny_graph()
    a, b = Controller(graph, seed=7), Controller(graph, seed=7)
    before = a.core.magnitude.copy(), a.motor.weights.copy(), a.baseline
    for _ in range(10):
        assert a.act([0.2, 0.8, 0]) == b.act([0.2, 0.8, 0])
    np.testing.assert_array_equal(a.core.magnitude, before[0])
    np.testing.assert_array_equal(a.motor.weights, before[1])
    assert a.baseline == before[2]


def test_episode_checkpoint_resume_matches_uninterrupted(tmp_path):
    from flyarcade.experiment import checkpoint, episode, restore

    graph = tiny_graph()
    a = Controller(graph, seed=31)
    episode(a, "catch", 10, training=True)
    path = tmp_path / "checkpoint.npz"
    checkpoint(a, path, {"graph_sha256": "fixture", "completed_episodes": 1})
    b = Controller(graph, seed=900)
    assert restore(b, path, "fixture")["completed_episodes"] == 1
    assert episode(a, "catch", 11, training=True) == episode(b, "catch", 11, training=True)
    np.testing.assert_array_equal(a.core.magnitude, b.core.magnitude)
    np.testing.assert_array_equal(a.motor.weights, b.motor.weights)
    with pytest.raises(ValueError, match="graph mismatch"):
        restore(b, path, "wrong")
