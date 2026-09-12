"""Regression tests for the Snake environment and its seed discipline."""

import numpy as np
import pytest

from flyarcade.v11.snake import (
    CHANNELS,
    GRID,
    MOVES,
    OPPOSITE,
    SnakeGame,
    features,
    heuristic,
)
from flyarcade.v11.snake_experiment import (
    FEATURE_FIT_SEEDS,
    confirm_evaluation_seeds,
    confirm_training_seed,
    dev_evaluation_seeds,
    dev_training_seed,
    episodes_to_threshold,
    state_dependence,
)


def test_wall_collision_ends_the_episode():
    game = SnakeGame(0)
    for _ in range(GRID + 2):
        if game.done:
            break
        game.step(0)  # drive straight up into the top wall
    assert game.done and game.death == "wall"


def test_self_collision_ends_the_episode():
    game = SnakeGame(0)
    # Grow the snake, then turn back into its own body.
    game.body = [(4, 4), (4, 3), (3, 3), (3, 4), (2, 4)]
    game.heading = 3
    _, _, done, _ = game.step(0)
    assert done and game.death == "self"


def test_immediate_reversal_is_refused_not_fatal():
    game = SnakeGame(0)
    assert game.heading == 3
    head = game.head
    game.step(2)  # a 180-degree reversal
    assert game.heading == 3
    assert game.head == (head[0], head[1] + 1)


def test_eating_food_grows_the_snake_and_pays_reward():
    game = SnakeGame(0)
    row, column = game.head
    game.food = (row, column + 1)
    before = len(game.body)
    _, reward, _, info = game.step(3)
    assert info["eaten"] == 1
    assert len(game.body) == before + 1
    assert reward > 0.9
    assert game.food != (row, column + 1)


def test_food_never_lands_on_the_snake():
    for seed in range(40):
        game = SnakeGame(seed)
        for _ in range(30):
            if game.done:
                break
            game.step(heuristic(game))
            assert game.food is None or game.food not in game.body


def test_episode_terminates_under_every_policy():
    rng = np.random.default_rng(0)
    for seed in range(30):
        for policy in ("random", "heuristic", "straight"):
            game = SnakeGame(seed)
            steps = 0
            while not game.done and steps < 1000:
                if policy == "random":
                    action = int(rng.integers(0, 4))
                elif policy == "heuristic":
                    action = heuristic(game)
                else:
                    action = 3
                game.step(action)
                steps += 1
            assert game.done and game.death is not None


def test_observation_is_normalised_and_correctly_shaped():
    for seed in range(20):
        game = SnakeGame(seed)
        while not game.done:
            observation = game.observe()
            assert observation.shape == (CHANNELS,)
            assert np.isfinite(observation).all()
            assert (observation >= 0).all() and (observation <= 1).all()
            assert np.array_equal(features(observation), observation)
            game.step(heuristic(game))


def test_danger_channels_predict_death():
    """Channels 9-12 must actually mean 'moving here ends the episode'."""
    checked = 0
    for seed in range(60):
        game = SnakeGame(seed)
        while not game.done and checked < 200:
            observation = game.observe()
            for action, danger in enumerate(observation[9:13]):
                if danger and action != OPPOSITE[game.heading]:
                    probe = SnakeGame(seed)
                    probe.body = list(game.body)
                    probe.heading = game.heading
                    probe.food = game.food
                    probe.step(action)
                    assert probe.done and probe.death in ("wall", "self")
                    checked += 1
            game.step(heuristic(game))
    assert checked > 20


def test_heuristic_beats_a_random_policy_by_a_wide_margin():
    """The task must be solvable, or a negative learning result means nothing."""
    rng = np.random.default_rng(0)
    scores = {}
    for policy in ("heuristic", "random"):
        eaten = []
        for seed in range(60):
            game = SnakeGame(seed)
            while not game.done:
                game.step(heuristic(game) if policy == "heuristic" else int(rng.integers(0, 4)))
            eaten.append(game.eaten)
        scores[policy] = float(np.mean(eaten))
    assert scores["heuristic"] > 8.0
    assert scores["random"] < 1.0


def test_moves_and_opposites_are_consistent():
    for action, opposite in OPPOSITE.items():
        assert tuple(-x for x in MOVES[action]) == MOVES[opposite]


def test_state_dependence_and_threshold_helpers():
    constant = [{"action_counts": [10, 0, 0, 0]}]
    assert state_dependence(constant) == pytest.approx(0.0)
    mixed = [{"action_counts": [5, 5, 5, 5]}]
    assert state_dependence(mixed) == pytest.approx(0.75)
    curve = [{"food": 0.0}] * 30 + [{"food": 5.0}] * 30
    assert episodes_to_threshold(curve, 2.0, window=25) is not None
    assert episodes_to_threshold(curve, 99.0, window=25) is None


def test_snake_seed_blocks_are_disjoint_from_v1_and_v11():
    v1 = (
        {1000 + 10_000 * s + e for s in range(3) for e in range(200)}
        | {3_000_000 + 100 * s + i for s in range(3) for i in range(20)}
        | set(range(1_900_000, 1_900_500))
        | set(range(2_000_000, 2_000_500))
    )
    v11 = (
        {4_000_000 + 10_000 * s + e for s in range(6) for e in range(3000)}
        | {4_500_000 + 1_000 * d + i for d in range(6) for i in range(20)}
        | set(range(4_200_000, 4_200_120))
        | set(range(4_300_000, 4_300_030))
        | {5_000_000 + 10_000 * s + e for s in range(3) for e in range(3000)}
        | {6_000_000 + 100 * s + i for s in range(3) for i in range(40)}
    )
    dev = (
        {dev_training_seed(s, e) for s in range(4) for e in range(8000)}
        | {s for d in range(4) for s in dev_evaluation_seeds(d)}
        | set(FEATURE_FIT_SEEDS)
    )
    confirm = {confirm_training_seed(s, e) for s in range(4) for e in range(8000)} | {
        s for c in range(4) for s in confirm_evaluation_seeds(c)
    }
    assert not dev & v1 and not dev & v11
    assert not confirm & v1 and not confirm & v11
    assert not dev & confirm
