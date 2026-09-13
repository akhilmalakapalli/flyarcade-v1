import numpy as np
import pytest

from flyarcade.v12.environments import TASKS, Breakout, Flappy, Pong, Snake, reflect
from flyarcade.v12.study import PURPOSES, seed_for


@pytest.mark.parametrize("task", TASKS)
def test_reset_and_trajectory_determinism_and_ranges(task):
    a, b = TASKS[task](123), TASKS[task](123)
    rng = np.random.default_rng(7)
    while not a.done:
        action = int(rng.integers(a.actions))
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
        assert len(x[0]) == a.width and np.isfinite(x[0]).all()
        assert ((x[0] >= 0) & (x[0] <= 1)).all()
        assert 0 <= x[3]["success"] <= 1
    assert a.time <= a.horizon
    np.testing.assert_array_equal(a.reset(123), TASKS[task](123).observe())


@pytest.mark.parametrize("task", TASKS)
def test_invalid_actions(task):
    game = TASKS[task]()
    for a in (-1, game.actions, True, 1.0):
        with pytest.raises(ValueError):
            game.step(a)
    with pytest.raises(ValueError):
        game.reset(1.0)
    game.done = True
    with pytest.raises(ValueError):
        game.step(0)


def test_flappy_pipe_and_boundary_collisions_and_score():
    g = Flappy()
    g.x = 0.01
    g.y = g.gap
    g.vy = 0
    g.step(0)
    assert g.passed == 1 and g.metrics()["success"] == pytest.approx(1 / 6)
    g.x = 0.01
    g.y = 0.1
    g.gap = 0.8
    g.vy = 0
    g.step(0)
    assert g.done and g.death == "pipe"
    g = Flappy()
    g.y = 0.97
    g.step(1)
    assert g.done and g.death == "boundary"


def test_pong_interception_miss_and_wall_bounce():
    g = Pong()
    g.x = 0.01
    g.y = g.paddle
    g.vy = 0
    g.step(1)
    assert g.hits == g.attempts == 1
    g.x = 0.01
    g.y = 0.99
    g.vy = 0
    g.paddle = 0.1
    g.step(1)
    assert g.hits == 1 and g.attempts == 2
    g.x = 0.5
    g.y = 0.99
    g.vy = 0.03
    g.step(1)
    assert g.vy < 0 and 0 <= g.y <= 1
    assert reflect(1.2) == pytest.approx(0.8)


def test_breakout_brick_removed_once_and_three_misses_end():
    g = Breakout()
    g.x = 0.1
    g.y = 0.99
    g.vx = 0
    g.vy = 0.055
    g.step(1)
    assert g.metrics()["bricks_destroyed"] == 1
    g.y = 0.99
    g.vy = 0.055
    g.step(1)
    assert g.metrics()["bricks_destroyed"] == 1
    for _ in range(3):
        g.x = 0.01
        g.y = 0.01
        g.vy = -0.055
        g.vx = 0
        g.paddle = 0.9
        g.step(1)
    assert g.done and g.misses == 3


def test_snake_food_growth_relative_actions_and_collisions():
    g = Snake()
    g.food = (4, 3)
    g.step(1)
    assert g.eaten == 1 and len(g.body) == 4
    assert g.heading == 1
    g.step(0)
    assert g.heading == 0  # relative left from east is north
    g.body = [(0, 0), (1, 0), (2, 0)]
    g.heading = 3
    g.step(1)
    assert g.death == "wall"
    g = Snake()
    g.body = [(2, 2), (2, 3), (3, 3), (3, 2), (4, 2)]
    g.heading = 0
    g.food = (0, 0)
    g.step(2)
    assert g.death == "self"


def test_snake_moving_into_vacating_tail_is_legal():
    g = Snake()
    g.body = [(2, 2), (2, 3), (3, 3), (3, 2)]
    g.heading = 0
    g.food = (0, 0)
    assert not g.danger(2)
    g.step(2)
    assert not g.done


@pytest.mark.parametrize("task", TASKS)
def test_heuristic_competence_against_random(task):
    scores = {}
    for policy in ("random", "heuristic"):
        values = []
        for seed in range(20):
            g = TASKS[task](seed)
            rng = np.random.default_rng(seed + 10)
            while not g.done:
                g.step(g.heuristic() if policy == "heuristic" else int(rng.integers(g.actions)))
            values.append(g.metrics()["success"])
        scores[policy] = np.mean(values)
    assert scores["heuristic"] > scores["random"]


def test_all_new_seed_blocks_disjoint_from_history_and_each_other():
    intervals = []
    for task in TASKS:
        for purpose in PURPOSES:
            low = seed_for(task, purpose)
            high = seed_for(task, purpose, seed=9, index=9999)
            assert low > 10_000_000  # Covers all historical v1/v11/Snake namespaces.
            intervals.append((low, high))
    ordered = sorted(intervals)
    assert all(a[1] < b[0] for a, b in zip(ordered, ordered[1:]))
