import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flyarcade.v12 import environments as v12env
from flyarcade_v13.environments import (
    LEVELS,
    TARGETS,
    Flappy,
    flappy_oracle,
    make_env,
    potential,
    reference_action,
    terminal_kind,
)
from flyarcade_v13.study import seed_for


def test_target_environments_are_the_frozen_v12_classes():
    for task, cls in TARGETS.items():
        assert cls is v12env.TASKS[task]
        assert type(make_env(task, 5)) is cls
    snapshot = json.loads(Path("artifacts/v13/historical_hashes.json").read_text())["files"]
    path = "src/flyarcade/v12/environments.py"
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == snapshot[path]


@pytest.mark.parametrize("task", list(TARGETS))
def test_deterministic_bounded_valid_trajectories(task):
    runs = []
    for _ in range(2):
        env = make_env(task, seed_for(task, "env_validation", index=3))
        rng = np.random.default_rng(1)
        trace = []
        while not env.done:
            obs = env.observe()
            assert obs.shape == (env.width,) and np.isfinite(obs).all()
            assert (obs >= 0).all() and (obs <= 1).all()
            action = int(rng.integers(env.actions))
            _, reward, done, metrics = env.step(action)
            assert np.isfinite(reward) and 0 <= metrics["success"] <= 1
            trace.append((obs.tobytes(), reward))
        with pytest.raises(ValueError):
            env.step(0)
        runs.append(trace)
    assert runs[0] == runs[1]


def test_invalid_actions_rejected():
    env = make_env("pong", 1)
    for bad in (-1, 3, 1.0, True):
        with pytest.raises(ValueError):
            env.step(bad)


def test_flappy_pipe_crossing_collision_and_boundaries():
    env = Flappy(0)
    env.x, env.y, env.vy, env.gap = 0.03, 0.5, 0.0, 0.5
    _, reward, done, m = env.step(0)
    assert m["obstacles_passed"] == 1 and not done and reward > 0.9 and env.x == 0.9
    env.x, env.y, env.vy, env.gap = 0.03, 0.5, 0.0, 0.2
    _, reward, done, m = env.step(0)
    assert done and m["death"] == "pipe" and reward == -1
    env = Flappy(0)
    env.y, env.vy = 0.95, 0.0
    _, _, done, m = env.step(1)
    assert done and m["death"] == "boundary"
    env = Flappy(0)
    env.y, env.vy = 0.03, -0.08
    _, _, done, m = env.step(0)
    assert done and m["death"] == "boundary"


def test_flappy_oracle_timing_uses_no_environment_randomness():
    env = Flappy(seed_for("flappy", "env_validation", index=0))
    state = env.rng.bit_generator.state
    flappy_oracle(env)
    assert env.rng.bit_generator.state == state


def test_flappy_oracle_gate_on_fresh_development_seeds():
    scores = []
    for i in range(100):
        env = Flappy(seed_for("flappy", "env_validation", index=i))
        while not env.done:
            env.step(flappy_oracle(env))
        scores.append(env.metrics()["success"])
    assert np.mean(scores) >= 0.80


def test_curricula_keep_interface_and_final_target():
    for task, levels in LEVELS.items():
        target = TARGETS[task]
        for level in levels:
            env = make_env(task, 7, level)
            assert env.width == target.width and env.actions == target.actions
            assert env.observe().shape == (target.width,)
            while not env.done:
                env.step(reference_action(env))
    small = make_env("snake", 3, "size4")
    assert small.size == 4 and all(0 <= c < 4 for cell in small.body for c in cell)
    assert make_env("breakout", 3, "bricks1").bricks.sum() == 1


def test_potentials_are_state_functions_and_terminal_kinds():
    for task in TARGETS:
        env = make_env(task, 11)
        assert potential(env) == potential(env) <= 0
    pong = make_env("pong", 2)
    while not pong.done:
        pong.step(1)
    assert terminal_kind(pong) == "truncated"
    snake = make_env("snake", 2)
    while not snake.done:
        snake.step(1)
    assert terminal_kind(snake) == "terminal"
