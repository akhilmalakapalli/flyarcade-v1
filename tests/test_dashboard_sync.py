"""Frames are exact: one decision per STEP, activity from that step, frozen-eval parity."""

import base64

import numpy as np
import pytest
from dashboard_support import PLAYABLE, ROOT, dashboard, run_episode  # noqa: F401

from flyarcade_dashboard.tasks import demo_seed


def _counts(frame):
    return np.frombuffer(base64.b64decode(frame["counts_b64"]), dtype=np.uint8)


@pytest.mark.parametrize("task", PLAYABLE)
def test_matches_the_frozen_evaluation_code_on_a_demo_seed(dashboard, task):  # noqa: F811
    frames = run_episode(dashboard, task, demo_index=3)
    agent, seed = dashboard.agent, demo_seed(task, 3)
    actions = [f["action"] for f in frames]
    counts = np.bincount(actions, minlength=agent.action_count).tolist()
    if task in ("catch", "dodge"):
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        import v11_run_trial

        reference = v11_run_trial.greedy_rows(agent.controller, task, [seed])[0]
        assert reference["action_counts"] == counts
        assert reference["success"] == frames[-1]["metrics"]["success"]
    elif task == "snake":
        from flyarcade.v11.snake_experiment import evaluate

        reference = evaluate(agent.controller, [seed])[0]
        assert reference["action_counts"] == counts
        assert reference["food"] == frames[-1]["metrics"]["food"]
        assert reference["return"] == pytest.approx(frames[-1]["episode_return"], abs=1e-12)
    else:
        from flyarcade_v13.controller import FlyFeatures
        from flyarcade_v13.environments import TARGETS
        from flyarcade_v13.study import evaluate_policy

        source = FlyFeatures(
            dashboard.graph, TARGETS[task].width, 4, ticks=agent.ticks, standardizer=agent.std
        )
        reference = evaluate_policy(agent.model, agent.model.params, source, task, [seed])[0]
        assert reference["action_counts"] == counts
        assert reference["return"] == pytest.approx(frames[-1]["episode_return"], abs=1e-12)
        assert reference["success"] == frames[-1]["metrics"]["success"]


@pytest.mark.parametrize("task", PLAYABLE)
def test_step_advances_exactly_one_decision(dashboard, task):  # noqa: F811
    dashboard.new_session(task, 0, 0)
    for expected in range(5):
        observation = np.asarray(dashboard.env.observe(), dtype=float).tolist()
        time_before = dashboard.env.time
        frame = dashboard.step()["frame"]
        assert frame["step"] == expected and dashboard.step_index == expected + 1
        assert dashboard.env.time == time_before + 1
        assert frame["observation"] == observation
        assert frame["render_after"] == dashboard.state()["render"]


@pytest.mark.parametrize("task", PLAYABLE)
def test_neural_activity_belongs_to_the_displayed_step(dashboard, task):  # noqa: F811
    frames = run_episode(dashboard, task, demo_index=2, limit=12)
    agent = dashboard.agent
    env = agent.reset(demo_seed(task, 2))
    for frame in frames:
        assert np.asarray(env.observe(), dtype=float).tolist() == frame["observation"]
        decision = agent.decide(env.observe())
        np.testing.assert_array_equal(decision["counts"], _counts(frame))
        assert decision["action"] == frame["action"] and decision["probs"] == frame["probs"]
        assert decision["counts"].max() <= frame["ticks"]
        env.step(decision["action"])


@pytest.mark.parametrize("task", PLAYABLE)
def test_same_task_and_demo_seed_give_identical_episodes(dashboard, task):  # noqa: F811
    first = run_episode(dashboard, task, demo_index=1)
    second = run_episode(dashboard, task, demo_index=1)
    assert [f["action"] for f in first] == [f["action"] for f in second]
    assert all(a["counts_b64"] == b["counts_b64"] for a, b in zip(first, second, strict=True))
    other = run_episode(dashboard, task, demo_index=4)
    assert [f["render_before"] for f in other][:1] != [] and other[0]["demo_seed"] != first[0][
        "demo_seed"
    ]


def test_switching_tasks_resets_cleanly(dashboard):  # noqa: F811
    dashboard.new_session("pong", 0, 0)
    for _ in range(7):
        dashboard.step()
    state = dashboard.new_session("flappy", 0, 0)
    assert state["step"] == 0 and dashboard.frames == [] and dashboard.episode_return == 0
    assert type(dashboard.env).__name__ == "Flappy"
    assert (dashboard.agent.hidden == 0).all()
    frame = dashboard.step()["frame"]
    assert frame["task"] == "flappy" and frame["step"] == 0
    dashboard.new_session("catch", 0, 0)
    assert dashboard.score_total == 0 and dashboard.step()["frame"]["step"] == 0
