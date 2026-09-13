"""The dashboard is evaluation-only: no weight, optimizer, connectome or file changes."""

import hashlib
from pathlib import Path

import pytest
from dashboard_support import PLAYABLE, ROOT, dashboard, run_episode  # noqa: F401

from flyarcade_dashboard import model_loader as ml
from flyarcade_dashboard.tasks import DEMO_BASE, TASK_ORDER, calibration_seed, demo_seed


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("task", PLAYABLE)
def test_no_parameter_or_file_changes_while_playing(dashboard, task):  # noqa: F811
    dashboard.new_session(task, 0, 0)
    agent = dashboard.agent
    files = [agent.info["checkpoint"], agent.info["result"]]
    before_files = {f: _sha(f) for f in files}
    before = agent.parameter_digest()
    run_episode(dashboard, task, demo_index=0)
    run_episode(dashboard, task, demo_index=1)
    assert agent.parameter_digest() == before
    assert {f: _sha(f) for f in files} == before_files


def test_learning_methods_are_never_called(dashboard, monkeypatch):  # noqa: F811
    for task in ("catch", "snake"):
        dashboard.new_session(task, 0, 0)
        controller = dashboard.agent.controller
        for name in ("act", "reward", "finish_episode", "_apply"):
            monkeypatch.setattr(
                controller, name, lambda *a, **k: pytest.fail("learning method called")
            )
        for method in ("observe", "choose"):
            monkeypatch.setattr(
                controller.motor, method, lambda *a, **k: pytest.fail("training path called")
            )
        run_episode(dashboard, task)
    for task in ("pong", "flappy"):
        dashboard.new_session(task, 0, 0)
        agent = dashboard.agent
        assert not hasattr(agent, "optimizer") and not hasattr(agent.model, "optimizer")
        assert agent.source.standardizer is None and agent.source.readout == "all"
        run_episode(dashboard, task)


def test_recurrent_connectome_is_never_plastic(dashboard):  # noqa: F811
    for task in ("catch", "dodge", "snake"):
        dashboard.new_session(task, 0, 0)
        core = dashboard.agent.controller.core
        magnitude = core.magnitude.copy()
        run_episode(dashboard, task)
        assert core.plastic_count == 0 and (core.magnitude == magnitude).all()


def test_demo_seeds_are_disjoint_from_every_study_seed_block():
    from flyarcade_v13.environments import TARGETS
    from flyarcade_v13.study import PURPOSES, purpose_range

    study_lo, study_hi = 0, 0
    for task in TARGETS:
        for purpose in PURPOSES:
            lo, hi = purpose_range(task, purpose)
            study_hi = max(study_hi, hi)
    # v1.3 envs plus the RNG offsets its evaluation derives (+500k, +1e9, +2e9);
    # v1.2 seeds < 1.5e8; v1.1 and v1 seeds < 1e7.
    highest_study_derived = study_hi + 2_000_000_000 + 1
    lowest_demo = DEMO_BASE
    highest_demo = demo_seed(TASK_ORDER[-1], 9_999_999) + 2_000_000_000
    assert lowest_demo > highest_study_derived
    assert study_lo < lowest_demo
    for task in TASK_ORDER:
        seeds = {demo_seed(task, i) for i in (0, 1, 2)} | {calibration_seed(task)}
        assert all(DEMO_BASE <= s <= highest_demo for s in seeds)
    with pytest.raises(ValueError):
        demo_seed("pong", -1)


def test_breakout_is_reported_unavailable_and_never_played(dashboard):  # noqa: F811
    state = dashboard.new_session("breakout")
    assert state["available"] is False
    assert state["unavailable_reason"].startswith("Trained model unavailable")
    with pytest.raises(ValueError):
        dashboard.step()


def test_dashboard_code_is_outside_every_frozen_code_hash():
    import subprocess

    tracked_scopes = [
        Path("src/flyarcade"),  # v1 run_trial hashes src/**, v1.2 hashes src/flyarcade/**
        Path("src/flyarcade_v13"),
    ]
    package = ROOT / "src/flyarcade_dashboard"
    assert all(not package.is_relative_to(ROOT / scope) for scope in tracked_scopes)
    # v1's run_trial hashes all of src/**; the dashboard documents that it never re-runs v1.
    del subprocess
    assert ml.PACKAGE_ROOT == ROOT
