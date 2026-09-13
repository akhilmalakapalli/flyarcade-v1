"""Shared fixtures for the dashboard tests (skip cleanly when frozen models are absent)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PLAYABLE = ("catch", "dodge", "snake", "pong", "flappy")


@pytest.fixture(scope="module")
def dashboard():
    from flyarcade_dashboard import model_loader as ml
    from flyarcade_dashboard.session import Dashboard

    if ml.find("data/malecns-v1.0/graph.npz") is None:
        pytest.skip("MaleCNS graph not available in any worktree")
    board = Dashboard()
    missing = [t for t in PLAYABLE if not ml.available_seeds(t, board.roots)]
    if missing:
        pytest.skip(f"frozen trained models unavailable for {missing}")
    return board


def run_episode(board, task, demo_index=0, limit=400):
    board.new_session(task, 0, demo_index)
    frames = []
    while len(frames) < limit:
        result = board.step()
        frames.append(result["frame"])
        if result["done"]:
            break
    return frames
