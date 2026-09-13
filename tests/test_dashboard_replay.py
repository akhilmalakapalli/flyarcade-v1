"""Replays reproduce the live frames exactly; the server stays small and safe."""

import json
import threading
import urllib.error
import urllib.request

import pytest
from dashboard_support import ROOT, dashboard, run_episode  # noqa: F401

from flyarcade_dashboard import replay


@pytest.mark.parametrize("task", ("pong", "flappy", "snake", "catch"))
def test_saved_episode_replays_exactly(dashboard, task, tmp_path):  # noqa: F811
    frames = run_episode(dashboard, task, demo_index=5)
    name = replay.save(dashboard, directory=tmp_path)
    assert [r["name"] for r in replay.list_replays(tmp_path)] == [name]
    loaded = replay.load(name, directory=tmp_path)
    assert loaded["label"] == "Demo playback — not a new experiment"
    assert loaded["steps"] == len(frames) and loaded["complete_episode"]
    assert len(loaded["frames"]) == len(frames)
    for live, again in zip(frames, loaded["frames"], strict=True):
        assert again == json.loads(json.dumps(live))  # every field, including activity


def test_replay_names_cannot_escape_the_replay_directory(tmp_path):
    for bad in ("../x", "a/b", "..", "x;rm"):
        with pytest.raises(ValueError):
            replay.load(bad, directory=tmp_path)


def test_default_replay_directory_is_the_ignored_cache():
    assert replay.REPLAY_DIR == ROOT / "cache" / "dashboard" / "replays"
    assert "artifacts" not in replay.REPLAY_DIR.parts and "runs" not in replay.REPLAY_DIR.parts


def test_server_api_round_trip(dashboard):  # noqa: F811
    from flyarcade_dashboard.server import serve

    server, _ = serve("127.0.0.1", 0, dashboard=dashboard)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    try:
        tasks = {t["task"]: t for t in call("/api/tasks")}
        assert set(tasks) == {"catch", "dodge", "pong", "snake", "flappy", "breakout"}
        assert not tasks["breakout"]["available"]
        state = call("/api/session/new", {"task": "pong", "model_seed": 0})
        assert state["demo_label"] == "Demo playback — not a new experiment"
        steps = [call("/api/session/step", {})["frame"]["step"] for _ in range(3)]
        assert steps == [0, 1, 2]
        network = call("/api/network")
        assert network["n"] == 2040 and network["label"] == "2,040-neuron MaleCNS-derived network"
        with urllib.request.urlopen(base + "/", timeout=10) as page:
            assert b"FlyArcade" in page.read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/static/../server.py", timeout=10)
    finally:
        server.shutdown()
        server.server_close()


def test_long_playback_does_not_grow_memory(dashboard):  # noqa: F811
    import gc

    import psutil

    process = psutil.Process()
    for task in ("pong", "flappy", "snake", "catch", "dodge"):  # warm every model once
        run_episode(dashboard, task)
    gc.collect()
    start = process.memory_info().rss
    for round_ in range(3):
        for task in ("pong", "flappy", "snake", "catch", "dodge"):
            run_episode(dashboard, task, demo_index=10 + round_)
    gc.collect()
    growth_mb = (process.memory_info().rss - start) / 2**20
    assert len(dashboard.frames) <= 400
    assert growth_mb < 40, f"RSS grew by {growth_mb:.1f} MiB over repeated playback"
