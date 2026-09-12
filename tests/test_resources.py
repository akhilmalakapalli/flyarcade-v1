import pytest

from flyarcade.config import Limits
from flyarcade.resources import ResourceGuard, ResourceLimit


def test_limits_reject_unbounded_configuration():
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Limits(max_seconds=value)
    with pytest.raises(ValueError):
        Limits().check_graph(5000, 1)
    with pytest.raises(ValueError):
        Limits(max_neurons=1.5)
    with pytest.raises(ValueError):
        Limits().check_graph(1.5, 0)


def test_guards(monkeypatch):
    guard = ResourceGuard()
    assert guard.check()["rss_mb"] > 0
    for key, value in (("rss_mb", 3000), ("free_disk_mb", 1), ("elapsed_seconds", 121)):
        stats = {"rss_mb": 10, "free_disk_mb": 2000, "elapsed_seconds": 0}
        stats[key] = value
        monkeypatch.setattr(guard, "snapshot", lambda: stats)
        with pytest.raises(ResourceLimit):
            guard.check()
