from pathlib import Path

import pytest


def test_seed_bootstrap_not_episode_pseudoreplication(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_results import describe

    assert describe([0.1, 0.2, 0.3]) == describe([0.1, 0.2, 0.3])
    assert describe([0.2, 0.2, 0.2])["descriptive_bootstrap_95"] == pytest.approx([0.2, 0.2])
    assert len(describe([0.1, 0.2, 0.3])["seed_values"]) == 3
    with pytest.raises(ValueError):
        describe([float("nan"), 1])
