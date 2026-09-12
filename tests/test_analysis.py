"""Regression tests for the post-hoc analysis and replay tooling."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import diagnostics

    return diagnostics


def test_probe_recovers_a_planted_signal_and_rejects_shuffled_labels(diagnostics):
    rng = np.random.default_rng(0)
    classes = np.array([-1, 0, 1])
    labels = rng.choice(classes, size=400)
    features = rng.normal(0, 0.4, (400, 6))
    features[:, 0] += labels
    accuracy = diagnostics.ridge_probe(
        features[:200], labels[:200], features[200:], labels[200:], classes
    )
    shuffled = diagnostics.ridge_probe(
        features[:200],
        rng.permutation(labels[:200]),
        features[200:],
        labels[200:],
        classes,
    )
    # Bayes-optimal accuracy for this planted separation is near 0.86.
    assert accuracy > 0.78
    assert shuffled < 0.6


def test_probe_on_pure_noise_stays_near_chance(diagnostics):
    """A probe must not manufacture structure; this guards the negative result."""
    rng = np.random.default_rng(1)
    classes = np.array([-1, 0, 1])
    labels = rng.choice(classes, size=400)
    features = rng.normal(0, 1.0, (400, 6))
    accuracy = diagnostics.ridge_probe(
        features[:200], labels[:200], features[200:], labels[200:], classes
    )
    majority = np.bincount(labels[200:] + 1).max() / 200
    assert accuracy <= majority + 0.1


def test_constant_action_reference_matches_lane_geometry(diagnostics):
    """Always-left pins the player at lane 0: catch succeeds only on lane 0."""
    seeds = list(range(3_000_000, 3_000_060))
    catch = diagnostics.constant_action_reference("catch", seeds)
    dodge = diagnostics.constant_action_reference("dodge", seeds)
    for policy in ("always_left", "always_right"):
        assert catch[policy] + dodge[policy] == pytest.approx(1.0)
        assert catch[policy] == pytest.approx(0.2, abs=0.12)


def test_published_artifacts_agree_with_each_other():
    """The manuscript's numbers must trace back to consistent committed artifacts."""
    import json

    root = Path(__file__).resolve().parents[1] / "artifacts"
    if not (root / "results_summary.json").exists():
        pytest.skip("summary artifacts not generated in this environment")
    summary = json.loads((root / "results_summary.json").read_text())
    replay = json.loads((root / "replay_check.json").read_text())
    audit = json.loads((root / "scientific_audit.json").read_text())
    assert summary["trial_count"] == replay["trials_replayed"] == 36
    assert replay["all_match"] is True
    assert audit["status"] == "PASS"
    for task in ("catch", "dodge"):
        frozen = summary["primary"][f"{task}/frozen"]
        assert frozen["change"]["mean"] == pytest.approx(0.0)
        assert summary["robustness"][task]["after"]["mean"] == pytest.approx(
            summary["primary"][f"{task}/learning"]["after"]["mean"]
        )
