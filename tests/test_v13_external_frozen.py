"""The externally frozen Pong result is integrated read-only and can never be re-run."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v13_external  # noqa: E402

from flyarcade_v13.environments import TARGETS  # noqa: E402
from flyarcade_v13.study import code_hash  # noqa: E402


@pytest.fixture(autouse=True)
def repository_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_frozen_pong_files_are_byte_identical_to_the_registered_hashes():
    registry = v13_external.load_registry()  # raises if any frozen file changed
    pong = registry["tasks"]["pong"]
    assert pong["frozen_plan_commit"] == "56c604b" and pong["results_commit"] == "94422be"
    assert pong["spent_confirmatory_seeds"] == [0, 1, 2, 3, 4]


def test_registered_headline_matches_the_frozen_summary_and_reported_numbers():
    pong = v13_external.load_registry()["tasks"]["pong"]
    summary = json.loads((ROOT / pong["summary"]).read_text())
    headline = pong["headline"]
    assert headline["biological_after"] == summary["biological_after"]
    assert round(headline["biological_after"], 3) == 0.793
    assert round(headline["random"], 3) == 0.253
    assert round(headline["biological_before_untrained"], 3) == 0.306
    assert round(headline["rewired_after"], 3) == 0.870
    assert headline["heuristic_reference"] == 1.0 and headline["sensory_after"] == 1.0
    assert headline["criterion_met"] is True and summary["criterion"]["met"] is True


def test_multitask_freeze_never_writes_pong_confirmatory_specs():
    tasks = v13_external.confirmatory_tasks(TARGETS)
    assert "pong" not in tasks
    assert set(tasks) == set(TARGETS) - {"pong", "flappy"}  # Flappy: see test_..._flappy.py


def test_every_pong_confirmatory_spec_is_refused_and_development_specs_are_not():
    confirm = sorted((ROOT / "experiments/v13_pong/specs/confirm").glob("*.json"))
    validate = sorted((ROOT / "experiments/v13_pong/specs/validate").glob("*.json"))
    assert len(confirm) == 15 and validate
    assert all(v13_external.spent(json.loads(p.read_text())) for p in confirm)
    assert not any(v13_external.spent(json.loads(p.read_text())) for p in validate)
    other = {"config": {"task": "breakout", "train_purpose": "conf_train"}, "evaluation": {}}
    assert not v13_external.spent(other)
    hidden = {
        "config": {"task": "pong", "train_purpose": "validate_train"},
        "evaluation": {"purpose": "conf_eval"},
    }
    assert v13_external.spent(hidden), "a conf purpose anywhere in the spec counts"


def test_suite_refuses_spent_specs_before_launching_anything(tmp_path, monkeypatch):
    import v13_suite

    launched = []
    monkeypatch.setattr(v13_suite, "run", lambda *a, **k: launched.append(a))
    spec = ROOT / "experiments/v13_pong/specs/confirm/pong-confirm-biological-s0.json"
    monkeypatch.setattr(
        sys, "argv", ["v13_suite.py", str(spec), "--log-dir", str(tmp_path / "logs")]
    )
    assert v13_suite.main() == 2
    assert launched == []


def test_report_entry_reads_pong_only_from_frozen_files():
    import v13_report

    entry = v13_external.external_summary_entry("pong", v13_report.describe)
    assert entry["external_freeze"] and entry["criteria"]["not_rescored_under_multitask_gate"]
    assert round(entry["primary"]["biological"]["mean"], 3) == 0.793
    assert [round(v, 3) for v in entry["primary"]["biological"]["seed_values"]] == [
        0.778,
        0.772,
        0.794,
        0.803,
        0.819,
    ]
    assert round(entry["primary"]["frozen"]["mean"], 3) == 0.306
    assert entry["perturbations"] is None


def test_integration_did_not_change_the_hashed_multitask_code():
    """Running multitask checkpoints and any later freeze hash exactly these files."""
    frozen_pong = json.loads((ROOT / "artifacts/v13/pong/confirmatory_summary.json").read_text())
    assert frozen_pong  # the Pong trials were produced by this same hashed code
    assert code_hash() == "504a571e76244f199713ffb732eedd645d4504aba13a7fd4088f6d7e2206ae59"
