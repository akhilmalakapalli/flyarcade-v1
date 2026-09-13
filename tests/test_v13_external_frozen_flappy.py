"""The externally frozen Flappy result is integrated read-only and can never be re-run."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v13_external  # noqa: E402

from flyarcade_v13.environments import TARGETS  # noqa: E402

CONFIRM = ROOT / "experiments/v13_flappy/specs/confirm"


@pytest.fixture(autouse=True)
def repository_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_frozen_flappy_files_are_byte_identical_to_the_registered_hashes():
    registry = v13_external.load_registry()  # raises if any frozen file changed
    flappy = registry["tasks"]["flappy"]
    assert flappy["frozen_plan_commit"] == "45bd666" and flappy["results_commit"] == "0443717"
    assert flappy["spent_confirmatory_seeds"] == [0, 1, 2, 3, 4]
    assert "conf_perturb" in flappy["spent_confirmatory_purposes"]
    plan = json.loads((ROOT / flappy["frozen_plan"]).read_text())
    assert set(plan["specs"]) <= set(flappy["file_sha256"])


def test_registered_flappy_headline_matches_the_frozen_summary_and_reported_numbers():
    flappy = v13_external.load_registry()["tasks"]["flappy"]
    summary = json.loads((ROOT / flappy["summary"]).read_text())
    verification = json.loads((ROOT / flappy["verification"]).read_text())
    h = flappy["headline"]
    assert h["biological_after"] == summary["biological_after"]
    assert round(h["biological_after"], 3) == 0.605
    assert [round(v, 3) for v in h["per_seed_after"]["biological"]] == [
        0.630,
        0.480,
        0.563,
        0.717,
        0.633,
    ]
    assert round(h["biological_before_untrained"], 3) == 0.0 and h["random"] == 0.0
    assert round(h["rewired_after"], 3) == 0.252
    assert h["heuristic_reference"] == 1.0 and h["sensory_after"] == 1.0
    assert h["criterion_met"] is True
    assert summary["criterion_without_replay"]["passed"] is True and verification["passed"] is True


def test_frozen_flappy_model_is_the_reported_model():
    flappy = v13_external.load_registry()["tasks"]["flappy"]
    config = flappy["frozen_config"]
    assert config == {
        "task": "flappy",
        "source": "fly",
        "lr": 0.0003,
        "transitions": 150000,
        "arch": "gru",
        "ticks": 16,
        "shaping": 0.0,
        "ent": 0.003,
    }
    assert "encoding" not in config and "potential" not in config  # v1.2 encoding, no shaping
    plan = json.loads((ROOT / flappy["frozen_plan"]).read_text())
    assert plan["recurrent_core"].startswith("MaleCNS recurrent weights frozen")


def test_multitask_freeze_never_writes_flappy_confirmatory_specs():
    assert "flappy" not in v13_external.confirmatory_tasks(TARGETS)


def test_every_flappy_confirmatory_spec_is_refused_and_development_specs_are_not():
    confirm = sorted(CONFIRM.glob("*.json"))
    development = sorted((ROOT / "experiments/v13_flappy/specs/validate").glob("*.json"))
    assert len(confirm) == 15 and len(development) == 5
    assert all(v13_external.spent(json.loads(p.read_text())) for p in confirm)
    assert not any(v13_external.spent(json.loads(p.read_text())) for p in development)
    perturb_only = {
        "config": {"task": "flappy", "train_purpose": "validate_train"},
        "evaluation": {"purpose": "validate_eval", "perturb_purpose": "conf_perturb"},
    }
    assert v13_external.spent(perturb_only)


@pytest.mark.parametrize("suite_name", ["v13_suite", "v13_flappy_suite"])
def test_both_suites_refuse_flappy_confirmatory_reruns(suite_name, tmp_path, monkeypatch):
    suite = __import__(suite_name)
    launched = []
    monkeypatch.setattr(suite, "run", lambda *a, **k: launched.append(a))
    spec = CONFIRM / "flappy-confirm-biological-s0.json"
    monkeypatch.setattr(sys, "argv", [f"{suite_name}.py", str(spec), "--log-dir", str(tmp_path)])
    assert suite.main() == 2
    assert launched == []


def test_report_entry_reads_flappy_only_from_frozen_files():
    import v13_report

    entry = v13_external.external_summary_entry("flappy", v13_report.describe)
    assert entry["external_freeze"] and entry["criteria"]["passed"]
    assert entry["criteria"]["not_rescored_under_multitask_gate"]
    assert entry["criteria"]["components"]["10_checkpoint_replay_and_provenance_verification"]
    p = entry["primary"]
    assert [round(v, 3) for v in p["biological"]["seed_values"]] == [
        0.630,
        0.480,
        0.563,
        0.717,
        0.633,
    ]
    assert round(p["frozen"]["mean"], 3) == 0.0 and round(p["rewired"]["mean"], 3) == 0.252
    assert (
        p["sensory"]["mean"] == 1.0 and p["reference"]["mean"] == 1.0 and p["random"]["mean"] == 0.0
    )
