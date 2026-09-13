"""Write final v1.4 coordination documents from the audited development artifacts."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("artifacts/v14")


def main():
    summary = json.loads((ROOT / "development_summary.json").read_text())
    audit = json.loads((ROOT / "scientific_audit.json").read_text())
    verification = json.loads((ROOT / "verification.json").read_text())
    assert audit["status"] == "PASS"
    assert verification["status"] in ("PASS", "PASS_WITH_PREEXISTING_STYLE_FAILURES")
    style_note = (
        f"Inherited style failures: {verification['preexisting_style']['lint_diagnostics']} "
        f"lint diagnostics in {len(verification['preexisting_style']['lint_files'])} "
        "frozen historical files; preserved by instruction. New v1.4 style checks pass."
    )
    count = re.search(r"(\d+) passed", verification["pytest"]["stdout"]).group(1)
    details = []
    for task, d in summary["tasks"].items():
        chosen = d["selection"]["final_selected"]
        val = d["validation"]["final"]
        if chosen and val["status"] == "COMPLETE":
            details.append(
                f"- {task}: {chosen['config']['arch']} / {chosen['phase']}, "
                f"{chosen['config']['ticks']} ticks, {chosen['config']['readout']} readout; "
                f"fresh validation {val['mean']:.6f}; target {d['target']}; {d['status']}."
            )
        else:
            details.append(
                f"- {task}: no stable configuration; bounded stopping criterion reached."
            )
    findings = "\n".join(details)
    timestamp = datetime.now(timezone.utc).isoformat()
    common = f"""The plan was committed as ee7d012 before training, on branch
flyarcade-v1.4-performance from dashboard base b444631. All {audit["historical_files_preserved"]}
protected historical files, including dashboard assets and README.md, are unchanged.
No historical confirmatory episodes were rerun/rescored and no new confirmatory
seeds were executed. Primary and final selections locked before fresh validation.
"""
    state = f"""# FlyArcade state

## Overall status
COMPLETE — v1.4 development and fresh validation only.

## Current milestone
None open. STOP: confirmatory testing requires explicit user approval and a new plan.

## Completed
- Three primary learners × five tasks × three fresh training seeds: 45 primary trials.
- Bounded conditional secondary factors and any triggered rescue attempts.
- Locked selections, fresh validation, machine-readable results and figures.
- Scientific audit PASS; repository tests PASS ({count} tests); v1.4 lint/format PASS.
{common}
## Outcomes
{findings}

## In progress
Nothing. No background training, validation or acquisition process remains.

## Next
1. Review artifacts/v14/final_summary.md and primary_comparison.csv.
2. Preserve ignored runs/v14 checkpoints and the original biological data.
3. Do not tune against validation or run confirmatory seeds without a new authorized plan.

## Tests
{count} passed; health, pip and new v1.4 Ruff lint/format checks pass.
{style_note} Audit confirms seed and
selection separation, matched primary budgets, immutable recurrent weights and
checkpoint hashes. Software replay uses fresh test seeds; no validation rescoring.

## Latest benchmark
Maximum trial process duration {audit["max_process_seconds"]:.3f} seconds;
maximum observed OS trial high-water RSS {audit["max_trial_peak_memory_mb"]:.2f} MiB.

## Resource status
Original 120-second/process and 2-GiB RSS guards unchanged; disk reserve 1 GiB.
No new connectome download, GPU or cloud use. Checkpointed processes resume safely.

## Known issues
Targets are development goals, not guaranteed results. Historical comparisons are
unmatched references. Secondary factors are conditional on primary winners; they
do not exhaust architecture/factor interactions. No fixed-substrate bottleneck
can be isolated without additional controls. n=3 uncertainty is descriptive.

## Last agent
Codex

## Timestamp
{timestamp}
"""
    Path("STATE.md").write_text(state)
    handoff = f"""# v1.4 handoff — development complete

{common}
## Final outcomes
{findings}

See artifacts/v14/final_summary.md for the requested primary-learner comparison
and selected validation table. Rescue outputs are separate from the primary CSV.
Historical 8×8 absolute-action Snake is used; v1.2 relative Snake and Breakout are
not part of this study. Only artificial downstream parameters learn; the authentic
2,040-neuron MaleCNS recurrent substrate and target environments stay fixed.

## Work completed and validation
- {audit["primary_trials"]} primary trials;
  {audit["total_completed_development_trials"]} total completed development trials.
- {audit["validation_evaluations"]} distinct selected-checkpoint validation evaluations.
- All attempts, configuration scores, histories, checkpoint hashes, peak RSS and
  elapsed time retained. Failed behavior is retained even when numerical runs pass.
- {count} tests pass. Scientific audit, health, pip and v1.4 style checks PASS.
- {style_note}
- Checkpoint software replay and full development-only Catch Linear training replay
  match. These are audit checks, not additional independent experimental seeds.
- {audit["historical_files_preserved"]} protected historical file hashes match.
  Dashboard behavior is unchanged.
- No background process remains. No resource guard was increased.

## Artifacts and commands
Read experiments/v14/development_plan.json, experiments/v14/METHODS.md,
artifacts/v14/implementation.json, selected_configs.json, validation_summary.json,
development_summary.json, scientific_audit.json and verification.json.
CSV files, result archive and PNG/SVG figures are under artifacts/v14/.
Full local checkpoints remain under ignored runs/v14/; retain backups.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/v14_suite.py
.venv/bin/python scripts/v14_validate.py
.venv/bin/python scripts/v14_report.py
.venv/bin/python scripts/v14_audit.py
.venv/bin/python scripts/v14_verify.py
```

Completed trials and validation are preserved on rerun. Never delete validation
outputs to facilitate additional tuning. Do not invoke historical scientific
replay scripts: the mission forbids reusing their spent confirmatory seeds.
RUNBOOK.md covers environment installation, individual trial resume and checks.

## Interpretation and next authorized action
Use the fixed common-PPO architecture comparison as the headline. Nonlinear
architectures are not individually optimized; secondary factor conclusions are
conditional and limited. Historical numerical differences are not matched causal
contrasts. Broader readout can expose engineered visual inputs directly. Failure
at a bounded budget is not proof of a fixed-biological-substrate capacity limit.
Review results next. Any new development requires a new untouched validation
block; confirmatory testing remains explicitly unauthorized.

## Repository
Work is isolated to flyarcade-v1.4-performance. Plan and implementation are committed;
new result/report changes are ready for the final v1.4 commit. Historical branches
remain unchanged. README.md and the historical manuscript were not edited.
Timestamp: {timestamp}
"""
    Path("HANDOFF.md").write_text(handoff)
    p = Path("CHANGELOG.md")
    old = p.read_text() if p.exists() else ""
    marker = "## v1.4 — prospective downstream architecture development"
    if marker not in old:
        p.write_text(
            marker + "\n\nCompleted matched Linear/MLP-PPO/GRU-PPO development on five "
            "unchanged tasks, bounded secondary analyses and fresh validation. "
            "Historical data and dashboard remain frozen; no confirmatory testing. "
            f"{count} tests and the scientific audit pass. See artifacts/v14/final_summary.md.\n\n"
            + old
        )
    print("Final v1.4 state, handoff and changelog generated.")


if __name__ == "__main__":
    main()
