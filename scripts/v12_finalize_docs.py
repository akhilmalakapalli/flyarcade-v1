"""Generate final state and handoff from verified v1.2 results."""

import json
from pathlib import Path

ROOT = Path("artifacts/v12")


def main():
    summary = json.loads((ROOT / "results_summary.json").read_text())
    audit = json.loads((ROOT / "scientific_audit.json").read_text())
    verification = json.loads((ROOT / "verification.json").read_text())
    assert audit["status"] == verification["status"] == "PASS"
    lines = []
    for task in ("flappy", "pong", "breakout", "snake"):
        d = summary["tasks"][task]
        primary = d["primary"]
        passed = sum(
            c["improved"] and c["beats_random_margin"] and c["state_sensitive"]
            for c in d["success_checks"]
        )
        lines.append(
            f"- {task}: before {d['before']['mean']:.6f}, biological "
            f"{primary['biological']['mean']:.6f}, random {primary['random']['mean']:.6f}, "
            f"rewired {primary['rewired']['mean']:.6f}; "
            f"criteria {'PASS' if d['meets_frozen_success_criteria'] else 'FAIL'} "
            f"({passed}/3 seeds meet all components)."
        )
    results = "\n".join(lines)
    state = f"""# FlyArcade state — v1.2 complete

## Status and milestones
Implementation, development, protocol freeze, 36 confirmatory trials, matched
controls, perturbations, post-hoc representations, six-task aggregation, figures,
audit and generated manuscript extension are complete. No milestone remains open.
M0 and biological acquisition were reused, not restarted. Canonical dataset:
HHMI Janelia MaleCNS `male-cns:v1.0`, authentic 2,040-neuron subgraph.

## Exact new outcomes
{results}

Success is the frozen conjunction of improvement, random +0.05 margin and
fixed-history state sensitivity on every seed. Failure does not mean weights
failed to change or prove the task impossible. Partial improvements are reported.
New Snake is relative-action 6×6, separate from historical absolute-action Snake.

## Preservation and validation
- 105 tests pass; health, pip consistency, Ruff lint and format checks pass.
- Scientific audit PASS: 36/36 new trials; seed separation and weight invariants.
- All four seed-zero checkpoints replay intact and all three perturbations exactly.
- Fresh full-budget Flappy seed-zero training history and actor/critic match exactly.
- All 18 historical Catch/Dodge checkpoint evaluations match exactly.
- All {audit["historical_files_preserved"]} protected historical files remain byte-identical.
- README.md unchanged; historical paper preserved in manuscript_v11_historical.md.
- Frozen source hash: `{audit["code_sha256"]}`.

## Resources and processes
No background process, download or training job remains. No guard was increased.
Maximum trial segment {audit["max_segment_seconds"]:.3f} seconds;
maximum recorded trial RSS snapshot {audit["max_rss_snapshot_mb"]:.2f} MiB (not peak).
Unchanged guards: 120 seconds/operation, 2 GiB RSS, 1 GiB disk reserve,
128 MiB download, 4,096 neurons, 250,000 edges. No new download was needed.

## Artifacts and next action
See artifacts/v12/results_summary.json, scientific_audit.json, verification.json,
results_archive.json.gz, CSV tables and figures/. Paper text is generated from
these audited results. RUNBOOK.md contains exact reproduction commands.
Changes are uncommitted and reviewable on the existing branch. No routine user
input is needed. Any additional tuning requires a new development plan and version.
Preserve ignored data/ and runs/ in backups; they contain authentic source responses
and full checkpoints. No credential value is included in artifacts.
"""
    Path("STATE.md").write_text(state)
    handoff = f"""# Handoff — completed v1.2 task battery

The requested extension is complete, including negative outcomes. No process is
running and no blocker or resource guard is pending. Do not edit README.md or
mutate any historical experiment. No commit was made.

## Read first
FLYARCADE_MASTER.md, STATE.md, DECISIONS.md, experiments/METHODS_V12.md,
experiments/v12_run_plan.json and RUNBOOK.md. Preserve historical Catch/Dodge and
absolute-action Snake exactly. The prior paper is archived verbatim; the current
paper appends a generated extension.

## Results
{results}

The frozen criteria require all three seeds to pass. Pong's first seed reaches
exactly the random +0.05 margin, which fails the strict greater-than criterion.
See results_summary.json for per-seed checks, descriptive intervals, raw outcomes
and all perturbations. Small gains are retained even when task-level criteria fail.
No recurrent plasticity was introduced: only the downstream actor–critic learned.

## Deliverables and validation
- New code: src/flyarcade/v12/ and scripts/v12_*.py.
- New tests: tests/test_v12_environments.py and test_v12_learning.py.
- Frozen development and confirmatory plans: experiments/v12_*.json.
- Artifacts: artifacts/v12/ primary/contrast/perturbation/raw/representation CSVs,
  full result archive, standardizers, core equivalence, historical replay,
  verification and scientific audits; five PNG/SVG figure pairs.
- 105 tests; health, pip, Ruff checks PASS. Scientific audit PASS for all 36 trials.
- Checkpoint replay covers every new task and perturbation at seed zero; a full
  Flappy repetition reproduces history and learned arrays. All 18 historical
  Catch/Dodge evaluations reproduce exactly;
  {audit["historical_files_preserved"]} historical hashes preserved.

## Reproduce in the existing environment
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/v12_restore_graphs.py
.venv/bin/python scripts/v12_suite.py
for task in flappy pong breakout snake; do
  .venv/bin/python scripts/v12_representation.py --task "$task"
done
.venv/bin/python scripts/v12_report.py
.venv/bin/python scripts/v12_audit.py
.venv/bin/python scripts/v12_manuscript.py
.venv/bin/python scripts/v12_verify.py
```
Completed trials and representation outputs are preserved. For a fresh repetition,
use a separate workspace with the delivered frozen plan, standardizers and source
graph, without that copy's v1.2 run outputs. Never delete historical runs.
Installation, bounded development, freeze and full checkpoint/perturbation replay
commands are in RUNBOOK.md. Do not rerun historical artifact writers merely to
verify this extension: verify.sh overwrites the protected historical timing file.

## Scientific and operational limitations
- Three training seeds; bootstrap intervals are descriptive and no p-values claimed.
- Artificial engineered inputs/actions, truncated circuit, simplified LIF/sign
  assumptions and fixed readout architecture constrain interpretation.
- Only ~15.05% incoming and ~7.12% outgoing anatomical synaptic weight retained.
- Development was limited to two candidates. Flappy's heuristic and feature-fit
  coverage are limited; new Snake differs from the previously successful task.
- Rewiring preserves degrees and outgoing counts, not all biological properties
  or incoming strength. It is not uniform random graph sampling.
- Representation analyses are post-hoc and use matched exogenous states. Biological
  replicas yield identical deterministic activity, not independent preparations.
- Perturbations of failed policies do not establish meaningful robustness.
- No model bug required confirmatory invalidation or parameter changes. A v1.2-only
  CSR optimization was verified exact before freeze and again on authentic graphs.
- Guards remain cooperative. Maximum segment {audit["max_segment_seconds"]:.3f} seconds;
  Trial RSS snapshot maximum {audit["max_rss_snapshot_mb"]:.2f} MiB, not a measured peak.

## Working tree and protected inputs
The new extension and allowed documentation changes are uncommitted, reviewable
and ready for a user-selected commit. README SHA256 remains
`{verification["readme_sha256"]}`.
Source graph SHA256 is `{audit["source_graph_sha256"]}`.
The token file remains ignored; no token was printed or needed for this extension.
Raw data and checkpoints remain ignored under data/ and runs/; keep backups.
No full connectome download occurred. Any future tuning must use a new version,
new development seeds and a newly frozen protocol; never retrofit these results.
"""
    Path("HANDOFF.md").write_text(handoff)
    print("STATE.md and HANDOFF.md generated from passing verification and audit.")


if __name__ == "__main__":
    main()
