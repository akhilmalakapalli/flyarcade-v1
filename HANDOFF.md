# Handoff — completed v1.2 task battery

The requested extension is complete, including negative outcomes. No process is
running and no blocker or resource guard is pending. Do not edit README.md or
mutate any historical experiment. No commit was made.

## Read first
FLYARCADE_MASTER.md, STATE.md, DECISIONS.md, experiments/METHODS_V12.md,
experiments/v12_run_plan.json and RUNBOOK.md. Preserve historical Catch/Dodge and
absolute-action Snake exactly. The prior paper is archived verbatim; the current
paper appends a generated extension.

## Results
- flappy: before 0.000000, biological 0.000000, random 0.000000, rewired 0.000000; criteria FAIL (0/3 seeds meet all components).
- pong: before 0.300926, biological 0.350000, random 0.252778, rewired 0.488889; criteria FAIL (2/3 seeds meet all components).
- breakout: before 0.365000, biological 0.368333, random 0.468333, rewired 0.401667; criteria FAIL (0/3 seeds meet all components).
- snake: before 0.005000, biological 0.042500, random 0.025000, rewired 0.061667; criteria FAIL (0/3 seeds meet all components).

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
  467 historical hashes preserved.

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
- Guards remain cooperative. Maximum segment 46.894 seconds;
  Trial RSS snapshot maximum 136.52 MiB, not a measured peak.

## Working tree and protected inputs
The new extension and allowed documentation changes are uncommitted, reviewable
and ready for a user-selected commit. README SHA256 remains
`a46572881f00b6963a0bbddcf2cc439d7acfd1ca46158711ad1a6fe8caaa7769`.
Source graph SHA256 is `3f4cbeedb17ca6a7d06162151429e0713ed72ce579d5f19a8845a446f6315697`.
The token file remains ignored; no token was printed or needed for this extension.
Raw data and checkpoints remain ignored under data/ and runs/; keep backups.
No full connectome download occurred. Any future tuning must use a new version,
new development seeds and a newly frozen protocol; never retrofit these results.
