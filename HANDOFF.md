# v1.4 handoff — development complete

The plan was committed as ee7d012 before training, on branch
flyarcade-v1.4-performance from dashboard base b444631. All 2022
protected historical files, including dashboard assets and README.md, are unchanged.
No historical confirmatory episodes were rerun/rescored and no new confirmatory
seeds were executed. Primary and final selections locked before fresh validation.

## Final outcomes
- catch: linear / ticks, 8 ticks, descending readout; fresh validation 0.938889; target 0.85; TARGET_MET.
- dodge: linear / primary, 4 ticks, descending readout; fresh validation 0.941111; target 0.97; BELOW_TARGET.
- snake: linear / all_readout, 8 ticks, all readout; fresh validation 12.156667; target 4.0; TARGET_MET.
- pong: linear / longer, 8 ticks, all readout; fresh validation 0.940370; target 0.9; TARGET_MET.
- flappy: gru / longer, 4 ticks, descending readout; fresh validation 0.297222; target 0.8; BELOW_TARGET.

See artifacts/v14/final_summary.md for the requested primary-learner comparison
and selected validation table. Rescue outputs are separate from the primary CSV.
Historical 8×8 absolute-action Snake is used; v1.2 relative Snake and Breakout are
not part of this study. Only artificial downstream parameters learn; the authentic
2,040-neuron MaleCNS recurrent substrate and target environments stay fixed.

## Work completed and validation
- 45 primary trials;
  81 total completed development trials.
- 27 distinct selected-checkpoint validation evaluations.
- All attempts, configuration scores, histories, checkpoint hashes, peak RSS and
  elapsed time retained. Failed behavior is retained even when numerical runs pass.
- 241 tests pass. Scientific audit, health, pip and v1.4 style checks PASS.
- Inherited style failures: 79 lint diagnostics in 7 frozen historical files; preserved by instruction. New v1.4 style checks pass.
- Checkpoint software replay and full development-only Catch Linear training replay
  match. These are audit checks, not additional independent experimental seeds.
- 2022 protected historical file hashes match.
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
Timestamp: 2026-09-13T18:57:44.206046+00:00
