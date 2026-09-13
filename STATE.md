# FlyArcade state

## Overall status
COMPLETE — v1.4 development and fresh validation only.

## Current milestone
None open. STOP: confirmatory testing requires explicit user approval and a new plan.

## Completed
- Three primary learners × five tasks × three fresh training seeds: 45 primary trials.
- Bounded conditional secondary factors and any triggered rescue attempts.
- Locked selections, fresh validation, machine-readable results and figures.
- Scientific audit PASS; repository tests PASS (241 tests); v1.4 lint/format PASS.
The plan was committed as ee7d012 before training, on branch
flyarcade-v1.4-performance from dashboard base b444631. All 2022
protected historical files, including dashboard assets and README.md, are unchanged.
No historical confirmatory episodes were rerun/rescored and no new confirmatory
seeds were executed. Primary and final selections locked before fresh validation.

## Outcomes
- catch: linear / ticks, 8 ticks, descending readout; fresh validation 0.938889; target 0.85; TARGET_MET.
- dodge: linear / primary, 4 ticks, descending readout; fresh validation 0.941111; target 0.97; BELOW_TARGET.
- snake: linear / all_readout, 8 ticks, all readout; fresh validation 12.156667; target 4.0; TARGET_MET.
- pong: linear / longer, 8 ticks, all readout; fresh validation 0.940370; target 0.9; TARGET_MET.
- flappy: gru / longer, 4 ticks, descending readout; fresh validation 0.297222; target 0.8; BELOW_TARGET.

## In progress
Nothing. No background training, validation or acquisition process remains.

## Next
1. Review artifacts/v14/final_summary.md and primary_comparison.csv.
2. Preserve ignored runs/v14 checkpoints and the original biological data.
3. Do not tune against validation or run confirmatory seeds without a new authorized plan.

## Tests
241 passed; health, pip and new v1.4 Ruff lint/format checks pass.
Inherited style failures: 79 lint diagnostics in 7 frozen historical files; preserved by instruction. New v1.4 style checks pass. Audit confirms seed and
selection separation, matched primary budgets, immutable recurrent weights and
checkpoint hashes. Software replay uses fresh test seeds; no validation rescoring.

## Latest benchmark
Maximum trial process duration 62.892 seconds;
maximum observed OS trial high-water RSS 330.89 MiB.

## Resource status
Original 120-second/process and 2-GiB RSS guards unchanged; disk reserve 1 GiB.
No new connectome download, GPU or cloud use. Checkpointed processes resume safely.

## Known issues
Targets are development goals, not guaranteed results. Historical comparisons are
unmatched references. Secondary factors are conditional on primary winners; they
do not exhaust architecture/factor interactions. No fixed-substrate bottleneck
can be isolated without additional controls. n=3 uncertainty is descriptive.

## Last agent
Claude (resumed from Codex)

## Timestamp
2026-09-13T18:57:44.206046+00:00
