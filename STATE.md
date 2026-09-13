# FlyArcade state

## Overall status
IN_PROGRESS — v1.4 development only on flyarcade-v1.4-performance.

## Current milestone
Balanced Linear / MLP-PPO / GRU-PPO primary comparison.

## Completed
- Latest dashboard base b444631 preserved; plan ee7d012 committed before training.
- 2,022 historical file hashes; original environments and recurrent weights reused.
- New adapters, affine PPO arm, bounded trainer, secondary/rescue orchestration.
- Existing 238-test suite passed; three additional imitation tests passed.

## In progress
- Primary grid: 5 tasks × 3 learners × 3 fresh training seeds.

## Next
1. Finish primary comparisons, lock primary selections.
2. Apply only predeclared secondary/rescue escalation.
3. Lock all choices, evaluate fresh validation once, report and audit.

## Tests
20 v1.4 tests passed; existing PPO and dashboard tests passed. No old confirmatory
episodes executed.

## Latest benchmark
Fresh feature fitting complete for Catch; primary timing pending.

## Resource status
Original guards unchanged: 120 seconds/process, 2 GiB RSS, 1 GiB disk reserve.
Checkpoint after 70 seconds; maximum 20 independently guarded segments per trial.

## Known issues
Development targets are not guaranteed. No confirmatory testing is authorized.

## Last agent
Codex
