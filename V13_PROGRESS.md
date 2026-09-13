# FlyArcade v1.3 progress

Branch: `flyarcade-v1.3-multitask` (user-specified name; mission text said
`flyarcade-v1.3-adaptive-learning`). Descends from v1.2 commit `00861e0`.

## Milestones
| Milestone | Status |
|---|---|
| M0_BRANCH_AND_PRESERVATION | COMPLETE |
| M1_ENVIRONMENT_VALIDATION | COMPLETE |
| M2_PPO_IMPLEMENTATION | COMPLETE |
| M3_SENSORY_ONLY_SOLVABILITY | COMPLETE (screen; 5-seed control validation pending in M8) |
| M4_REPRESENTATION_DIAGNOSTICS | COMPLETE for tick sweep (bio vs rewired pending post-freeze) |
| M5_FLY_MLP_DEVELOPMENT | COMPLETE (screen) |
| M8_DEVELOPMENT_VALIDATION | IN PROGRESS (mlp_validate + sensory_validate) |
| PONG_EXTERNAL_FREEZE | COMPLETE — frozen and confirmed separately; see D035 |
| M6..M14 | pending |

## Evidence
- M0: `artifacts/v13/historical_hashes.json` (638 protected files at 66a1272);
  `paper/manuscript_v12_historical.md` byte-identical archive. Baseline 105 tests pass.
- M1: `artifacts/v13/environment_validation.json` (100 fresh env_validation seeds/task).
  Flappy MPC oracle 1.000 with unchanged v1.2 dynamics (gate >= 0.70 passed; no FlappyV13).
  v1.2 heuristic 0.067. Random/reference: Pong 0.241/1.000, Breakout 0.454/0.584,
  Snake 0.017/0.956. Deterministic trajectories verified.
- M2: `src/flyarcade_v13/` NumPy MLP/GRU actor-critic with finite-difference-verified
  gradients, clipped PPO + GAE, batched frozen core bit-identical to the v1.2
  controller (incl. noise and lesions), exact checkpoint resume (MLP and GRU).
- M3: `artifacts/v13/development/sensory_screen.json`: 8 configs x 2 seeds x 4 tasks,
  every config beats random: Flappy 1.000, Pong 1.000, Breakout up to 0.753
  (reference 0.620), Snake up to 0.999.

## Failures / fixes
- Checkpoint resume differed in the last bit: F-ordered orthogonal-init views vs
  C-ordered restored arrays hit different Accelerate BLAS kernels. Fixed by
  C-contiguous parameters; resume is now exact.

- M4: `artifacts/v13/representation-<task>.json`: linear held-out state R2 rises with
  ticks (4 -> 16): flappy 0.25->0.57, pong 0.22->0.68, breakout 0.34->0.65, snake 0.20->0.55.
  Balanced action decoding stays modestly above majority. Short-episode tasks (Flappy,
  Snake) show low participation ratio / high correlation (post-reset transient).
- M5: `artifacts/v13/development/mlp_screen.json` (16 configs x 2 seeds x 4 tasks, 128/128).
  Best dev means: pong 0.829 (rand 0.244), breakout 0.518 (0.435), flappy 0.215 (0.000),
  snake 0.256 (0.018). All top-3 per task use 8 ticks. v1.2 fly results were
  0.350 / 0.368 / 0.000 / 0.043.
- Machine idle sleep paused trials once (23:10-23:32); suites now run under caffeinate.

## Pong is closed (external freeze, D035)
Pong was pre-registered at `56c604b` and confirmed at `94422be` on
`flyarcade-v1.3-pong-rescue`, merged here unchanged. Biological 0.793, untrained 0.306,
random 0.253, rewired 0.870, sensory-only 1.000, heuristic 1.000; criterion met.
**Pong confirmatory seeds 0-4 are spent.** `v13_freeze.py` skips Pong, `v13_suite.py`
refuses any spent Pong confirmatory spec, and `v13_report.py` reads Pong only from
`experiments/v13_external_frozen.json`. Do not select, freeze, re-run or re-score Pong.
Pong t8 rewired standardizers 0-4 already exist (identical procedure).

## Next action
Collect mlp_validate/sensory_validate; apply the development gate to Flappy, Breakout
and Snake; if they pass, write selection.json, fit rewired standardizers at 8 ticks,
freeze and run confirmatory for those three tasks only.
If a task fails: GRU rung (then curriculum) on development seeds only.
