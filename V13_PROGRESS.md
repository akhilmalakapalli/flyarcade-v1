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
| M4_REPRESENTATION_DIAGNOSTICS | IN PROGRESS |
| M5_FLY_MLP_DEVELOPMENT | IN PROGRESS |
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

## Next action
Collect `mlp_screen`, finish tick-sweep representation diagnostics, then choose
top-3 per task for 5-seed full-budget validation (or GRU/curriculum rung).
