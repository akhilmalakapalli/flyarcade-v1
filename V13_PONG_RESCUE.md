# v1.3 Pong rescue — handoff

Branch `flyarcade-v1.3-pong-rescue`, a git worktree at `../flyarcade-v1-pong-rescue`,
based on `flyarcade-v1.3-multitask` commit `2f17881`. The frozen plan was committed
(`56c604b`) **before** any confirmatory run.

## Coordination note for the multitask session
A separate session was running the v1.3 multitask screen in the main working tree.
By user decision this sprint did not rerun that screen: it took the completed Pong
screen, validated the best configuration on fresh development seeds, froze it, and
ran confirmatory trials. It ran in its own worktree so nothing under that session's
running jobs was touched. Its main-tree files were not modified.

**Consequence the multitask session must respect:** Pong confirmatory seeds 0–4
(`conf_train`, `conf_model`, `conf_rollout`, `conf_eval`, `conf_probe`) have now been
used by this frozen plan. A later, different Pong freeze on the same confirmatory
seeds would be a second look at held-out data and must be reported as such, not as an
independent confirmation.

## Result — confirmatory criterion MET
Greedy success (hits / attempted returns), 40 confirmatory episodes per seed,
learning off. Criterion pre-registered in `experiments/v13_pong_development.json`.

| Condition | Mean | Seeds 0–4 |
|---|---|---|
| biological, learned | **0.793** | 0.778, 0.772, 0.794, 0.803, 0.819 |
| biological, untrained (frozen) | 0.306 | 0.275, 0.286, 0.317, 0.236, 0.414 |
| degree-rewired, learned | 0.870 | 0.869, 0.869, 0.883, 0.881, 0.847 |
| sensory-only, learned | 1.000 | 1.000 × 5 |
| random | 0.253 | |
| heuristic | 1.000 | |

Every biological seed beats its untrained score and random + 0.10, is finite, and is
state-dependent; the mean is at least 0.70.

## What was frozen
NumPy MLP actor-critic (128 tanh) + clipped PPO + GAE, Adam; 8 ticks per action,
descending-neuron readout; lr 3e-4 annealed, entropy 0.003, potential-based intercept
shaping coefficient 1.0; γ 0.99, λ 0.95, clip 0.2, 4 epochs, 16 envs × 128 steps,
8 minibatches, value coefficient 0.5, grad-norm 0.5; no exploration floor;
150,000 transitions. Frozen config hash
`0d3783628e341cc4a974580bb42f3a9b816d33fd41b1bf2cd2fa98675f015d39`.

## Evidence trail
- Phase 1 decoding (development): `artifacts/v13/representation-pong.json`.
- 16-config screen: every 8-tick config 0.742–0.829, every 4-tick config 0.486–0.654.
- Validation gate on fresh dev seeds: 0.825 / 0.811 / 0.794, `artifacts/v13/pong/validation_gate.json`.
- Confirmatory: `artifacts/v13/pong/confirmatory_summary.json`, `confirmatory_per_seed.csv`.
- Verification: `artifacts/v13/pong/confirmatory_verification.json` (graph hashes,
  reference reproduction, exact policy replay, seed disjointness).

## Caveats
- The recurrent MaleCNS core stayed frozen; learning is in an artificial MLP readout.
- Supervised decoding of the exact heuristic action from fly features is poor (MLP
  balanced accuracy 0.506 at 8 ticks), yet the reward learner reaches 0.79. Exact
  heuristic-action imitation is a stricter target than the task needs.
- The learned policies almost never choose "stay" (0–1 of 40 probe states): bang-bang
  left/right control. They are still state-dependent.
- Rewired beats biological on all five seeds, consistent with earlier tasks.

## Reproduce
```sh
cd ../flyarcade-v1-pong-rescue
export PYTHONPATH=src
python scripts/v13_suite.py experiments/v13_pong/specs/confirm/*.json --workers 6 --log-dir runs/v13-pong-rescue/logs
python scripts/v13_pong_rescue.py report
```
The worktree needs `data/malecns-v1.0`, `data/v12` and `data/v13` (symlinked here to
the main checkout's ignored `data/`).
