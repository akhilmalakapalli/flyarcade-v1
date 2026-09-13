# v1.3 Flappy rescue — running log

Branch `flyarcade-v1.3-flappy-rescue` (worktree `../flyarcade-v1-flappy-rescue`), based on
`flyarcade-v1.3-multitask` commit `55e5b54`. Development seeds only until a freeze.

## Scientific question
Why does a PPO learner reading the frozen MaleCNS descending population reach only ~0.2
Flappy success when the same learner on the engineered state reaches 1.000, and can a
legitimate intervention close that gap?

## Established before this rescue (reused, not rerun)
- Target environment: unchanged v1.2 `Flappy`; MPC oracle 1.000 on 100 fresh dev seeds
  (`artifacts/v13/environment_validation.json`).
- Sensory-only PPO: 1.000 on all 16 screen runs (`runs/v13-dev/sensory_screen`).
- Fly MLP screen (16 configs x 2 dev seeds, 200k transitions): best 0.215 (c06: 8 ticks,
  entropy 0.01, no shaping); every config with the old no-flap-projection shaping is worse
  than its unshaped twin.
- Greedy vs stochastic mismatch ruled out: greedy 0.16-0.23 vs stochastic training
  success 0.10-0.18, entropy ~0.1, flap rate ~10% (close to the oracle's ~8%). The policies
  flap about as often as they should, but at the wrong moments.

## Coordination
The multitask session is concurrently validating Flappy c02/c06/c10 (5 dev seeds) on its
own branch and will likely attempt a Flappy GRU/curriculum rung. This rescue does not
touch its files or jobs. No Flappy confirmatory seed has been used by anyone yet.

## Engineering decision: rescue code is outside the hashed multitask code
`src/flyarcade_v13/**` and `scripts/v13_trial.py` are hashed into every multitask
checkpoint and frozen plan. All rescue code lives in `src/flyarcade_v13_flappy/` plus
`scripts/v13_flappy_*.py`; `install()` adds opt-in `encoding` / `potential` config keys
through dispatchers that fall back to the originals. Tests prove default configs train
bit-identically and the base code hash stays `504a571e...`.

## Phase 1 — representation diagnostic (`artifacts/v13/flappy-rescue/representation.json`)
6,075 matched development states (`representation` purpose, alternating oracle/random
behaviour), held out by episode, descending readout.

| Encoding, ticks | y | vy | time-to-pipe | gap | signed err | abs err | proj. crossing err | vel. err |
|---|---|---|---|---|---|---|---|---|
| v1.2, 4 | .57 | .59 | .76 | .40 | .54 | .40 | .61 | .61 |
| v1.2, 8 | .64 | .74 | .79 | .58 | .64 | .47 | .66 | .74 |
| v1.2, 12 | .70 | .84 | .82 | .68 | .75 | .54 | .72 | .82 |
| augmented, 12 | .79 | .63 | .82 | .59 | .82 | .70 | .82 | .78 |

(held-out Pearson r of the linear probe; linear R2, MLP R2 and action metrics in the JSON)

Oracle action (flap on 8% of test states): even the perfect engineered state reaches only
0.70 / 0.73 balanced accuracy (linear / MLP); fly features 0.49-0.56. The MPC action is a
discontinuous function of state, so action decoding cannot separate failure modes here.

**Interpretation.** Task variables are present but noisy (signal-to-noise limited, A-by-
noise): every variable improves monotonically with integration time. The MLP probe never
beats the linear probe, so nonlinearity (B) is not the limit. The augmented encoding
improves the decision variables (signed/absolute/projected error) but dilutes vy and gap
because 26 channels share 218 input neurons. Flappy state changes slowly (gap fixed for 23
steps), so temporal integration across steps (C: GRU) and longer ticks are the most
promising interventions.

**Next decision.** Focused screen at 12 ticks, v1.2 encoding: MLP with/without the arrival
potential, GRU with/without it. Augmented encoding held back until these report.

## Phase 2/3 — focused screen 1 (dev seeds 0,1; screen_train / screen_eval; 150k transitions)
Common: fly source, descending readout, v1.2 encoding, 12 ticks, lr 3e-4 annealed, clip 0.2,
gamma 0.99, lambda 0.95, 4 epochs, 16 envs x 128 steps, 8 minibatches, vf 0.5,
grad-norm 0.5, no exploration floor, 40 greedy evaluation episodes.

| Arm | Status |
|---|---|
| GRU 128->64, no shaping, ent 0.003 | running |
| GRU, arrival potential x1.0, ent 0.003 | **terminated at ~41k**: training success 0.02-0.05 vs 0.15-0.20 for the matched no-shaping GRU |
| MLP, arrival potential x1.0 | never started (dropped after the GRU arm above) |
| GRU, no shaping, ent 0.001, 150k | greedy 0.525 / 0.421 (mean 0.473), probe [28, 12] / [34, 6]; below ent 0.003 |
| GRU, arrival potential x0.5, ent 0.003 | running (the gentler end of the requested range) |
| MLP 128, no shaping, ent 0.003 | running (control: does the GRU add anything at 12 ticks?) |

**Why arrival shaping x1.0 hurt.** Flapping is bang-bang: every flap snaps vy to +0.055, far
from the smooth "desired velocity", so the potential drops sharply on each flap and the
shaping term penalises exactly the action the task needs. The previous no-flap-projection
shaping also hurt in every matched multitask screen pair. Dense velocity-matching shaping is
misaligned with this control problem.

**Early signal at ~41k transitions.** GRU without shaping: training success 0.19 / 0.17, one
seed's greedy curve 0.396, already above the best final MLP result of the previous screen.

Note: the multitask session's own Flappy validation jobs stopped by themselves around 04:31
(last log at 26k-57k of 500k transitions); CPUs were free when this screen was relaunched.

### Screen 1 outcome
| Arm (12 ticks, v1.2 encoding) | Result |
|---|---|
| **GRU, no shaping, ent 0.003, 150k** | **greedy dev success 0.583 / 0.637 (mean 0.610)**, before 0.000, probe actions [30, 10] both seeds, finite; curves peaked 0.75 |
| GRU, no shaping, ent 0.001, 150k | greedy 0.525 / 0.421 (mean 0.473), probe [28, 12] / [34, 6]; below ent 0.003 |
| MLP, no shaping | terminated at 49k: training 0.020 / 0.013 vs GRU 0.19 / 0.17 at ~41k |
| GRU, arrival x0.5 | terminated at ~46k: training 0.025 / 0.112 |
| GRU, arrival x1.0 | terminated at 81,920: training 0.052 / 0.120; greedy curve s0 0.073, s1 0.417 |

**Interpretation.** Temporal integration is the decisive intervention: with ticks, encoding,
budget and seeds matched, the GRU learns and the MLP does not. Both potential shapings slow
learning. The GRU curve is still rising at 150k.

**Next decisions (running in parallel).** Screen 2: the same GRU at 400k transitions, and
at 16 ticks (decoding improved monotonically to 16). Escalation arm: v1.3 Flappy augmented
sensory encoding + the same GRU at 12 ticks, after fitting its own development standardizer.

### Screen 2 outcome (dev seeds 0,1)
| GRU, no shaping, ent 0.003 | Greedy dev success | Probe | Curve tail |
|---|---|---|---|
| **16 ticks, 150k** | **0.746 / 0.742 (mean 0.744)** | [24,16] / [25,15] | up to 0.865, still rising |
| 12 ticks, augmented encoding, 150k | 0.433 / 0.592 (mean 0.513) | [35,5] / [32,8] | below v1.2 encoding at 12 ticks (0.610) |
| 12 ticks, 400k | 0.400 / 0.533 (mean 0.467) | [38,2] / [38,2] | 0.31-0.57; **worse than 150k** (0.610) |

**Interpretation.** Longer integration keeps helping (12 -> 16 ticks: 0.610 -> 0.744 at a matched
budget), consistent with the representation diagnostic. The augmented encoding does not help the
learner: splitting 218 input neurons across 26 channels degrades vy and gap more than the derived
channels help. It is dropped; the final encoding remains the unchanged v1.2 encoding.

**Next (screen 3, running).** GRU at 16 ticks with a 300k budget, and at 20 ticks with 150k (own
development standardizer fitted by the unchanged `v13_prepare.py` procedure).

**Budget is not monotone.** The 12-tick GRU at 400k transitions (lr annealed over the whole
budget) finished below the same configuration at 150k (0.467 vs 0.610), with less
state-dependent probes. Longer budgets therefore need a direct comparison, not an assumption.

### Screen 3 outcome and candidate selection
| GRU, no shaping, ent 0.003 | Greedy dev success (seeds 0,1) | Probe |
|---|---|---|
| 20 ticks, 150k | 0.617 / 0.692 (mean 0.654) | [24,16] / [28,12] |
| 16 ticks, 300k | 0.662 / 0.608 (mean 0.635) | [21,19] / [20,20]; below 150k again |

Integration sweep at 150k: 12 ticks 0.610, **16 ticks 0.744**, 20 ticks 0.654. **Selected for
5-seed development validation: GRU (128 tanh -> GRU 64), 16 ticks, v1.2 encoding, descending
readout, no shaping, entropy 0.003, lr 3e-4 annealed, 150k transitions.** Chosen before
validation from the screen seeds; the 12-tick 400k result argues against assuming a longer
budget helps. Rewired t16 standardizers 0-4 fitted by the unchanged procedure.

## Development validation (5 fresh dev seeds; validate_train / validate_eval; 60 episodes)
GRU (128 tanh -> GRU 64), 16 ticks, v1.2 encoding, descending readout, no shaping, ent 0.003,
150k transitions.

| Seed | After | Before | State probe | Greedy top-action fraction |
|---|---|---|---|---|
| 0 | 0.553 | 0.000 | 0.450 | 0.880 |
| 1 | 0.553 | 0.003 | 0.200 | 0.899 |
| 2 | 0.617 | 0.000 | 0.225 | 0.891 |
| 3 | 0.811 | 0.003 | 0.275 | 0.898 |
| 4 | 0.731 | 0.000 | 0.200 | 0.893 |
| **mean** | **0.653** | 0.001 | 0.270 | |

Random 0.000, MPC 1.000, gap closure 0.653, 5/5 seeds beat random + 0.10, finite, graph hashes
match. **Gate: PASS** (`artifacts/v13/flappy-rescue/validation_gate.json`). The preferred 0.70
level (reported only) was not reached. Decision: freeze. Further screening would select on
already-seen seeds, and both longer-budget attempts (12 ticks 400k, 16 ticks 300k) got worse.
