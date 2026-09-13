# FlyArcade v1.5 — Phase 1 diagnosis

Written before any v1.5 development training. All numbers below come from fresh v1.5
`diagnosis` / `diagnosis_training` seeds (namespace [1e12, 1.5e12)), read-only v1.4
development histories, or read-only replays of v1.4 *development* checkpoints. No v1.4
validation seed, historical confirmatory seed or v1.3 confirmatory model was used.
Machine-readable sources are in `artifacts/v15/diagnosis/`.

## Summary

| Task | v1.4 validation | Dominant limitation (evidence below) | Secondary limitations |
|---|---:|---|---|
| Catch | 0.939 | B integration time (4–8 ticks) + D/E policy/optimisation at v1.4 operating point | residual far-distance misses |
| Dodge | 0.941 | B integration time | edge-lane precision |
| Snake | 12.16 | C readout scope: v1.4 gain was **input leakage**; substrate-mediated path limited by B and F | self-collision (shared with heuristic) |
| Pong | 0.940 | C readout scope: v1.4 gain was **input leakage**; substrate path limited by B | far-serve tracking |
| Flappy | 0.297 | **B integration time (v1.4 used 4 ticks; v1.3 used 16)** | gap-alignment precision at the pipe; F budget |

Letters follow the requested categories: A substrate representation, B integration time,
C readout scope, D policy architecture, E optimisation, F budget, G environment/reward.
No task shows evidence of an A-type hard ceiling at 16–24 ticks (see §6).

## 1. Why Flappy regressed from 0.605 (v1.3) to 0.297 (v1.4)

Code audit: v1.4's Flappy pipeline is v1.3's pipeline (same frozen LIF core, v1.2
encoder, GRU 128→64, PPO/GAE code, 16 envs × 128 steps, chunk 16, 4 epochs, 8
minibatches, ent 0.003, lr 3e-4 linear anneal, descending readout, standardizer
procedure; `diff src/flyarcade_v13/study.py src/flyarcade_v14/study.py` differs only in
seed plumbing, defaults and evaluation bookkeeping). The configuration differences were:

| Factor | v1.3 frozen Flappy | v1.4 selected Flappy |
|---|---|---|
| ticks/action | **16** | **4** (primary rule fixed ticks=4; the only tick factor tested was 8, at 65k) |
| transitions | 150,000 | 131,072 ("longer") |
| architecture / GRU size | GRU 128→64 | GRU 128→64 |
| readout | descending | descending |
| PPO settings, recurrent handling, normalisation, curriculum | identical | identical (curriculum/rescue arms were not selected) |
| checkpoint selection | final | final |

Controlled reconstruction on fresh diagnosis seeds (2 seeds each, GRU, descending,
150k transitions, identical otherwise):

| ticks | seed scores | mean |
|---:|---|---:|
| 4 | 0.298, 0.245 | 0.272 |
| **16** | **0.562, 0.673** | **0.617** |

This reproduces both historical numbers (0.605 and 0.297). The regression is explained by
integration time, not by a GRU/recurrent-state bug, normalisation drift or seed luck.
The v1.4 tick factor (8 ticks, 65k) also hurt, which is why escalation never reached 16.
No hidden-state leakage was found: resets zero the GRU row before the first step of each
episode in rollout, replay and evaluation, and checkpoint resume is bit-identical
(6-segment resume test: identical parameters, history and evaluation rows).

Representation evidence (linear state R² from descending neurons): 0.10 at 4 ticks,
0.32 at 8, 0.44 at 12, 0.54 at 16, 0.66 at 24. At 4 ticks the descending population
also shows a refractory-driven period-2 alternation between decisions
(PC1 lag-1 autocorrelation −0.77), which disappears by 16 ticks (+0.27).

Behavioural failure mode of the v1.4 Flappy model (fresh diagnosis seeds): 86–89% of
episodes end **at a pipe** (above or below the gap about equally), 5–7% at the boundary.
The policy flies but aligns imprecisely with the gap — consistent with a noisy estimate
of height/velocity relative to the gap.

## 2. Representation (Phase-1 decoding, matched states)

~4,000 matched states per task (reference action on even steps, random on odd), two
independent neural RNG repeats, ridge penalties chosen on an inner episode split.
Cells: mean linear state R² / reference-action balanced accuracy on held-out episodes.

| Task | Ticks | descending | nonvisual (DN+CB) | cb_intrinsic | pooled types | visual inputs | all 2,040 | sensory encoding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catch | 4 | 0.46 / 0.84 | 0.52 / 0.84 | 0.18 / 0.49 | 0.44 / 0.73 | 0.95 / 1.00 | 0.93 / 1.00 | 1.00 / 1.00 |
| catch | 8 | 0.65 / 0.93 | 0.67 / 0.94 | 0.32 / 0.57 | 0.65 / 0.87 | 0.98 / 1.00 | 0.96 / 1.00 | 1.00 / 1.00 |
| catch | 12 | 0.75 / 0.96 | 0.73 / 0.96 | 0.33 / 0.60 | 0.73 / 0.93 | 0.99 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 |
| catch | 16 | 0.81 / 0.98 | 0.81 / 0.98 | 0.45 / 0.65 | 0.79 / 0.96 | 0.99 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 |
| dodge | 4 | 0.45 / 0.76 | 0.51 / 0.76 | 0.21 / 0.50 | 0.43 / 0.71 | 0.95 / 0.97 | 0.93 / 0.91 | 1.00 / 1.00 |
| dodge | 8 | 0.61 / 0.84 | 0.64 / 0.81 | 0.30 / 0.54 | 0.62 / 0.77 | 0.98 / 0.98 | 0.96 / 0.90 | 1.00 / 1.00 |
| dodge | 12 | 0.72 / 0.88 | 0.69 / 0.86 | 0.30 / 0.57 | 0.67 / 0.84 | 0.99 / 0.98 | 0.97 / 0.94 | 1.00 / 1.00 |
| dodge | 16 | 0.78 / 0.88 | 0.80 / 0.89 | 0.43 / 0.63 | 0.76 / 0.87 | 0.99 / 0.98 | 0.98 / 0.95 | 1.00 / 1.00 |
| snake | 4 | 0.47 / 0.65 | 0.48 / 0.65 | 0.10 / 0.44 | 0.37 / 0.60 | 0.94 / 0.95 | 0.88 / 0.94 | 1.00 / 0.96 |
| snake | 8 | 0.62 / 0.76 | 0.64 / 0.78 | 0.18 / 0.47 | 0.56 / 0.68 | 0.97 / 0.96 | 0.94 / 0.94 | 1.00 / 0.96 |
| snake | 12 | 0.69 / 0.80 | 0.69 / 0.81 | 0.21 / 0.47 | 0.64 / 0.70 | 0.98 / 0.95 | 0.96 / 0.94 | 1.00 / 0.96 |
| snake | 16 | 0.73 / 0.82 | 0.73 / 0.84 | 0.22 / 0.52 | 0.69 / 0.73 | 0.99 / 0.96 | 0.96 / 0.94 | 1.00 / 0.96 |
| pong | 4 | 0.25 / 0.48 | 0.24 / 0.50 | -0.03 / 0.41 | 0.20 / 0.46 | 0.88 / 0.65 | 0.75 / 0.64 | 1.00 / 0.72 |
| pong | 8 | 0.45 / 0.53 | 0.46 / 0.54 | 0.01 / 0.45 | 0.44 / 0.52 | 0.94 / 0.68 | 0.88 / 0.65 | 1.00 / 0.72 |
| pong | 12 | 0.58 / 0.56 | 0.56 / 0.59 | 0.09 / 0.49 | 0.57 / 0.56 | 0.96 / 0.69 | 0.91 / 0.68 | 1.00 / 0.72 |
| pong | 16 | 0.68 / 0.59 | 0.61 / 0.60 | 0.10 / 0.50 | 0.65 / 0.58 | 0.97 / 0.69 | 0.93 / 0.68 | 1.00 / 0.72 |
| pong | 24 | 0.75 / 0.61 | 0.73 / 0.62 | 0.21 / 0.54 | 0.74 / 0.59 | 0.98 / 0.70 | 0.96 / 0.68 | 1.00 / 0.72 |
| flappy | 4 | 0.10 / 0.64 | 0.02 / 0.68 | 0.24 / 0.67 | 0.30 / 0.63 | 0.83 / 0.70 | 0.68 / 0.69 | 0.97 / 0.70 |
| flappy | 8 | 0.32 / 0.61 | 0.20 / 0.61 | 0.17 / 0.57 | 0.38 / 0.58 | 0.90 / 0.70 | 0.79 / 0.65 | 0.97 / 0.70 |
| flappy | 12 | 0.44 / 0.65 | 0.36 / 0.66 | 0.22 / 0.58 | 0.47 / 0.62 | 0.92 / 0.72 | 0.84 / 0.72 | 0.97 / 0.70 |
| flappy | 16 | 0.54 / 0.65 | 0.45 / 0.63 | 0.11 / 0.52 | 0.54 / 0.62 | 0.94 / 0.71 | 0.87 / 0.69 | 0.97 / 0.70 |
| flappy | 24 | 0.66 / 0.64 | 0.59 / 0.63 | 0.23 / 0.55 | 0.64 / 0.64 | 0.95 / 0.72 | 0.90 / 0.67 | 0.97 / 0.70 |

Findings:

* **Integration time helps every task monotonically** for the substrate-mediated readouts
  (descending, nonvisual, pooled). No saturation by 16 ticks for Pong/Flappy (still rising
  at 24).
* **Direct input leakage:** the 218 visual-projection neurons are driven directly by
  Bernoulli copies of the encoded observation (≈218/C neurons per channel). They decode
  state almost as well as the sensory encoding itself (R² 0.83–0.99). The all-2,040
  readout is never better than visual-only.
* Descending ≈ nonvisual (descending + central-brain intrinsic): the 529 intrinsic
  neurons add little linear information; alone they are weak (R² ≤ 0.45).
* Per-neuron trial-to-trial SNR is < 1 for > 95% of descending neurons in every task
  (median 0.03–0.55): information is distributed, so wider integration and population
  readout matter more than any single neuron. Pong descending activity has the lowest
  per-neuron SNR (≈0.03) and the highest participation ratio (≈260–340).
* Normalisation is not a bottleneck: standardized clip fraction ≤ 0.1% and floor-amplified
  neurons ≤ 3.5% for descending readouts.
* Reference-action decoding is a weak target for Flappy/Pong even from the perfect
  sensory encoding (0.70/0.72): those references are planning outcomes, not linear
  functions of the current observation, so policy memory/nonlinearity is needed.

## 3. Readout scope and sensory leakage — behavioural test

Linear PPO, 8 ticks, 65,536 transitions (the v1.4 all-readout operating point),
diagnosis seeds, 2 seeds:

| Task | all 2,040 | visual inputs only | nonvisual (no inputs) |
|---|---:|---:|---:|
| Snake (food) | 11.17 | **14.02** | 3.81 |
| Pong | 0.873 | 0.868 | 0.721 |

The v1.4 all-readout gains (Snake 3.3→11.6, Pong 0.65→0.87) are reproduced entirely by
reading the directly stimulated input neurons, and removing them loses the gain.
**Conclusion: v1.4's Snake and Pong headline improvements were predominantly direct
sensory leakage, not network computation.** v1.5 must separate the two.

## 4. Learner diagnostics (v1.4 histories, `v14_learner_diagnostics.*`)

* Every v1.4 run had PPO clip fraction ≈ 0.000–0.014 and approx-KL ≈ 1e-5–1e-3:
  updates were very small (gradients of norm 2–3 clipped to 0.5 at lr 3e-4).
* The "collapsed" MLP/GRU arms on Catch/Dodge were largely **non-learning**: entropy
  1.10→1.09 (MLP Catch), not action collapse after learning; they operated on the
  4-tick representation whose descending state R² is only ≈0.45.
* Flappy GRU arms: entropy fell 0.5→0.2 and rollout dominant action ≈0.86–0.88 — a
  mostly-no-flap policy, as expected for this task, not a collapse; value loss rose over
  training (critic tracking an improving policy).
* Critic explained variance is low in v1.5 diagnostics too (−0.6 to 0.7), including
  solved sensory runs, so the critic is weak but not the limiting factor.
* Late collapse was mild (v1.4 curve peak − final ≤ 0.09 except Snake all-readout 0.85
  food, Snake t8 0.27), so best-checkpoint selection is a cheap safeguard.

Diagnostic step-size test (GRU, 16 ticks, descending, diagnosis seeds, 2 seeds):

| Task | lr 3e-4 | lr 1e-3 |
|---|---:|---:|
| Catch | 0.981 | 0.995 |
| Dodge | 0.988 | 0.984 |
| Snake | 9.16 | 7.71 |
| Pong | 0.836 | 0.879 |
| Flappy | 0.617 | 0.333 (probe 0.05 / 0.00 → collapse) |

A larger learning rate is not a general fix; it helps Catch/Pong slightly and harms
Flappy/Snake. Optimisation (E) is secondary to integration time (B).

## 5. Sensory-only ceiling (no SNN; same PPO; 131k, Flappy 150k; 2 seeds)

| Task | Linear | GRU | reference controller | random |
|---|---:|---:|---:|---:|
| Catch | 0.943 | 1.000 | 1.000 | 0.21 |
| Dodge | 1.000 | 0.991 | 1.000 | 0.80 |
| Snake | 8.44 | 11.99 | 16.35 | 0.17 |
| Pong | 0.703 | 1.000 | 1.000 | 0.26 |
| Flappy | 0.262 | 1.000 | 1.000 | 0.00 |

With perfect information this PPO/GRU learner solves Catch, Pong and Flappy within the
budget, so on those tasks the remaining gap is the representation reaching the learner
(integration/readout), not the learner. Snake is also budget/learner limited even with
perfect input (GRU 12.0 at 131k vs heuristic 16.4). A linear policy cannot solve
Flappy/Pong even on perfect input, so linear results on the SNN partly reflect nonlinear
features computed by the substrate.

## 6. Failure states and ceilings (fresh diagnosis seeds, 100 episodes)

* **Catch**: v1.4 model misses concentrate on objects starting 3–4 lanes away
  (success 0.82–0.89 vs 0.96–1.0 for distance ≤ 2), which require a correct move on
  every step. Reference = 1.000: no metric ceiling below 1.
* **Dodge**: errors concentrate at distance 0–1 near edge lanes (player 0/4), where one
  escape direction is blocked. Random already scores 0.80, so absolute gains are
  compressed; reference = 1.000.
* **Snake**: reference controller 16.35 food (81% self-collision deaths), random 0.17.
  v1.4 selected model 11.8–12.3 with 92% self deaths. Self-collision limits both the
  heuristic and the learner — part of the remaining gap is task difficulty (G).
* **Pong**: v1.4 all-readout model hit rate 0.85–1.0 across serve distances, no single
  rare state dominates; the 4-tick descending model fails on far serves (0.05–0.2 hit at
  distance ≥ 0.8) — slow/imprecise tracking.
* **Flappy**: see §1 (pipe-alignment precision).

## 7. Implications for the v1.5 plan

1. Default integration 16 ticks; test 24 (and 32 for Flappy) — strongest, most general lever.
2. Headline readouts must **exclude the directly stimulated input neurons**
   (descending, nonvisual, pooled). all-2,040, visual-only and sensory-only run as
   pre-declared leakage controls and are not eligible for the headline selection.
3. Re-run the Linear / MLP / GRU comparison at the diagnosed integration time, since the
   v1.4 comparison was confounded by 4-tick representations.
4. Best-checkpoint selection on a dedicated seed block; longer budgets only where learning
   curves are still rising; capacity and step-size changes as bounded, conditional arms.
5. Curriculum is not supported by the diagnosis (v1.4 curriculum hurt; Flappy fails at
   the target task's pipe precision, not early boundary deaths) and is excluded.
6. State-balanced/targeted training for Dodge is not needed: 16-tick GRU already reaches
   ≈0.98–0.99 on diagnosis seeds.
7. Imitation warm-start remains a separately labelled rescue only.
