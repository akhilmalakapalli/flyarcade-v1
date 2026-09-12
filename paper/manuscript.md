# FlyArcade-v1: Learning Artificial Behaviors with a Drosophila Connectome-Constrained Spiking Network

**Status: complete.** Three studies, all preregistered and all reported in the
direction they came out. Study 1 (v1, frozen at commit `ec27ad0`) is a negative
result. Study 2 (v1.1) obtains genuine held-out learning on two lane tasks. Study 3
extends the identical architecture to Snake. Every number below is read from a
recorded result file; none is estimated or restated from memory.

## Abstract

We ask whether a spiking network whose recurrent connectivity is an authentic
subgraph of the HHMI Janelia MaleCNS v1.0 *Drosophila* connectome can learn
artificial arcade behaviours, and what actually limits it when it cannot.

We extract 2,040 traced neurons (218 visual-projection, 1,293 descending, 529
central-complex intrinsic) connected by 126,676 directed edges carrying 1,054,402
synapses, with no isolates. Every descending neuron lies within two directed hops of
visual-projection input. Sensory features are injected into visual-projection
neurons and an artificial decoder reads descending population rates.

**A naive reward-modulated plasticity rule failed.** Across three seeds, two tasks
and four conditions, held-out performance did not improve (mean paired change -0.014
catch, -0.017 dodge) and never exceeded a uniform-random policy. Post-hoc analysis
showed the circuit carried linearly decodable task information throughout (probe
0.654 against a 0.421 majority rate) while the policy collapsed onto a near-constant
action (entropy 1.098 → 0.361 and 0.128 nats). The limitation was in how reward was
linked to neural activity, not in the sensory representation.

**A reward-guided actor-critic learning mechanism, plus two feature-conditioning
fixes, produced genuine learning.** On confirmatory seeds never used for tuning,
catch rose from 0.206 to 0.661 against a 0.197 random baseline and dodge from 0.794
to 0.926 against 0.803, on 3/3 seeds each, with the policy remaining
state-dependent.

**The same architecture learned Snake**, a harder sequential task with terminal
failure, four actions and a growing body. Food eaten per episode rose from 0.100 to
1.483 against a 0.150 random baseline (contrast +1.333, interval [1.050, 1.625],
3/3 seeds), and survival from 5.0 to 29.9 steps. A fixed heuristic scores 17.258, so
the learned policy reaches only about 8% of the way from random to heuristic:
genuine learning, modest competence.

**Biological topology never outperformed a degree-matched rewired control** on any
of the three tasks. Scaled between each task's random and heuristic anchors, the
biological circuit scored 0.578 / 0.627 / 0.078 on catch / dodge / Snake against
0.772 / 0.683 / 0.104 for the rewired control. The rewired descending code is
higher-dimensional (participation ratio 54.5 vs 29.5) and less correlated (0.112 vs
0.152). A methodological caution comes with this: an earlier Snake run in which the
rewired arm appeared to fail completely was an artefact of standardising it with the
biological circuit's statistics, and vanished once each topology was calibrated by
the same procedure applied to its own activity.

**Once real learning existed, lesions became interpretable**, and their severity
tracked task difficulty. Under 10% neuron ablation the learned gain retained was
1.10 for dodge, 0.28 for catch and 0.12 for Snake; 25% ablation drove Snake below
its random baseline.

These are conclusions about a connectome-constrained computational model, not about
a trained fruit fly brain.

## 1. Introduction

Connectome-constrained modelling invites a natural hope: that wiring measured from a
real brain will, by itself, confer functional advantages. Testing that hope requires
a system in which a biological wiring diagram can be held fixed while everything
else — the learning rule, the readout, the task — is varied and controlled.

FlyArcade-v1 is such a system, deliberately small enough to run to completion on one
CPU. A fixed MaleCNS-derived subgraph provides the recurrent substrate. Artificial
but explicit interfaces provide sensing and action. Three tasks of increasing
difficulty provide behaviour. Against that fixed substrate we ran three studies.

The first failed, and the manner of its failure set up the rest: the network's
descending population demonstrably carried the information needed to act, yet the
learning rule drove the policy to a constant action regardless of input. Replacing
that rule — while changing nothing about the connectome, the membrane dynamics or
the sensory encoding — turned the failure into learning on all three tasks. The
control that was supposed to demonstrate the value of biological wiring instead
showed that a degree-matched random graph does at least as well.

We separate throughout what was measured from a connectome from what we engineered
on top of it, and we report each task's result on its own terms rather than forcing a
single conclusion across tasks that disagree.

## 2. The MaleCNS dataset and the extracted subgraph

### 2.1 What was measured

Source: HHMI Janelia MaleCNS v1.0, dataset `male-cns:v1.0` on
`https://neuprint.janelia.org`, accessed through the supported `neuprint-python`
client. 163 authenticated queries returned 29,506,842 bytes; every exact query,
response and checksum is recorded in `artifacts/malecns_manifest.json`.

Preserved verbatim: original neuPrint `bodyId` values, `status='Traced'`, superclass
and class annotations, soma side and ROI innervation, directed `ConnectsTo`
relationships with integer synapse counts, consensus neurotransmitter predictions
where annotated, and the upstream/downstream totals used for boundary accounting.

### 2.2 Selection

Selection was fixed on anatomical grounds alone, before any task outcome existed:

```
n.status='Traced' AND (n.`LAL(L)` OR n.`LAL(R)` OR n.GNG)
AND (n.superclass IN ['visual_projection','descending_neuron']
     OR (n.superclass='cb_intrinsic' AND n.class='CX'))
```

This yields **2,040 neurons** — 218 visual-projection, 1,293 descending, 529
central-complex intrinsic — joined by **126,676 directed edges** carrying
**1,054,402 synapses**, with **no isolates**. Autapses were removed and duplicate
pairs aggregated before any thresholding.

Consensus neurotransmitter annotations: acetylcholine 1,190, glutamate 393, GABA
343, unclear 65, dopamine 23, octopamine 18, serotonin 8.

### 2.3 Measured topology

**Every one of the 1,293 descending neurons is reachable from the visual-projection
population within two directed hops** — 735 at one hop, 558 at two. The
sensory-to-motor pathway the model needs is anatomically present, which is what makes
a behavioural negative result interpretable rather than vacuous
(`artifacts/connectome.png`, `artifacts/tables/connectome_summary.csv`).

### 2.4 Boundary truncation, the dominant limitation

The selected population retains only **15.05% of incoming** and **7.12% of outgoing**
synaptic weight relative to each neuron's whole-brain totals. Roughly 85% of the
drive these neurons receive in the animal is absent, replaced by a uniform artificial
background current of 0.18. This is a severely truncated circuit and a live candidate
explanation for any dynamical result reported here.

## 3. Neural model

**Engineered, not measured.** Discrete leaky integrate-and-fire neurons on sparse
edge arrays. Each neuron integrates
`v[t+1] = clip(0.85*v[t] + drive[t] + signed_sparse_input[t], -3, 4)` when not
refractory; threshold 1, reset 0, one-tick refractory period, one-tick transmission
delay. One action spans four ticks. These are dimensionless model parameters, not
fitted time constants.

Initial magnitude on edge i→j is `2.5*count(i,j)/sum_k count(k,j)`: anatomical counts
set *relative* strength and are **not** conductances. Consensus acetylcholine is
modelled as positive and GABA as negative; every other transmitter has zero fast
effect in the primary model, because receptor context is absent. Those neurons keep
their structural edges and annotations. Signs never change and never flip sign during
learning.

This membrane model is identical across all three studies. `tests/test_v11.py`
asserts that the v1.1 core reproduces the v1 core's spike trains bit-for-bit, so
every behavioural difference reported here is attributable to the learning
mechanism and the readout, never to the dynamics.

## 4. Learning mechanism

### 4.1 What v1 did

One global scalar advantage — reward minus an exponential moving baseline —
multiplied a symmetric spike-timing pair term on every plastic synapse, with no value
estimate, no discounting, no entropy term and no exploration floor. An artificial
softmax readout over descending rates was trained by the same scalar.

### 4.2 What v1.1 does

A reward-guided actor-critic mechanism that links neural activity to successful
actions locally rather than by one brain-wide scalar:

1. **Local eligibility traces.** Each synapse accumulates its own presynaptic trace
   gated by the postsynaptic surrogate derivative.
2. **A learned critic**: a linear value estimate over the same descending features.
3. **Temporal-difference reward prediction error**,
   `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`, computed once per action. The circuit
   is advanced exactly once per step; `V(s_{t+1})` is read when the next action's
   features are computed, never by re-simulating the network.
4. **Neuron-specific learning signals.** Each synapse is driven by the signal of its
   own postsynaptic neuron, obtained by projecting the actor's score through the
   readout weights.
5. **Separate actor and critic learning rates.**
6. **Stochastic softmax action selection** during training.
7. **Entropy regularisation**, applied without a trace so exploration pressure is
   immediate rather than credited to past actions.
8. **An exploration floor**: the sampling distribution is mixed with the uniform
   distribution, and learning uses the *exact* score function of that mixed
   distribution.
9. **Bounded weights and clipped updates** on actor, critic and recurrent core.
10. **Firing-rate monitoring** with explicit rejection criteria in every development
    run. No activity normalisation proved necessary; firing stayed near 0.21 spikes
    per neuron per tick in all three tasks.

### 4.3 Two feature-conditioning repairs

Both were found by diagnosing development failures, and both are frozen preprocessing
constants rather than searched parameters:

- **Common-mode removal.** Descending rates share a large state-independent
  component: mean absolute per-neuron mean 0.091 against a per-neuron standard
  deviation of 0.109. Feeding raw centred rates to a linear readout lets that
  state-independent direction dominate the update — precisely the mechanism of v1's
  collapse. Per-neuron statistics are estimated once, on development rollouts of the
  frozen core under fixed policy-independent behaviour, and frozen.
- **Readout scaling.** With 1,293 features, `||phi||^2` is of order 1,293, making
  every learning rate roughly a thousand times larger than it appears. In the first
  development run the critic saturated inside one episode (|delta| pinned at its clip
  of 10) and entropy fell to 0.055 nats before episode two. Dividing standardised
  features by `sqrt(width)` fixes this.

### 4.4 Staged plasticity

Three scopes were available: **A** readout only, recurrent core entirely frozen; **B**
additionally the 79,275 synapses onto descending neurons; **C** additionally the
central-complex intrinsic population, 123,434 synapses. **Stage A learned on every
task, so stages B and C were never used** and support no claim here. The recurrent
MaleCNS core is frozen in every result in this paper.

## 5. Task definitions

All three tasks are headless, seeded and deterministic given a seed.

**Catch and dodge.** Five lanes, a player starting in the middle, a falling object
whose lane is drawn uniformly. One episode is 24 actions: six objects, each landing
after four actions. Actions move left / stay / right. Catch succeeds by occupying the
object's lane at landing, dodge by avoiding it. Evaluation reports the fraction of
six landings that succeed. Dense shaping (0.1 times the change in absolute lane
distance) is used for training only and is deliberately **not** the evaluation
metric.

**Snake.** An 8×8 grid, snake starting at length 3, food placed uniformly on a free
cell, four absolute actions, an immediate 180° reversal refused rather than fatal.
Episodes end on wall collision, self collision, 200 steps, or 64 steps without food.
Reward is bounded and interpretable: **+1.0 for food, -1.0 for death, -0.01 per
step**. No distance-to-food shaping was needed, so none is used — the agent cannot
score without genuinely playing Snake.

Snake's observation is 21 normalised channels, injected by the same Bernoulli scheme
the lane tasks use: relative food direction (4), normalised food distance (1),
heading one-hot (4), immediate danger in each direction (4), and 3×3 local occupancy
around the head (8). No pixels and no CNN. `tests/test_snake.py` verifies that the
danger channels actually predict death, that food never lands on the body, that
reversal is refused, and that episodes always terminate.

## 6. Controls

Every confirmatory comparison uses matched seeds and identical training budgets.

- **Learning disabled** ("frozen"): the same circuit and readout, no updates.
- **Degree-preserving rewired topology**: in/out degree and outgoing anatomical
  strength preserved, incoming strength not preserved; 10 attempted swaps per edge.
  Identical learning rule, neuron count, readout architecture and budget.
- **Random policy**: uniform over the action set.
- **Heuristic**: a deterministic reference with the same observations. For catch and
  dodge it scores 1.000, establishing solvability. For Snake it greedily approaches
  food while refusing fatal moves, scoring 17.258 food — a reference, not an optimum.

**Each topology is standardised by the same procedure applied to its own activity.**
Reusing the biological circuit's statistics for the rewired arm would hand one arm a
calibrated readout and the other an uncalibrated one; §9.2 shows this is not a
hypothetical concern.

## 7. Study 1 — why the naive rule failed

### 7.1 The negative result

36 trials, three seeds, two tasks, four conditions
(`artifacts/tables/primary_heldout.csv`):

| Task | Condition | Before | After | Change | Bootstrap 95% |
|---|---|---|---|---|---|
| catch | learning | 0.189 | 0.175 | -0.014 | [-0.100, 0.108] |
| catch | frozen | 0.189 | 0.189 | 0.000 | [0.000, 0.000] |
| catch | rewired | 0.203 | 0.158 | -0.044 | [-0.125, 0.050] |
| catch | readout only | 0.189 | 0.164 | -0.025 | [-0.092, 0.067] |
| dodge | learning | 0.811 | 0.794 | -0.017 | [-0.058, 0.025] |
| dodge | frozen | 0.811 | 0.811 | 0.000 | [0.000, 0.000] |
| dodge | rewired | 0.797 | 0.819 | 0.022 | [-0.058, 0.108] |
| dodge | readout only | 0.811 | 0.833 | 0.022 | [-0.067, 0.083] |

Random baselines 0.194 and 0.806; heuristic 1.000. No condition improved and none
exceeded random. Weights did change substantially (recurrent L1 change ≈563, readout
≈48), so this was not a silent no-op.

### 7.2 The mechanism

On frozen saved weights, under a matched uniform-random behaviour policy so a
collapsed policy's degenerate state distribution cannot inflate the statistics
(`artifacts/policy_collapse.png`):

- **Information was present and survived training.** A shared-covariance linear probe
  recovered the required movement direction at 0.654 with initial weights and
  0.658 (catch) / 0.713 (dodge) after training, against a 0.421 majority-class rate
  and 0.383–0.425 shuffled controls.
- **The policy collapsed.** Mean action entropy fell from 1.098 nats — the uniform
  maximum is 1.099 — to 0.361 and 0.128, with the most likely action reaching
  probability 0.896 and 0.976.
- **Constant-action policies explain the scores exactly.** Always-left / always-stay
  / always-right score 0.208 / 0.233 / 0.133 on catch and 0.792 / 0.767 / 0.867 on
  dodge. The trained controllers' 0.175 and 0.794 fall inside those ranges.

The trained controller was, behaviourally, an observation-independent constant
action. The information was there; the rule did not link it to successful actions.

## 8. Study 2 — reward-guided learning on catch and dodge

### 8.1 Protocol

Development used seed blocks disjoint from everything v1 touched; confirmatory runs
used further blocks used for nothing else, asserted by test. The search was
coordinate-wise from a documented baseline — a full grid over seven axes is
unaffordable on CPU — with every configuration recorded including rejected ones in
`artifacts/v11_development/`. Rejection criteria were applied **before** any
performance comparison: entropy below 0.1 nats, state-dependence below 0.05, mean
|TD error| above 5, or firing outside [0.01, 0.6].

Because coordinate optima need not compose, the combined configuration was
re-validated on five development seeds, two never used by the sweeps, and compared
head to head with the runner-up. The two tied on performance (0.736 vs 0.741 averaged
over both tasks); the tie was broken on a pre-specified anti-collapse criterion — the
chosen configuration retains far more state-dependence on dodge (0.30 vs 0.12,
against a 0.05 rejection threshold) — not on score.

`experiments/v11_run_plan.json` then froze the hyperparameters, reward function,
plasticity mask, sensory encoding, motor decoding, training schedule and stopping
criterion, and stated the success criterion in advance.

### 8.2 Result

`artifacts/tables/v11_primary.csv`, Figure `artifacts/v11_confirmatory.png`:

| Task | Condition | Before | After | Change | Bootstrap 95% | State-dependence |
|---|---|---|---|---|---|---|
| catch | actor-critic | 0.206 | **0.661** | +0.456 | [0.388, 0.567] | 0.442 |
| catch | frozen | 0.206 | 0.206 | 0.000 | [0.000, 0.000] | 0.000 |
| catch | rewired | 0.206 | **0.817** | +0.611 | [0.508, 0.738] | 0.486 |
| dodge | actor-critic | 0.794 | **0.926** | +0.132 | [0.121, 0.154] | 0.402 |
| dodge | frozen | 0.794 | 0.794 | 0.000 | [0.000, 0.000] | 0.000 |
| dodge | rewired | 0.794 | **0.938** | +0.143 | [0.083, 0.200] | 0.418 |

Against random (0.197 and 0.803) the learned arm gains **+0.464** [0.408, 0.563] and
**+0.124** [0.100, 0.158], on 3/3 seeds. All three preregistered criteria are met and
the frozen arm reproduces its initial score exactly.

The frozen arm's greedy mode is degenerate by construction — zero-initialised actor
weights tie every logit — so the **random** policy is the informative chance
reference, and it is the one the success criterion is stated against.

## 9. Study 3 — Snake

### 9.1 Snake-specific changes, and why they were necessary

Snake reuses the MaleCNS subgraph, the LIF dynamics, the learning rule, common-mode
removal, `sqrt(N)` scaling, the guards, the deterministic seeding and the greedy
evaluation methodology unchanged. Four parameters differ, each documented in
`experiments/snake_run_plan.json` with the development measurement that forced it:

| Parameter | v1.1 | Snake | Reason |
|---|---|---|---|
| `actor_lr` | 0.4 | 8.0 | Snake's mean absolute TD error is ≈0.17 against 0.3–0.5 on the lane tasks, so identical rates give far smaller updates. Development food at `actor_lr` 0.4 / 1.5 / 4.0 / 8.0 was 0.35 / 0.65 / 1.35 / 1.95; at the v1.1 rate Snake did not learn at all. |
| `entropy_coefficient` | 0.03 | 0.003 | With TD errors near 0.06–0.17 the entropy gradient dominated the reward gradient and the policy never differentiated: entropy stayed at 1.22 of a 1.386 maximum. |
| `exploration_floor` | 0.1 | 0.02 | A forced random action in Snake is frequently fatal, so a 10% floor caps achievable survival. Catch and dodge have no terminal failure state. |
| `training_episodes` | 600 | 4000 | Snake episodes begin at about five steps, so a 600-episode budget supplies roughly a twentieth of the updates the lane tasks receive. |

`critic_lr`, `discount` and `trace_decay` are unchanged. These are genuine
task-specific incompatibilities of reward scale and episode structure, not tuning
toward a desired outcome: the first three were each shown to block learning
outright.

Because Snake episodes lengthen as the policy improves, a full 4,000-episode budget
does not fit one 120-second guarded process. Rather than raise the guard, trials
checkpoint at episode boundaries and **resume in a fresh independently guarded
process**, following the v1 pattern. No limit was raised.

### 9.2 A preprocessing artefact that looked like a topology effect

An initial Snake run standardised the rewired arm with the **biological** circuit's
statistics. Under that mismatch the rewired arm appeared to fail completely: 0.100 /
0.075 / 0.100 food and 5.0–7.0 steps, indistinguishable from no learning. Refitting
the standardisation per graph by the identical frozen procedure raised the same arm
to 1.875 / 1.200 / 2.725 food. **The entire apparent topology effect was a readout
calibration mismatch.** We report this because it is the most dangerous kind of error
available in this design, and because it would have supported exactly the conclusion
a connectome paper is tempted to draw.

### 9.3 Confirmatory result

`artifacts/tables/snake_primary.csv`, Figure `artifacts/snake_confirmatory.png`.
All three seeds are reported; none is excluded.

| Condition | Food before | Food after | Seeds | Steps | Return | State-dep. | AUC | Episodes to 1 food |
|---|---|---|---|---|---|---|---|---|
| actor-critic | 0.100 | **1.483** | 1.150, 1.750, 1.550 | 29.85 | +0.202 | 0.712 | 0.847 | 663, 633, 636 |
| frozen | 0.100 | 0.100 | 0.100, 0.100, 0.100 | 5.00 | -0.950 | 0.000 | 0.000 | never |
| rewired | 0.100 | **1.933** | 1.875, 1.200, 2.725 | 33.37 | +0.600 | 0.691 | 1.139 | 564, 1151, 502 |

Random 0.150 food; heuristic 17.258.

Contrasts (`artifacts/tables/snake_contrasts.csv`):

- actor-critic − frozen: **+1.383** [1.050, 1.650]
- actor-critic − random: **+1.333** [1.050, 1.625], positive on 3/3 seeds
- actor-critic − rewired: -0.450 [-1.175, 0.550], interval spanning zero

All three preregistered Snake criteria are met: food improved on every seed, exceeded
random by more than 0.5 on every seed, and the policy stayed state-dependent
(state-dependence 0.712; final action entropy 0.425 nats against a 1.386 maximum).

**Competence is modest and we state it plainly.** Scaled between random and
heuristic, the learned policy reaches 0.078 — about 8% of the way. The learning curve
was still rising at the frozen 4,000-episode budget (training food per 800-episode
block: 0.52, 0.78, 0.95, 0.90, 1.01 on development seed 0), so this is a budget-bound
result, not a plateau. We did not extend the budget after seeing confirmatory
results.

### 9.4 Demonstration

`scripts/snake_visualize.py` replays a recorded evaluation episode with learning
disabled, showing gameplay, the 21 sensory channels, descending population activity
and the policy with its value estimate in one figure
(`artifacts/snake_demo_frame.png`), and writes an animated GIF of the full episode
(`artifacts/snake_demo.gif`). The recorded episode is the **median**-performing of
the 40 evaluation episodes, not the best; its rank and seed are stored in
`artifacts/snake_demo.json`.

## 10. Topology comparison

With identical learning rule, neuron count, readout architecture, training budget and
per-graph standardisation procedure:

| Task | Biological | Rewired | Contrast | Interval |
|---|---|---|---|---|
| catch | 0.661 | 0.817 | -0.156 | [-0.325, -0.021] |
| dodge | 0.926 | 0.938 | -0.011 | [-0.079, 0.071] |
| Snake (food) | 1.483 | 1.933 | -0.450 | [-1.175, 0.550] |

**Biological topology never exceeded the rewired control on any task.** Only catch's
interval excludes zero; dodge and Snake are statistically indistinguishable. We do
not claim a uniform effect where two of three tasks cannot resolve one.

A post-hoc analysis of the descending representation, on frozen cores under matched
uniform-random behaviour (`artifacts/v11_topology_diagnostic.json`), explains the
direction:

| Property | Biological | Rewired |
|---|---|---|
| Participation ratio (effective dimensions) | 29.5 | **54.5** |
| Mean absolute pairwise correlation | 0.152 | **0.112** |
| Linear probe accuracy (majority 0.431) | 0.774 | **0.901** |

Degree-preserving rewiring decorrelates the descending population and nearly doubles
the number of directions carrying variance, giving a linear readout more independent
features. The biological wiring concentrates descending activity into a
lower-dimensional, more correlated code.

**This is not a claim that the biology is worse.** The readout is an artificial
linear decoder on a circuit missing 85% of its incoming synaptic weight, and
correlated population structure may serve functions these tasks cannot see —
robustness, multiplexing of other behaviours, metabolic economy. The rewired control
also preserves in/out degree and outgoing strength but *not* incoming strength, and
its mixing is unvalidated, so it does not isolate a single graph property. What the
experiments establish is narrow: for these tasks, this readout and this learning
rule, connectome-derived topology is not what makes learning possible.

## 11. Perturbation and lesion analysis

Lesions were run only after state-dependent learning existed. In v1 the same
perturbations changed nothing, and that was correctly refused as evidence of
robustness: a policy that ignores its observations cannot be degraded by ablating
them. All perturbations below use frozen learned weights and masks sampled once per
trial.

**Catch and dodge** (`artifacts/tables/v11_robustness.csv`):

| Task | Intact | Sensory noise sd 0.1 | 10% edge ablation | 10% neuron ablation |
|---|---|---|---|---|
| catch | 0.661 | 0.453 | 0.617 | **0.329** |
| dodge | 0.926 | 0.914 | 0.924 | 0.939 |

**Snake** (`artifacts/tables/snake_robustness.csv`, Figure
`artifacts/snake_robustness.png`):

| Perturbation | Food | Steps |
|---|---|---|
| intact | 1.483 | 29.85 |
| sensory noise sd 0.1 | 1.242 | 27.93 |
| 10% edge ablation | 0.417 | 15.07 |
| 5% neuron ablation | 0.442 | 13.94 |
| 10% neuron ablation | 0.308 | 12.00 |
| 25% neuron ablation | 0.100 | 7.62 |

Snake shows a clean graded dose-response to neuron ablation, in both food and
survival, down to its random baseline at 25%. Dodge is unaffected because it sits
near its ceiling and a mostly-correct policy still avoids a single object; we do not
call that robustness.

## 12. Cross-task comparison

`artifacts/tables/cross_task.csv`, Figure `artifacts/cross_task.png`. Metrics differ
by task, so comparisons are scaled between each task's own random (0) and heuristic
(1) anchors.

| Task | Metric | Actions | Episodes | Random | After | Rewired | Heuristic | Normalised (bio / rewired) | Firing rate |
|---|---|---|---|---|---|---|---|---|---|
| catch | landing success | 3 | 600 | 0.197 | 0.661 | 0.817 | 1.000 | 0.578 / 0.772 | 0.2167 |
| dodge | landing success | 3 | 600 | 0.803 | 0.926 | 0.938 | 1.000 | 0.627 / 0.683 | 0.2173 |
| Snake | food eaten | 4 | 4000 | 0.150 | 1.483 | 1.933 | 17.258 | 0.078 / 0.104 | 0.2085 |

**Does one MaleCNS-derived substrate support learning across all three tasks?** Yes.
The identical frozen recurrent core, with only the readout and the learning
mechanism adapting, beats its random baseline on all three, on every seed.

**Does biological versus rewired behave consistently?** The *direction* is consistent
— biological never wins — but the magnitude is not, and only catch resolves the
difference from zero. We decline to state a single topology conclusion.

**Are harder sequential tasks more lesion-sensitive?** Yes, clearly. Fraction of the
learned gain retained under 10% neuron ablation: dodge 1.10, catch 0.28, Snake 0.12.
Snake also loses most of its gain to 10% edge ablation (0.20 retained) where catch
loses little (0.90). Tasks requiring sustained sequential control over long episodes
depend on more of the circuit than a four-step reactive decision does.

**Does the learning architecture generalise?** Yes, with documented reparameterisation:
the rule, preprocessing, staged scope and evaluation methodology carried over
unchanged, while four parameters had to be rescaled for Snake's sparser reward and
terminal-failure structure (§9.1).

**Do activity statistics differ across tasks?** Remarkably little. Final firing rates
are 0.2167, 0.2173 and 0.2085 spikes per neuron per tick — the recurrent core
operates in the same regime regardless of task, consistent with stage A leaving it
frozen.

## 13. Limitations

1. **Boundary truncation.** 85% of incoming and 93% of outgoing synaptic weight lies
   outside the selected population, replaced by a constant background current. This
   is the largest threat to any dynamical conclusion here.
2. **Three seeds.** All intervals are descriptive bootstraps over three training
   seeds. They cannot exclude small effects and support no confirmatory significance
   claim.
3. **Sign simplification.** Only acetylcholine and GABA have fast effects; 393
   glutamatergic, 23 dopaminergic, 18 octopaminergic, 8 serotonergic and 65 unclear
   neurons are structurally present but functionally silent.
4. **The recurrent core never learned.** Every positive result uses stage A, with the
   connectome-derived weights frozen. The learning lives in an artificial readout that
   reads a connectome-shaped representation.
5. **Artificial interfaces.** Sensory encoding and motor decoding are engineered and
   carry no biological interpretation. There is no claim of retinotopy, and descending
   neurons are treated as command outputs with no muscles or ventral nerve cord.
6. **The rewired control is imperfect.** It does not preserve incoming strength and
   its mixing is unvalidated, bounding the topology reading.
7. **Snake competence is modest and budget-bound**, at 0.078 of the way from random to
   heuristic, with the learning curve still rising at the frozen budget.
8. **Task scale.** Two five-lane reactive tasks and one 8×8 Snake. These are not fly
   behaviours.
9. **Not a fly.** Every conclusion concerns a connectome-constrained computational
   model. Nothing here is evidence about how *Drosophila* learns, nor that e-prop-style
   reward-guided plasticity is biologically implemented in the fly.

## 14. Reproducibility

- **v1**: all 36 trials replay bit-identically from saved checkpoints across 144
  evaluation sets (`artifacts/replay_check.json`); the scientific audit hash-verifies
  every result file and confirms seed disjointness and frozen-condition invariance.
- **v1.1**: determinism confirmed by deleting `runs/v11-*` and re-running the full
  confirmatory suite — every number reproduced exactly. The later encoder refactor was
  verified bit-for-bit against the stored results.
- **Snake**: the eprop and frozen arms were re-run against the final frozen plan and
  reproduced identically, verified field by field.
- Every trial ran under unchanged cooperative guards: 4,096 neurons, 250,000 edges,
  2 GiB RSS, ≥1 GiB free disk, 120 s per independently guarded process. Snake trials
  checkpoint and resume in fresh processes rather than raising any limit. **No
  resource limit was raised at any point in any study.**
- Seed blocks are disjoint by construction and asserted by tests: v1 (1,000 /
  3,000,000 / 1,900,000 / 2,000,000), v1.1 (4,000,000 / 4,200,000 / 4,300,000 /
  4,500,000 / 5,000,000 / 6,000,000), Snake (7,000,000 / 7,200,000 / 7,500,000 /
  8,000,000 / 9,000,000).

```sh
sh scripts/verify.sh
.venv/bin/python scripts/replay_all.py                 # v1: 36/36 MATCH
.venv/bin/python scripts/scientific_audit.py           # v1: PASS
.venv/bin/python scripts/v11_summarize.py
.venv/bin/python scripts/snake_run_suite.py            # skips completed trials
.venv/bin/python scripts/snake_summarize.py
.venv/bin/python scripts/cross_task_analysis.py
.venv/bin/python scripts/snake_visualize.py --trial runs/snake-eprop-1 --gif
```

## 15. Data and code availability

Connectivity data: HHMI Janelia MaleCNS v1.0, [project site](https://male-cns.janelia.org/),
[release notes](https://male-cns.janelia.org/release/) and
[programmatic access instructions](https://male-cns.janelia.org/download/), used under
the terms linked from that site (CC-BY); verify attribution against the response
bundle recorded in `artifacts/malecns_manifest.json`. Access used
[neuprint-python](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html).

Method background: Florian, R. V. (2007),
[Reinforcement learning through modulation of spike-timing-dependent synaptic plasticity](https://pubmed.ncbi.nlm.nih.gov/17444757/);
Bellec et al. (2020), [A solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y).

Code is in this repository. Raw connectivity bundles, per-trial run directories and
checkpoints live in Git-ignored `data/` and `runs/`; compact checksummed manifests,
per-seed metrics, all tables and all figures are tracked under `artifacts/`. The
preregistrations are `experiments/run_plan.json`, `experiments/v11_run_plan.json` and
`experiments/snake_run_plan.json`. No credentials are stored in the repository;
`.secrets/` is Git-ignored.
