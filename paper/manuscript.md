# Credit assignment, not connectivity, gates learning in a MaleCNS-derived visuomotor controller

**Status: two completed studies.** v1 (frozen at commit `ec27ad0`) is a preregistered
negative result: reward-modulated STDP produced no held-out learning and collapsed
the policy onto a constant action. v1.1 replaces the learning rule with an
actor-critic e-prop three-factor rule over the *same* connectome, the same circuit
and the same encoding, and obtains a confirmatory positive result on seeds never
used for tuning. Neither study was retuned against its own held-out data.

## Abstract

We asked whether a spiking controller whose recurrent connectivity is an authentic
subgraph of the HHMI Janelia MaleCNS v1.0 connectome (2,040 traced neurons, 126,676
directed edges, no isolates) can learn two minimal lane arcade tasks, and if not,
what the binding constraint is.

**Study 1 (v1).** Under global reward-modulated STDP with a scalar advantage, the
controller did not improve on either task, in any of four conditions, on any of
three seeds (mean paired held-out change -0.014 catch, -0.017 dodge; all intervals
spanning zero), and never exceeded a uniform-random policy. Post-hoc diagnostics
showed the descending population carried linearly decodable task information the
whole time (probe 0.654 against a 0.421 majority rate) while the policy collapsed
to a near-constant action (entropy 1.098 → 0.361 / 0.128 nats). The failure was
credit assignment, not sensory representation.

**Study 2 (v1.1).** Replacing only the learning rule — per-synapse eligibility
traces, a learned critic, a temporal-difference error, neuron-specific learning
signals, entropy regularisation and an exploration floor — produces learning on
confirmatory seeds that were never used for tuning. Catch rises from 0.206 to
**0.661** against a 0.197 random baseline (+0.464 over random, 3/3 seeds), dodge
from 0.794 to **0.926** against 0.803 (+0.124, 3/3 seeds), and the policy stays
state-dependent rather than collapsing. Three separate repairs were each necessary:
removing the descending population's common mode, scaling the readout so 1,293
features do not make every learning rate three orders of magnitude too large, and
maintaining explicit exploration pressure.

**The biological topology is not what makes the task learnable.** Under an identical
rule, budget, neuron count and readout, a degree-preserving rewired graph learned
catch *better* than the biological one (0.817 vs 0.661; paired difference -0.156,
interval [-0.325, -0.021]) and was indistinguishable on dodge. A post-hoc analysis
explains it: the rewired circuit's descending representation has nearly twice the
effective dimensionality (participation ratio 54.5 vs 29.5) and lower pairwise
correlation (0.112 vs 0.152), so a linear readout has more independent directions to
exploit, and a linear probe decodes the required action better (0.901 vs 0.774).

The combined result is narrow but clean: **the connectome-derived circuit contains
task-relevant information, an appropriate credit-assignment mechanism is required to
turn that information into behaviour, and the biological wiring itself confers no
advantage over a degree-matched random graph for this task and this readout.**

---

# Part I — v1: the negative baseline

## 1. What is biological measurement and what is engineering

This separation is the paper's central epistemic commitment, and we state it
before any result.

**Measured from MaleCNS v1.0** (`male-cns:v1.0` on `https://neuprint.janelia.org`,
163 authenticated queries, 29,506,842 bytes, exact responses checksummed in
`artifacts/malecns_manifest.json`):

- Neuron identity: original neuPrint `bodyId` values, `status='Traced'`.
- Superclass and class annotations, soma side, ROI innervation.
- Directed `ConnectsTo` relationships and their integer synapse counts.
- Consensus neurotransmitter predictions where annotated.
- Upstream/downstream synaptic totals used for boundary accounting.

**Engineered by us, and not biological claims**:

- The leaky integrate-and-fire dynamics, thresholds, refractory period and
  one-tick transmission delay are dimensionless model choices, not fitted
  physiology.
- Anatomical synapse counts set relative weight magnitudes. They are not
  conductances.
- The sign rule (acetylcholine excitatory, GABA inhibitory, everything else zero
  fast effect) is a deliberate conservative simplification. It omits
  receptor-dependent glutamatergic effects and all neuromodulation. The 393
  glutamatergic, 23 dopaminergic, 18 octopaminergic, 8 serotonergic and 65
  unclear neurons retain their structural edges and annotations but exert no fast
  synaptic effect in the primary model.
- The eight engineered feature channels, their Bernoulli injection into
  visual-projection neurons, and the electrode assignment by sorted body ID are
  artificial interfaces. There is no claim of retinotopy.
- The three-action softmax decoder over descending spike rates is an artificial
  readout, not a reconstruction of motor control. Descending neurons are treated
  as command outputs; no muscles or ventral nerve cord are modelled.
- The reward-modulated eligibility rule is an engineering implementation inspired
  by reward-modulated STDP ([Florian, 2007](https://pubmed.ncbi.nlm.nih.gov/17444757/)),
  not a reproduction of a published fly circuit.

**Boundary loss is the dominant limitation and we do not minimise it.** The
selected population retains only **15.05% of incoming** and **7.12% of outgoing**
synaptic weight relative to each neuron's whole-brain totals. Roughly 85% of the
synaptic drive these neurons receive in the animal is absent and replaced by a
uniform artificial background current of 0.18. This is a severely truncated
circuit, not a whole-animal simulation, and it is a live candidate explanation for
any negative dynamical result reported here.

## 2. Circuit selection and measured topology

Selection was fixed on anatomical grounds alone, before any game outcome existed,
and is recorded verbatim in `artifacts/malecns_manifest.json`:

```
n.status='Traced' AND (n.`LAL(L)` OR n.`LAL(R)` OR n.GNG)
AND (n.superclass IN ['visual_projection','descending_neuron']
     OR (n.superclass='cb_intrinsic' AND n.class='CX'))
```

This yields 2,040 neurons: **218 visual-projection**, **1,293 descending** and
**529 central-complex intrinsic** (Figure `artifacts/connectome.png`,
`artifacts/tables/connectome_summary.csv`). Autapses were removed and duplicate
pairs aggregated before any thresholding; selected isolates would have been
retained, and there are none.

Measured connectivity result: **every one of the 1,293 descending neurons is
reachable from the visual-projection population within two directed hops** — 735
at one hop, 558 at two. The sensory-to-motor pathway the model needs is
anatomically present. This is a fact about the acquired subgraph, and it is what
makes the behavioural negative result interpretable rather than vacuous.

## 3. Tasks, protocol and preregistration

Two headless five-lane tasks, catch and dodge, each 24 actions per episode with
six falling objects. Evaluation reports the fraction of six landings that succeed.
Dense reward shaping (0.1 times the change in absolute lane distance) is used for
training only and is deliberately *not* the evaluation metric. The controller
receives no task identity bit.

The full protocol was frozen in `experiments/run_plan.json` before any held-out
run: three model seeds (0, 1, 2), two tasks, four conditions, 200 training
episodes, 20 shared held-out evaluation seeds (`3000000 + 100*seed + i`) disjoint
from all training seeds (`1000 + 10000*seed + episode`). Two readout designs were
compared during development on seed 99 and separate environment seeds only; the
per-neuron design was selected there and never revisited. No held-out retuning
was performed at any point.

Four conditions: **learning** (both mechanisms plastic); **frozen** (both frozen);
**rewired** (degree-preserving swapped graph, both mechanisms plastic);
**readout_only** (recurrent weights frozen). Rewiring preserves in/out degree and
outgoing anatomical strength but not incoming strength, and recomputed incoming
normalisation and source signs can alter effective dynamics — so this control does
not isolate a single graph property, and we do not claim uniform sampling or
adequate mixing (1,086,365 of 1,266,760 attempted swaps accepted).

## 4. Primary result: no held-out learning

36 trials completed (`artifacts/tables/trial_index.csv`). Mean paired held-out
change, aggregated at the independent training-seed level
(`artifacts/tables/primary_heldout.csv`, Figure `artifacts/heldout_results.png`):

| Task | Condition | Before | After | Change | Bootstrap 95% |
|---|---|---|---|---|---|
| catch | learning | 0.189 | 0.175 | **-0.014** | [-0.100, 0.108] |
| catch | frozen | 0.189 | 0.189 | 0.000 | [0.000, 0.000] |
| catch | rewired | 0.203 | 0.158 | -0.044 | [-0.125, 0.050] |
| catch | readout_only | 0.189 | 0.164 | -0.025 | [-0.092, 0.067] |
| dodge | learning | 0.811 | 0.794 | **-0.017** | [-0.058, 0.025] |
| dodge | frozen | 0.811 | 0.811 | 0.000 | [0.000, 0.000] |
| dodge | rewired | 0.797 | 0.819 | 0.022 | [-0.058, 0.108] |
| dodge | readout_only | 0.811 | 0.833 | 0.022 | [-0.067, 0.083] |

Random baselines are 0.194 (catch) and 0.806 (dodge); the deterministic heuristic
scores 1.000 on both. **No condition improved, and no trained condition exceeded
the random baseline.** The frozen control reproduced its pre-training score
exactly in all six trials, confirming that evaluation itself is free of drift and
that the two "before" and "after" passes are genuinely matched.

Weights did change substantially — recurrent L1 change ≈563 and readout L1 change
≈48 in a representative learning trial — so this is not a silent no-op. The
plasticity ran; it simply produced no held-out benefit.

Control contrasts (`artifacts/tables/contrasts.csv`) are all small with intervals
spanning zero: learning minus frozen -0.014 (catch) and -0.017 (dodge); learning
minus rewired +0.017 and -0.025; learning minus readout_only +0.011 and -0.039.
**We find no evidence that biological topology helped, and none that recurrent
plasticity added anything over readout-only learning.** With three seeds these
intervals are descriptive; they cannot exclude small effects, and we make no
confirmatory claim in either direction.

## 5. Why it fails: policy collapse with intact sensory information

The primary result alone cannot distinguish "the circuit never delivered usable
information" from "the learning rule failed to exploit available information". We
separated these post hoc, on frozen saved weights, with probe decoders that are
analyst instruments and were never part of any controller or any model selection
(`scripts/diagnostics.py`, Figure `artifacts/policy_collapse.png`,
`artifacts/tables/diagnostics.csv`). To avoid the confound that a collapsed policy
visits a degenerate state distribution, all comparisons below use a **matched**
uniform-random behaviour policy, identical across conditions.

**Task information is present and survives training.** The probe is a
shared-covariance linear discriminant fitted on one half of the decisions and
scored on the other. With initial weights it recovers the required movement
direction at 0.654 accuracy, against a 0.421 majority-class rate and a 0.383
shuffled-label control. After training it still reads 0.658 (catch) and 0.713
(dodge), with shuffled controls at 0.425 and 0.358. Recurrent plasticity neither
created nor destroyed the signal. Because the probe is linear, these figures are
a lower bound on the information actually present in the descending population.

**The policy collapses.** Mean action entropy falls from 1.098 nats — statistically
indistinguishable from the 1.099 uniform maximum — to 0.361 (catch) and 0.128
(dodge). The most likely action's mean probability rises from 0.347 to 0.896 and
0.976. Under its own policy the trained catch controller emits "right" on 426 of
480 decisions, and the trained dodge controller emits "left" on 468 of 480.

**The collapsed policies explain the scores exactly.** Degenerate constant-action
policies evaluated on the same held-out seeds
(`artifacts/tables/constant_action_reference.csv`) score catch 0.208 / 0.233 /
0.133 and dodge 0.792 / 0.767 / 0.867 for always-left / always-stay / always-right.
The trained controllers' 0.175 and 0.794 fall inside these ranges. A five-lane
task with a uniformly drawn object rewards a wall-hugging constant action at
roughly 1/5 for catch and 4/5 for dodge, which is also why the random baseline sits
where it does. The trained controller is, behaviourally, an observation-independent
constant action.

This is the paper's substantive finding: **in this configuration, reward-modulated
eligibility learning under dense shaping converges on a degenerate deterministic
policy before it exploits the task-relevant signal that is demonstrably available
in the descending population.** The bottleneck is credit assignment and the loss of
exploration, not the connectome-derived front end.

## 6. Robustness and lesions

Frozen final weights evaluated under Gaussian observation noise (sd 0.1), 10% edge
ablation and 10% neuron ablation, with masks sampled once per trial and held fixed
(`artifacts/tables/robustness.csv`, Figure `artifacts/robustness.png`):

| Task | Unperturbed | Sensory noise | Edge ablation | Neuron ablation |
|---|---|---|---|---|
| catch | 0.175 | 0.172 | 0.178 | 0.178 |
| dodge | 0.794 | 0.794 | 0.794 | 0.792 |

Every perturbation leaves performance unchanged to within 0.006. **This must not be
reported as robustness of a learned solution.** Ablating 10% of edges or neurons —
including sensory and output neurons — cannot degrade a policy that already ignores
its observations. The correct reading is that these lesions confirm the behavioural
degeneracy established in Section 5. A lesion experiment is only informative about a
circuit that is doing something, and this one is not.

## 7. Transfer

Equal-budget comparisons in both directions (`artifacts/tables/transfer.csv`,
Figure `artifacts/transfer.png`): 100 source + 100 target episodes, versus 200
target-only episodes (equal total budget) and 100 target-only episodes (equal
target exposure), evaluated on matched held-out target environments.

| Direction | 100 src + 100 tgt | 200 target | 100 target | Paired vs 200 target |
|---|---|---|---|---|
| dodge → catch | 0.200 | 0.175 | 0.183 | +0.025 |
| catch → dodge | 0.819 | 0.794 | 0.800 | +0.025 |

Both directions show a +0.025 mean advantage for the transfer arm, with three-seed
intervals spanning zero. Given that no arm learned anything and all arms sit at the
random baseline, we read this as noise, not as positive transfer. Reporting it as
transfer would require a source task that was learned in the first place.

## 8. Reproducibility

- `scripts/replay_all.py`: all **36 trials** restored from their saved final
  checkpoints and re-evaluated in separate guarded processes reproduced their
  held-out, sensory-noise, edge-ablation and neuron-ablation results
  **bit-identically** — 144 evaluation sets, zero mismatches
  (`artifacts/replay_check.json`). Pre-training evaluations predate the surviving
  checkpoint and are not replayable from it; we do not claim otherwise.
- `scripts/scientific_audit.py`: 36 result files hash-verified against the summary;
  training/evaluation seed disjointness, history lengths, frozen-condition
  invariance and readout-only recurrent invariance all confirmed
  (`artifacts/scientific_audit.json`).
- 56 tests pass; lint, format and dependency checks pass.
- All 36 trials ran under the unchanged cooperative guards — 4,096 neurons,
  250,000 edges, 2 GiB RSS, 120 s per trial. Peak observed: 28.7 s and 183 MiB for
  a single trial, 632 s total. **No resource limit was raised at any point**,
  including to complete runs.

Verification:

```sh
sh scripts/verify.sh
.venv/bin/python scripts/summarize_results.py
.venv/bin/python scripts/export_tables.py
.venv/bin/python scripts/replay_all.py
.venv/bin/python scripts/scientific_audit.py
.venv/bin/python scripts/diagnostics.py --task catch
.venv/bin/python scripts/diagnostics.py --task dodge
```

## 9. Limitations

1. **Boundary truncation.** 85% of incoming and 93% of outgoing synaptic weight is
   outside the selected population, replaced by a constant background current. This
   is the single largest threat to any dynamical conclusion here.
2. **Three seeds.** All intervals are descriptive bootstraps over three training
   seeds. They cannot exclude small effects and support no confirmatory claim.
3. **Sign simplification.** Only acetylcholine and GABA have fast effects.
   Glutamatergic, dopaminergic, octopaminergic, serotonergic and unclear neurons
   are structurally present but functionally silent, which plausibly suppresses
   real circuit dynamics.
4. **One learning rule, one hyperparameter setting.** The negative result concerns
   *this* rule at *these* settings. We deliberately did not search for a
   configuration that works, because held-out results were frozen against retuning.
   A negative result under a preregistered protocol is not evidence that no
   configuration could learn these tasks.
5. **Artificial interfaces.** Encoder and decoder are engineered and carry no
   biological interpretation. Any learning observed in this system could occur in
   the readout alone and must never be attributed to recurrent plasticity.
6. **The rewired control is imperfect.** It does not preserve incoming strength and
   its mixing is unvalidated, so "topology did not matter" is bounded by what that
   control actually holds fixed.
7. **Task scale.** Two minimal five-lane tasks with a three-action space. They are
   not fly behaviour and not a general benchmark.

## 10. What this study does and does not support

**Supported.** An authenticated MaleCNS v1.0 subgraph can be acquired under strict
resource bounds with full provenance, and used as the recurrent substrate of a
CPU-only spiking controller that runs a complete preregistered experiment in ten
minutes and replays bit-identically. Within this subgraph, all 1,293 descending
neurons lie within two directed hops of visual-projection input, and descending
population rates carry linearly decodable task information. Under the preregistered
rule, reward-modulated plasticity produced no held-out improvement on either task,
in any condition, on any seed, and converged instead on near-deterministic
observation-independent policies.

**Not supported.** That fly connectome topology helps or hinders learning. That
reward-modulated STDP can or cannot learn these tasks in general. That this circuit
is robust to lesions. That any transfer occurred. That any of this reflects how
Drosophila actually performs visuomotor control.

The honest summary is narrow and, we think, still worth recording: *the biology we
could measure delivered a usable signal; the learning rule we engineered on top of
it threw that signal away.*

## Data references

- HHMI Janelia and collaborators. [MaleCNS project](https://male-cns.janelia.org/)
  and [v1.0 release notes](https://male-cns.janelia.org/release/).
- [Official programmatic access instructions](https://male-cns.janelia.org/download/).
- [neuprint-python authorization and quickstart](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html).
- Florian, R. V. (2007). [Reinforcement learning through modulation of
  spike-timing-dependent synaptic plasticity](https://pubmed.ncbi.nlm.nih.gov/17444757/).

Connectivity data are used under the terms linked from the MaleCNS site (CC-BY);
verify attribution against the response bundle recorded in
`artifacts/malecns_manifest.json`.

---

# Part II — v1.1: repairing credit assignment

Everything in Part I is preserved unchanged as the negative baseline. v1.1 alters
the learning rule and the readout conditioning only. The connectome, the selected
2,040-neuron subgraph, the body IDs, the LIF membrane dynamics, the transmitter sign
rule, the sensory encoding, the tasks and the reward function are all identical, and
`tests/test_v11.py` asserts that a frozen v1.1 core reproduces v1's spike trains
bit-for-bit.

## 11. What changed, and why each change was needed

v1's rule updated every plastic synapse by one global scalar advantage times a
symmetric STDP pair term, with no value estimate, no discounting, no entropy term
and no exploration floor. v1.1 replaces it with an actor-critic e-prop three-factor
rule:

1. **Local eligibility traces.** Each synapse accumulates its own presynaptic trace
   gated by the postsynaptic surrogate derivative.
2. **A learned critic.** A linear value estimate over the same descending features.
3. **Temporal-difference error.** `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`,
   computed once per action; the circuit is advanced exactly once per step.
4. **Neuron-specific learning signals.** Each synapse is driven by the signal of its
   own postsynaptic neuron, obtained by projecting the actor's score through the
   readout weights, never by one scalar shared across the population.
5. **Separate actor and critic learning rates.**
6. **Stochastic softmax action selection** during training.
7. **Entropy regularisation**, applied without a trace so exploration pressure is
   immediate.
8. **An exploration floor**: the sampling distribution is mixed with the uniform
   distribution, and the score function used for learning is the exact score of that
   mixed distribution.
9. **Bounded weights and clipped updates** on the actor, the critic and the core.
10. **Firing-rate monitoring** in every development run, with explicit rejection
    criteria. No activity normalisation proved necessary; firing stayed near 0.2
    spikes per neuron per tick throughout.

Two further repairs turned out to be as important as the rule itself, and both were
found by diagnosing failures on development seeds rather than by tuning:

- **Common-mode removal.** Descending rates share a large state-independent
  component: the mean absolute per-neuron mean is 0.091 against a per-neuron
  standard deviation of 0.109. Feeding raw centred rates to a linear readout lets
  that state-independent direction dominate the policy gradient, which is precisely
  the mechanism of v1's collapse. Per-neuron standardisation is estimated once, on
  development rollouts of the frozen core under a uniform-random behaviour policy,
  and frozen.
- **Readout scaling.** With 1,293 features, `||phi||^2` is of order 1,293, so every
  learning rate is effectively about a thousand times larger than it appears. In the
  first development run the critic saturated within a single episode (|delta| pinned
  at its clip of 10) and entropy fell to 0.055 nats before episode two. Dividing the
  standardised features by `sqrt(width)` fixes this; it is a fixed preprocessing
  constant, not a tuned parameter.

## 12. Development protocol

Development used seed blocks disjoint from everything v1 touched and from the v1.1
confirmatory blocks: training in the 4,000,000 block, development evaluation in the
4,500,000 block, feature fitting in the 4,200,000 block. `tests/test_v11.py` asserts
the disjointness. **No v1 held-out seed was used at any point in v1.1 development.**

A full grid over the seven requested axes is unaffordable on CPU, so the search was
coordinate-wise from a documented baseline: sweep one axis at a time across three
development seeds, 600 training episodes per trial, and keep a change only if it
improved consistently. Every configuration is recorded in
`artifacts/v11_development/`, including rejected ones.

Rejection criteria were applied *before* any performance comparison: final action
entropy below 0.1 nats, a state-dependence below 0.05 (one action dominating
irrespective of state), a mean absolute TD error above 5 (critic divergence), or a
firing rate outside [0.01, 0.6]. Every surviving configuration beat the random
policy on 3/3 development seeds.

The staged plasticity strategy was followed in order. **Stage A — readout-only, with
the recurrent MaleCNS core entirely frozen — learned, so stages B and C were not
used.** The protocol's instruction is to prefer the simplest model that genuinely
learns, and stage A qualifies. Stage B (plasticity on the 79,275 synapses
terminating on descending neurons) and stage C (additionally the central-complex
intrinsic population, 123,434 synapses) are implemented and tested, but were not
needed and are not part of any claim here.

Coordinate-wise optima were `actor_lr` 0.4, `critic_lr` 0.2, `discount` 0.9,
`trace_decay` 0.7, `entropy_coefficient` 0.03, `exploration_floor` 0.1. Because
coordinate optima need not compose, the combined configuration was re-validated on
five development seeds — two of which the sweeps never used — and compared head to
head against the runner-up (`entropy_coefficient` 0.01, `exploration_floor` 0.05).
The two were tied on performance (mean over both tasks 0.736 vs 0.741). The tie was
broken on a pre-specified criterion rather than on score: the chosen configuration
retains far more state-dependence on dodge (0.30 vs 0.12, where 0.05 is the
rejection threshold) and higher action entropy (0.90 vs 0.51). Since the entire
point of v1.1 is to prevent collapse, the larger anti-collapse margin wins.

## 13. Freezing and the confirmatory design

Before any confirmatory run, `experiments/v11_run_plan.json` froze the
hyperparameters, the reward function, the plasticity mask, the sensory encoding, the
motor decoding, the training schedule and the stopping criterion, and stated the
success criterion in advance:

1. training success improves from the first to the last fifth of training;
2. held-out greedy success exceeds the random policy by more than 0.05 on every
   seed;
3. the policy stays state-dependent — state-dependence above 0.05 and final action
   entropy above 0.1.

Confirmatory seeds are in blocks never used for tuning: training
`5,000,000 + 10,000*seed + episode`, evaluation `6,000,000 + 100*seed + i` for
i = 0..39. Evaluation takes the greedy mode of the learned policy with all learning
disabled. Conditions: the biological circuit with e-prop, the same circuit with
learning disabled, the degree-preserving rewired circuit with the identical rule and
budget, plus the random and heuristic reference policies.

The rewired arm is standardised by the same procedure applied to *its own* rates.
Reusing the biological graph's statistics would have handed the biological arm a
calibrated readout and the control an uncalibrated one.

## 14. Confirmatory result: the controller learns

`artifacts/tables/v11_primary.csv`, Figure `artifacts/v11_confirmatory.png`:

| Task | Condition | Before | After | Change | Bootstrap 95% | State-dependence |
|---|---|---|---|---|---|---|
| catch | e-prop | 0.206 | **0.661** | +0.456 | [0.388, 0.567] | 0.442 |
| catch | frozen | 0.206 | 0.206 | 0.000 | [0.000, 0.000] | 0.000 |
| catch | rewired | 0.206 | **0.817** | +0.611 | [0.508, 0.738] | 0.486 |
| dodge | e-prop | 0.794 | **0.926** | +0.132 | [0.121, 0.154] | 0.402 |
| dodge | frozen | 0.794 | 0.794 | 0.000 | [0.000, 0.000] | 0.000 |
| dodge | rewired | 0.794 | **0.938** | +0.143 | [0.083, 0.200] | 0.418 |

Random baselines are 0.197 (catch) and 0.803 (dodge); the heuristic scores 1.000.
The e-prop arm beats the random policy by **+0.464** [0.408, 0.563] on catch and
**+0.124** [0.100, 0.158] on dodge, on 3/3 seeds in both cases. All three
preregistered success criteria are met. The frozen arm reproduces its initial score
exactly, so the gain is attributable to learning and not to evaluation drift.

Note that the frozen arm's greedy mode is degenerate by construction: with
zero-initialised actor weights every logit ties and the mode is a constant action.
The *random* policy is therefore the more informative chance reference, and it is
the one the success criterion is stated against.

## 15. The biological topology confers no advantage

This is the comparison the protocol asked for as soon as genuine learning existed:
identical learning rule, identical neuron count, identical readout architecture,
identical training budget, identical standardisation procedure, biological versus
degree-preserving rewired graph.

| Contrast | Mean | Bootstrap 95% | Seeds |
|---|---|---|---|
| catch: e-prop − rewired | **-0.156** | [-0.325, -0.021] | -0.325, -0.121, -0.021 |
| dodge: e-prop − rewired | -0.011 | [-0.079, 0.071] | -0.079, -0.025, +0.071 |

On catch the rewired control is **better** than the biological graph on all three
seeds, with an interval excluding zero. On dodge the two are indistinguishable. We
report this in the direction it came out.

A post-hoc analysis of the descending representation explains the direction
(`artifacts/v11_topology_diagnostic.json`, frozen cores, matched uniform-random
behaviour, development-block seeds):

| Property | Biological | Rewired |
|---|---|---|
| Participation ratio (effective dimensions) | 29.5 | **54.5** |
| Mean absolute pairwise correlation | 0.152 | **0.112** |
| Linear probe accuracy (majority 0.431) | 0.774 | **0.901** |

Degree-preserving rewiring decorrelates the descending population and nearly doubles
the number of directions carrying variance, so a linear readout has more independent
features to exploit. The biological wiring concentrates descending activity into a
lower-dimensional, more correlated code.

**We do not claim the biology is worse.** The readout here is an artificial linear
decoder on a truncated circuit that is missing 85% of its incoming synaptic weight,
and correlated population structure may serve functions this task cannot see —
including robustness, multiplexing of other behaviours, or metabolic economy. What
the experiment does establish is narrow and worth stating plainly: for this task,
this readout and this learning rule, connectome-derived topology is not the
ingredient that makes learning possible, and a degree-matched random graph does at
least as well.

## 16. Lesions, now that there is a policy to lesion

v1 reported that perturbations changed nothing, and correctly refused to call that
robustness, because a collapsed policy cannot be degraded by ablating inputs it
ignores. With a genuinely state-dependent policy the same lesions become
informative (`artifacts/tables/v11_robustness.csv`, Figure
`artifacts/v11_robustness.png`):

| Task | Unperturbed | Sensory noise sd0.1 | 10% edge ablation | 10% neuron ablation |
|---|---|---|---|---|
| catch | 0.661 | 0.453 | 0.617 | **0.329** |
| dodge | 0.926 | 0.914 | 0.924 | 0.939 |

On catch the learned policy degrades substantially: neuron ablation costs 0.332 and
sensory noise 0.208, while edge ablation costs only 0.044. The ordering is
interpretable — removing 10% of neurons deletes whole readout features, whereas
removing 10% of edges perturbs a highly redundant recurrent drive. Dodge is
unaffected because it sits near its ceiling and a mostly-correct policy still avoids
a single object.

This contrast is itself a methodological result: **lesion insensitivity is only
evidence about a circuit that is doing something.** Running the identical lesions
against v1 and v1.1 shows the same numbers meaning opposite things.

## 17. What Part II adds, and what it does not

**Supported.** Replacing global reward-modulated STDP with an actor-critic e-prop
rule — holding the connectome, circuit, encoding, tasks and reward fixed — converts
a preregistered negative result into a confirmatory positive one on untouched seeds.
The readout-only stage suffices; recurrent plasticity was never required. Common-mode
removal and readout scaling were each necessary. The learned policy is
state-dependent and degrades under lesions in an interpretable order. A
degree-matched random graph learns at least as well as the biological one, and its
descending code is higher-dimensional and less correlated.

**Not supported.** That this is how Drosophila learns anything. That e-prop is
biologically implemented in the fly. That the connectome helps or hinders learning in
general — we tested one task family, one readout and one rule. That the rewired
control isolates a single graph property: it preserves in/out degree and outgoing
strength but not incoming strength, and its mixing is unvalidated. That three seeds
support confirmatory significance claims; the intervals remain descriptive.

The honest one-line summary of both studies together: *the biology we could measure
delivered a usable signal, the first learning rule we engineered threw it away, the
second one used it — and a random graph with the same degrees used it slightly
better.*
