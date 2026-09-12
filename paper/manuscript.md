# What does and does not emerge from reward-modulated plasticity in a MaleCNS-derived visuomotor controller?

**Status: complete v1 study with a negative primary result.** All preregistered
trials, controls, ablations, robustness and transfer comparisons finished and were
verified by exact checkpoint replay. Reward-modulated plasticity did **not** produce
held-out improvement on either task. This manuscript reports that outcome and the
mechanistic analyses that characterise it. Nothing was retuned in response to
held-out results.

## Abstract

We built a spiking visuomotor controller whose recurrent connectivity is an
authentic, checksummed subgraph of the HHMI Janelia MaleCNS v1.0 connectome
(2,040 traced neurons, 126,676 directed edges, 1,054,402 synapses, no isolates),
and trained it on two minimal lane arcade tasks with a reward-modulated
eligibility rule at both the recurrent synapses and an artificial descending
readout. Across three independent training seeds, two tasks and four conditions,
the mean paired held-out change after 200 training episodes was -0.014 (catch)
and -0.017 (dodge) landing-success, with descriptive three-seed bootstrap
intervals spanning zero in every condition. Trained controllers did not exceed a
uniform-random policy (catch 0.175 vs 0.194; dodge 0.794 vs 0.806), while a
deterministic heuristic with identical observations scored 1.000 on both tasks —
so the tasks are solvable and the controller did not solve them.

Diagnostics on frozen saved weights, evaluated under a matched state distribution,
locate the failure precisely. Descending-population rates carry substantial
task-relevant information both before and after training: an analyst-fitted linear
probe recovers the required movement direction at 0.654 (initial) and 0.658–0.713
(trained) accuracy, against a 0.421 majority-class rate and a 0.358–0.425
shuffled-label control. What collapses is the policy. Mean action entropy falls from 1.098 nats
(uniform is log 3 = 1.099) to 0.36 (catch) and 0.13 (dodge), with the most likely
action reaching probability 0.90 and 0.98. The trained controllers converge on
near-constant single actions whose held-out scores (catch 0.13–0.23; dodge
0.77–0.87 for the three degenerate policies) bracket the observed results
entirely. The negative result is therefore not a sensory bottleneck in the
connectome-derived circuit; it is a credit-assignment failure in the
reward-modulated rule under dense shaping.

Topology mattered less than the framing of such studies usually assumes: a
degree-preserving rewired control was statistically indistinguishable from the
biological graph (catch +0.017, dodge -0.025), as were freezing all plasticity and
restricting learning to the readout. Performance was also insensitive to 10% edge
ablation, 10% neuron ablation and sensory noise — which, given policy collapse, is
evidence of degeneracy rather than of robustness.

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
