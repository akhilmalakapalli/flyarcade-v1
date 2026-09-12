# Frozen v1 methods

The machine-readable preregistration is `run_plan.json`. Development seed 99 and
environment seeds 1,900,000 / 2,000,000 were used for two readout candidates before
held-out trials. Per-neuron readout was selected. No held-out retuning is allowed.
The study is exploratory: three training seeds support limited uncertainty estimates.

## Circuit

Use Traced MaleCNS v1.0 neurons innervating `LAL(L)`, `LAL(R)` or `GNG` and
annotated as `visual_projection`, `descending_neuron`, or `cb_intrinsic` with
`class=CX`. Selection depends only on anatomy/annotations. It yields 2,040 neurons
and 126,676 directed non-autaptic edges. Preserve original body IDs, integer
relationship weights, raw neuronal properties and checksummed raw edge ROI data.
All 1,293 descending neurons are graph-reachable from the 218 visual neurons;
only about 15%/7% of global incoming/outgoing synaptic weight is retained. Missing
external drives are replaced with a uniform artificial current, a major modeling
limitation. Descending neurons are treated as motor-command outputs, not muscles.

## Dynamics and plasticity

One action spans four discrete simulation ticks. Each neuron integrates
`v[t+1] = clip(0.85*v[t] + drive[t] + signed_sparse_input[t], -3, 4)` when not
refractory. Threshold is 1, reset 0, one-tick refractory period, one-tick synaptic
transmission delay. These are dimensionless model parameters, not fitted time
constants. Initial magnitude on edge i→j is `2.5*count(i,j)/sum_k count(k,j)`.
Consensus acetylcholine is modeled positive, GABA negative. Other/unclear
transmitters have zero fast effect in the primary model; their graph edges and
annotations remain present. This conservative sign simplification omits substantial
biology, including receptor-dependent glutamatergic effects and neuromodulation.

For recurrent plasticity, old pre/post traces produce a signed pair term:
`pair = trace(pre)*spike(post) - trace(post)*spike(pre)`;
`eligibility = clip(0.95*eligibility + pair, -5, 5)`;
`trace = 0.8*trace + spike`. Reward advantage is the reward minus an exponential
moving baseline (baseline step 0.02), clipped to [-2,2]. Update magnitudes by
`0.0002*advantage*eligibility`, clipped to [0.25,2] times initial magnitude. Signs
and structural connectivity never change. This is an engineering implementation
inspired by reward-modulated eligibility rules, not a literal reproduction of a
published biological circuit. See [Florian (2007)](https://pubmed.ncbi.nlm.nih.gov/17444757/)
for reward-modulated STDP and eligibility traces.

Visual neurons receive a background current of 0.18 plus Bernoulli feature
stimulation of amplitude 1.5. The eight engineered channels encode positive and
negative target error, lane match, player position/complement, object position/
complement and falling phase. Assignment cycles through sorted body IDs; there
is no claim of a biological retinotopic mapping. All other neurons receive only
the background and recurrent input.

Each descending neuron's four-tick spike fraction, centered at 0.25, enters an
artificial three-action softmax readout. Readout policy eligibility decays by
0.5 per action and adds `(one_hot(action)-probability)*features`; weights update
by `0.01*advantage*eligibility`, clipped to [-4,4]. Initial weights are Gaussian
sd0.01. No backpropagation is used. Learning can arise in this artificial readout;
it must not be attributed to recurrent STDP alone.

## Controls, evaluation and transfer

Four conditions: full learning; both learning mechanisms frozen; directed-degree
rewired graph with full learning; readout learning with recurrent weights frozen.
Rewiring uses 10 attempted swaps per edge with seed `700+training_seed`, preserving
in/out degree and outgoing anatomical strength, not incoming strength. Recomputed
incoming normalization and source signs can change effective dynamics: the
comparison does not isolate every graph property. Report acceptance diagnostics;
mixing to a uniform random graph is not established.

Each main trial trains 200 episodes on one task, with model seeds 0/1/2. Training
environment seed is `1000+10000*seed+episode_index`. Held-out seeds are
`3000000+100*seed+i` for i=0..19; policy/encoder evaluation RNG uses that seed plus
500000. Initial and final evaluations share seeds and freeze **both** learning
mechanisms and the reward baseline. Every episode resets voltage, refractory
state and traces. Evaluation saves/restores the training RNG object. Checkpoints
save weights, baseline, RNG, configuration and completed-episode history every
25 episodes; interruption preserves the last complete checkpoint.

Robustness uses frozen final weights, Gaussian observation noise sd0.1 (clipped
to observation bounds), 10% edge ablation, and 10% neuron ablation, independently.
Masks are sampled once per trial with seed `800+training_seed` and held fixed
across all evaluation episodes. Ablation includes sensory/output neurons. Transfer
trains 100 source then 100 target episodes in both directions, compared with 200
target-only episodes (equal total budget) and a fresh 100-target model (equal
target exposure). The 100-target fresh and transfer target phases use different
training seed blocks; held-out target environments are matched.

Report success rates, paired post-minus-pre changes, control contrasts and
perturbation/transfer results at the independent training-seed level. Do not treat
20 evaluation episodes as 20 independent trained models. Bootstrap seed means
descriptively with three seeds; no confirmatory p-values or broad biological
claims. Retain negative results. Each trial is independently guarded at 120 s,
2 GiB RSS and the existing disk reserve; trials execute sequentially.

## Post-hoc diagnostics (added after the preregistered grid completed)

The preregistered primary result is negative, so a separate analysis asks where the
failure sits. It reads frozen saved weights only. It trains no controller, selects
no model, and none of its output feeds back into the model, the protocol or any
reported outcome. It is not part of the preregistration and is labelled exploratory
wherever it appears.

Probe decoders are analyst instruments, not controller components. A probe measures
how much task information the descending population carries; its accuracy is never
reported as a behavioural result. The probe is a shared-covariance regularised
linear discriminant fitted on one half of the recorded decisions and scored on the
other, predicting the required movement direction (left/stay/right, derived from
the lane difference). An earlier one-vs-rest least-squares probe was rejected
because it cannot place an ordinal middle class at the argmax and therefore
understates decodable information. Being linear, the probe is a lower bound on the
information present. Every probe is reported alongside its majority-class rate and
a shuffled-label control fitted the same way.

A trained controller that collapses onto one action visits a degenerate state
distribution, which inflates probe accuracy, per-neuron discriminability and the
majority-class rate alike. All condition comparisons therefore use a **matched**
behaviour policy: actions actually executed are drawn uniformly from a fixed
separate stream, identical across conditions, while the controller's own action
probabilities are still recorded for the visited states.

Constant-action reference policies (always-left, always-stay, always-right)
are evaluated on the same held-out seeds. They are the correct comparison for a
collapsed policy, because a five-lane task with a uniformly drawn object rewards a
wall-hugging constant action at roughly 1/5 for catch and 4/5 for dodge.

## Reproducibility verification

Every completed trial is restored from its saved final checkpoint and its held-out,
sensory-noise, edge-ablation and neuron-ablation evaluations are re-run and
compared exactly. Each trial replays in its own subprocess so the unchanged 120-second
per-trial guard applies as it did during training; the sequential total is not
treated as one guarded operation and no limit is raised. Pre-training evaluations
predate the surviving checkpoint and are explicitly not replayable from it. This
verifies implementation consistency, not physiological validity.

---

# Snake (Study 3) methods

The machine-readable preregistration is `experiments/snake_run_plan.json`. Snake
reuses the v1.1 architecture; only the differences are recorded here.

## Environment

8x8 grid, snake starting at length 3 pointing right, food placed uniformly on a free
cell with a seeded generator. Four absolute actions (up/down/left/right); an
immediate 180-degree reversal is refused rather than fatal. Episodes end on wall
collision, self collision, 200 steps, or 64 steps without food. Reward is bounded and
interpretable: +1.0 for food, -1.0 for death, -0.01 per step. No distance-to-food
shaping is used; none was needed, and none is included precisely so that the agent
cannot score without playing Snake.

## Sensory encoding

21 normalised channels, Bernoulli-injected into visual-projection neurons by the same
scheme the lane tasks use: relative food direction (4), normalised food distance (1),
heading one-hot (4), immediate danger in each direction (4), 3x3 local occupancy
around the head (8). No pixel or convolutional input. Channels are engineered
interfaces and carry no biological claim.

## Preserved from v1.1

MaleCNS v1.0 2,040-neuron subgraph and body IDs; LIF membrane dynamics and the
transmitter sign rule; the actor-critic reward-guided learning rule; per-neuron
common-mode removal; sqrt(width) feature scaling; `critic_lr`, `discount` and
`trace_decay`; stage A plasticity scope with the recurrent core entirely frozen;
resource guards; deterministic seeding; greedy evaluation with learning disabled.

## Snake-specific changes

Four parameters differ, each forced by a measured development failure rather than
chosen to improve a result: `actor_lr` 0.4 -> 8.0, `entropy_coefficient` 0.03 ->
0.003, `exploration_floor` 0.1 -> 0.02, `training_episodes` 600 -> 4000. The reasons,
with the development measurements behind them, are recorded in
`experiments/snake_run_plan.json` under `snake_specific_changes`. At the unchanged
v1.1 parameters Snake did not learn at all.

## Resource handling

Snake episodes lengthen as the policy improves, so a 4,000-episode budget does not
fit one 120-second guarded process. Trials checkpoint at episode boundaries and
resume in a **fresh independently guarded process**, following the v1 pattern. A
guard is never reset to continue an oversized operation and no limit is raised.

## Seeds

Development training 7,000,000 + 10,000*dev_seed + episode; development evaluation
7,500,000 block; feature fitting 7,200,000 block; confirmatory training 8,000,000 +
10,000*seed + episode; confirmatory evaluation 9,000,000 + 100*seed + i for i=0..39.
`tests/test_snake.py` asserts these are disjoint from each other and from every v1
and v1.1 block.

## Standardisation and the topology control

Each topology is standardised by the same frozen procedure applied to **its own**
activity: `artifacts/snake_standardizer.json` for the biological graph and
`artifacts/snake_standardizer_rewired_{seed}.json` for each rewired control. An
earlier run that standardised the rewired arm with biological statistics made it look
as though rewiring abolished Snake learning entirely; that effect disappeared under
correct per-graph calibration. Any future topology comparison must calibrate each
graph by the same procedure applied to its own rates.

## Perturbations

Applied only after state-dependent learning existed, with frozen learned weights and
masks sampled once per trial: sensory noise sd 0.1, 10% edge ablation, and 5%, 10%
and 25% neuron ablation. Insensitivity is not reported as robustness unless the
intact policy demonstrably uses the affected information.
