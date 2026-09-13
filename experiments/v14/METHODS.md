# v1.4: downstream architecture development study

The question is how downstream learning architecture changes the behavior extracted
from the same fixed 2,040-neuron MaleCNS-derived spiking substrate. This is a
prospective development comparison, followed by bounded conditional secondary
analyses and fresh validation. It is not a confirmatory experiment.

## Version and preservation

Branch `flyarcade-v1.4-performance` starts from the latest dashboard base `b444631`.
The development plan was committed in `ee7d012` before training; implementation
was committed in `03c2acc`. `artifacts/v14/implementation.json` records training
source/dependency hashes and the numerical environment. The historical snapshot
covers 2,022 files, including frozen results, plans, models, data and dashboard
source. Historical summary scores are read as references only. Historical
confirmatory episodes are never executed, rescored or reused.

## Controlled primary comparison

Five tasks: Catch, Dodge, historical 8×8 absolute-action Snake, Pong and Flappy.
The original environment classes and metrics remain intact. Catch/Dodge wrappers
only accumulate the original event score. Snake wrappers expose raw food, steps
and body length; they retain original movement, reversal, collision, food,
starvation and time-limit behavior. Pong and Flappy use their frozen target classes
without overrides. Software checks compare target trajectories on new test seeds.

All three arms use the same original input mapping, recurrent graph, synaptic
magnitudes, transmitter-sign assumptions and LIF dynamics. Catch/Dodge retain
8 engineered channels; Snake retains 21. Pong/Flappy retain complementary v1.3
encoding. Each primary model observes descending-neuron rates after four ticks.
No recurrent synapse learns and no biological neuron is added. The MLP and GRU
units are artificial downstream computation, not additional biological neurons.

- Linear: affine categorical actor and affine scalar critic.
- MLP: 128-unit projection, LayerNorm/tanh, 64-unit tanh layer, actor/critic heads.
- GRU: the same 128-unit projection, 64-unit GRU, actor/critic heads.

The affine arm uses the same PPO/GAE/Adam as the nonlinear arms, rather than
reproducing the historical online-TD update. This deliberately isolates readout
architecture; historical TD is a separate reference. All arms receive 65,536
training transitions, 16 environments, 128-step rollouts, four PPO epochs and
eight minibatches. GRU chunks contain 16 steps with recorded initial hidden states
and episode reset masks. Temporal-difference bootstrap treats pure time limits
as truncations; death/starvation remains terminal. Original training rewards are
unchanged. PPO uses gamma 0.99, GAE lambda 0.95, clip 0.2, entropy coefficient 0.003,
value coefficient 0.5, Adam learning rate 0.0003 and gradient norm limit 0.5.
Learning rate declines linearly to a 5% floor. There is no architecture-specific
primary hyperparameter search. This compares a common operating point, not the
best possible hyperparameters for each architecture.

Feature fitting uses at least 8,192 fresh development states under alternating
reference/random behavior, completing batches of episodes. The standardizer uses
per-neuron means, SD floor 0.02 and clipping at four SD, without square-root-width
normalization, matching v1.3 PPO preprocessing. Fits are task/tick/readout specific
and shared across learners. No validation state contributes to feature fitting.

## Seeding, selection and escalation

All scientific seed namespaces start at 100 billion, beyond historical and
9-billion dashboard seeds. Task blocks are 100 billion, purpose blocks 5 billion,
training-seed slots 1 million. Neural +1-billion and behavior +2-billion RNG
derivations remain disjoint from other purposes. Three paired development seed
slots are shared across configurations; development reuse is intentional common
random numbers. Feature fitting, training, model initialization, sampling,
learning curves, development evaluation, probes, imitation and validation use
separate purposes. Future confirmatory blocks are reserved and rejected by the
execution API.

Every primary learner runs all three seeds. Each configuration gets 40 development
evaluation episodes per seed and the same 16-episode learning-curve block every
16,384 transitions. Selection requires finite runs, fixed-history observation
intervention diversity above 0.05 and dominant trajectory action fraction below
0.98 on all three seeds. Eligible configurations are ranked by mean minus one
quarter of population SD; numerical ties favor simpler models. If none qualifies,
the best stable model is retained but labeled as failing the behavior gate.

Primary winners are fixed before secondary runs. The prescribed secondary order is
8 ticks, all-2,040-neuron readout, 131,072 transitions, then training curriculum
for Snake/Flappy. Each change uses the current development winner, with no free
factorial search. Dodge receives only the tick comparison to limit expenditure
on a high-baseline task. Escalation stops when an eligible development winner
reaches the target. Per-task configuration caps include the three primary arms.

Snake curriculum starts on grids 4, 6, then 8, using the same historical game class.
Flappy uses the existing wider/slower curriculum, then its exact target. Episode
starts switch levels at 25% and 50% of the transition budget; episodes already in
progress finish at their original level. Actual exposure counts are extracted
from checkpoints in `curriculum_exposure.json`. Curriculum effects use a matching
long-budget target-only comparator when available.

Only Snake and Flappy can receive one labeled SECONDARY RESCUE PHASE if their
predeclared useful-performance threshold is unmet after secondary escalation.
It adds 8,192 heuristic demonstration transitions and four supervised passes before
131,072 ordinary target-task PPO transitions. The supervised optimizer is separate;
PPO moments start fresh. GRU imitation uses 16-step truncated sequences and reset
masks. This extra teacher information and transition budget are reported separately
and never mixed into the primary architecture comparison.

## Fresh validation and reporting

Primary and final selections lock before any validation is accessed. Each distinct
selected checkpoint gets one 100-episode fresh validation block for each of the
three trained seeds. When primary and final selections coincide, their one
validation result serves both roles. There is no validation training, retuning,
early stopping or checkpoint selection. Independent primary/final validation
blocks mean their difference is descriptive, not a paired environment contrast.

Each run records all actor/critic/PPO parameters, histories, score, action fractions,
policy entropy, state probe, finite checks, timing, OS high-water memory, checkpoint
and provenance hashes. Snake food is not equated numerically with other tasks'
success fractions. Descriptive uncertainty resamples three trained-seed means
10,000 times with fixed analysis RNG 1400; it supplies no confirmatory p-values.
Historical comparisons are context only: their learners, budgets and preprocessing
differ and are not matched v1.4 controls.

The scientific audit checks all attempted configurations, primary transition
balance, seed separation, selection rules, validation provenance, historical hashes,
checkpoints, fixed-core invariants and resource caps. Checkpoint inference replay
uses fresh software-test seeds, never validation or historical confirmatory seeds.
One primary development training trajectory is repeated as a software audit and
is not counted as an additional independent training seed or development candidate.

## Limits of interpretation

Engineered observations and artificial actions do not model natural fly vision or
muscle control. Only a limited fraction of MaleCNS connectivity is retained. A
weak downstream learner cannot establish an information-capacity limitation of
the biological substrate. Conditional secondary analyses do not test all
architecture–tick–readout interactions. In particular, a tick result for a selected
Linear model cannot establish what a separately optimized GRU would achieve.
All-neuron readout accesses visual and intrinsic populations as well as descending
neurons; improvements may reflect easier access to engineered input information.
Three training seeds and finite budgets limit generalization. No new sensory-only
control was authorized in this plan, so a fixed-substrate bottleneck cannot be
isolated from learning, encoding and readout limitations.
