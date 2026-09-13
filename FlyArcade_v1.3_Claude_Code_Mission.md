# FlyArcade v1.3 Adaptive Learning Mission
## Autonomous Claude Code execution specification

This file is the master plan for building **FlyArcade v1.3** from the completed `flyarcade-v1.2-multitask` branch.

The purpose of v1.3 is not to erase the v1.2 failures. It is to test whether the same 2,040-neuron MaleCNS-derived spiking substrate can support the four harder tasks—**Flappy Bird, Pong, Breakout, and relative-action Snake**—when the learning system is made substantially more capable.

The intended result is a scientifically defensible capability hierarchy, not a tuned demo.

---

# 0. AUTONOMOUS OPERATING MODE

You are Claude Code working inside the FlyArcade repository.

This file is the highest-level execution specification for v1.3.

## Mandatory continuation loop

At the start of work, after any context compaction, after any restart, or whenever you are unsure what remains:

1. Read this entire file.
2. Read `V13_PROGRESS.md` if it exists.
3. Read `STATE.md`.
4. Inspect the current branch and working tree.
5. Identify the first incomplete v1.3 milestone.
6. Execute that milestone.
7. Run the relevant tests.
8. Update `V13_PROGRESS.md` with:
   - what was completed;
   - exact evidence/artifacts;
   - failures encountered;
   - current best development results;
   - exact next action.
9. Continue immediately.

Do **not** stop simply because:
- one configuration failed;
- one task did not learn;
- one test exposed a bug;
- a development run was poor;
- the first architecture was inadequate.

Diagnose, repair what is scientifically allowed, document it, and continue.

Do not ask the user routine coding questions. Make conservative engineering choices and record them in `DECISIONS.md`.

Only stop early if:
- required data are genuinely missing and cannot be reconstructed;
- a credential only the user can provide is required;
- an irreversible action outside the v1.3 branch is required;
- resource limits make the predefined plan impossible even after optimization.

## Scientific anti-cheating rule

The project is allowed to iterate aggressively on **development data**.

The project is **not** allowed to repeatedly inspect confirmatory results and keep tuning until they become positive.

The correct loop is:

```text
diagnose
-> develop on development seeds
-> validate on new development seeds
-> freeze code/protocol
-> run untouched confirmatory seeds once
-> report honestly
```

After `experiments/v13_run_plan.json` is frozen and confirmatory evaluation begins, numerical-result-affecting changes invalidate the confirmatory study.

If final confirmatory results fail despite a strong development program, preserve the failure. Do not silently tune on it.

---

# 1. CREATE THE v1.3 BRANCH AND PRESERVE HISTORY

The completed source branch is:

```text
flyarcade-v1.2-multitask
```

The known v1.2 commit at the time this file was written is:

```text
00861e0e5cb7e08584836395c41f2df1d5234da5
```

Start with:

```bash
git fetch origin
git switch flyarcade-v1.2-multitask
git pull --ff-only
git switch -c flyarcade-v1.3-adaptive-learning
git push -u origin flyarcade-v1.3-adaptive-learning
```

If `flyarcade-v1.3-adaptive-learning` already exists, verify it descends from the completed v1.2 branch and continue there.

Before model changes:

- hash every protected historical tracked file;
- save the hash snapshot under `artifacts/v13/historical_hashes.json`;
- preserve v1, v1.1, historical Snake, and v1.2 artifacts;
- do not overwrite `artifacts/v12/`;
- do not overwrite old `runs/`;
- do not rewrite old run plans;
- do not replace old checkpoints;
- preserve current paper as `paper/manuscript_v12_historical.md` before later updating the active manuscript.

Create v1.3-only namespaces:

```text
src/flyarcade/v13/
scripts/v13_*.py
tests/test_v13_*.py
experiments/v13_development_plan.json
experiments/v13_run_plan.json
experiments/METHODS_V13.md
artifacts/v13/
runs/v13-...
V13_PROGRESS.md
```

---

# 2. DIAGNOSIS: WHY v1.2 DID NOT SOLVE THE FOUR NEW TASKS

Use this as the starting hypothesis, then verify each point from code and experiments.

## 2.1 v1.2 final outcomes

The v1.2 task battery produced approximately:

| Task | Biological | Frozen/before | Random | Rewired | Heuristic |
|---|---:|---:|---:|---:|---:|
| Flappy | 0.000 | 0.000 | 0.000 | 0.000 | 0.0875 |
| Pong | 0.350 | 0.301 | 0.253 | 0.489 | 1.000 |
| Breakout | 0.368 | 0.365 | 0.468 | 0.402 | 0.617 |
| Snake | 0.0425 | 0.005 | 0.025 | 0.0617 | 0.928 |

None of the four met every frozen v1.2 success criterion.

The correct conclusion is not “the fly network cannot learn these games.” The correct conclusion is:

> the v1.2 Stage-A learning system was not sufficient to reliably extract successful behavior from the fixed neural substrate on these harder tasks.

## 2.2 The v1.2 policy is linear

The current successful v1.1/v1.2 Stage-A policy is essentially:

```text
1,293 standardized descending-neuron rates
-> linear actor
-> action
```

with a linear value function.

This is the first major bottleneck.

The harder tasks require nonlinear interactions.

### Relative-action Snake

The new Snake policy chooses:

```text
left / straight / right
```

but observations encode:
- absolute food direction;
- current heading;
- local dangers.

The correct action depends on interactions like:

```text
food east + heading north -> turn right
food east + heading south -> turn left
```

A direct linear readout has difficulty representing these conditional rules.

This is especially important because the repository already contains an older **absolute-action Snake** experiment that did learn meaningfully with the same 2,040-neuron substrate. That old task mapped food direction much more directly to up/down/left/right actions.

Therefore, new Snake failure is strong evidence for a readout-capacity problem rather than evidence that 2,040 neurons are inherently insufficient.

### Flappy

Correct flap decisions depend jointly on:
- bird height;
- vertical velocity;
- target gap;
- distance/time to pipe.

Again, this is nonlinear state-action structure.

### Breakout

Correct paddle motion depends on:
- ball position;
- velocity;
- predicted intercept;
- brick state.

### Pong

Pong is simpler but still dynamic and trajectory-dependent.

## 2.3 v1.2 uses one-pass online TD learning

The current actor-critic consumes transitions online and effectively uses each sample once.

This is inefficient because the expensive part of the project is producing fly-network features.

v1.3 should collect trajectories once and then reuse the stored fly features for multiple PPO optimization epochs.

## 2.4 Episode counts are a bad training-budget unit

v1.2 used roughly:

```text
Flappy   1200 episodes
Pong      600 episodes
Breakout  600 episodes
Snake    2000 episodes
```

But early failure can make an “episode” only a few actions long.

Historical Snake already documented this exact issue: short early episodes meant that a nominally large episode count supplied far fewer learning updates than Catch/Dodge.

v1.3 must budget training primarily by:

```text
environment transitions
```

not only episodes.

## 2.5 Long-horizon rewards are discounted too aggressively

v1.2 inherited approximately:

```text
gamma = 0.9
trace_decay = 0.7
```

For delayed rewards:

```text
0.9^20 ≈ 0.12
0.9^40 ≈ 0.015
```

This is very weak credit for an action that matters 20–40 steps later.

v1.3 should develop around:

```text
gamma ≈ 0.99
GAE lambda ≈ 0.95
```

rather than blindly inheriting the short-horizon values.

## 2.6 Flappy is currently a benchmark problem before it is a learning problem

In v1.2:
- biological = 0;
- random = 0;
- rewired = 0;
- heuristic ≈ 0.0875.

That is a red flag.

Before spending neural-learning compute, determine whether:
- the environment is actually solvable under current dynamics;
- the heuristic is simply poor;
- collision/scoring logic is pathological;
- timing/dynamics are too unforgiving;
- observation variables are inadequate.

Flappy must pass a solvability gate first.

## 2.7 Existing representation probes are not sufficient

Examples from v1.2:
- Flappy action-probe accuracy is about the same as its ~95.5% majority-action baseline, so the apparent high accuracy is meaningless.
- Pong has some linearly decodable action information.
- Breakout only weakly exceeds majority.
- Snake biological linear action decoding does not exceed majority.
- rewired networks often have higher dimensionality/lower population correlation.

v1.3 must use:
- balanced accuracy;
- class counts;
- continuous state-variable regression;
- small nonlinear diagnostic probes;
- matched states across topology conditions.

## 2.8 Four SNN ticks per action may be too little

The current controller uses only four spiking ticks per environment action.

Treat this as a development parameter:

```text
ticks_per_action in {4, 8, 12, 16}
```

Measure:
- information quality;
- rate variance;
- runtime;
- learning.

Do not assume the maximum is best.

## 2.9 More neurons are not the first fix

The same 2,040-neuron graph already supports:
- strong Catch;
- strong Dodge;
- meaningful historical Snake.

Therefore intervention order must be:

1. nonlinear decoder;
2. better temporal credit;
3. sample-efficient learning;
4. longer neural integration;
5. curriculum/reward shaping;
6. recurrent fly plasticity if truly needed;
7. more neurons only as a separate last-resort experiment.

---

# 3. v1.3 SCIENTIFIC QUESTION

The v1.3 study should answer:

> Can increasingly capable learning mechanisms extract successful behavior from the same MaleCNS-derived spiking substrate on harder artificial sensorimotor tasks?

Use a capacity ladder:

```text
A1: historical linear online actor-critic
A2: frozen fly core + nonlinear MLP PPO
A3: frozen fly core + GRU PPO
B/C: biologically constrained recurrent plasticity only if A2/A3 remain insufficient
```

For each task, retain the **simplest architecture that robustly succeeds in development**.

Do not make every task maximally complex by default.

---

# 4. ENVIRONMENT VALIDATION BEFORE FLY LEARNING

## 4.1 Preserve target environments when possible

For:
- Pong;
- Breakout;
- relative-action 6x6 Snake;

keep the final target dynamics/observations identical to v1.2 unless a genuine bug is discovered.

Training curricula may use easier variants, but final evaluation must use the target environment.

## 4.2 Flappy solvability gate

Before RL development:

1. Build a competent deterministic oracle/heuristic.
2. Prefer short-horizon model-predictive control or beam search using known deterministic dynamics.
3. Evaluate over at least 100 fresh development seeds.
4. Test pipe crossing, wall collision, and action timing.

Required gate:

```text
mean primary success >= 0.70
```

Prefer >= 0.80.

First try to achieve this **without changing Flappy dynamics**.

If a competent oracle still cannot solve the task:
- diagnose the environment;
- make the minimum defensible correction;
- create an explicitly versioned `FlappyV13`;
- document every change from v1.2;
- regenerate random/heuristic baselines;
- never directly mix v1.2 and v1.3 Flappy numbers.

Possible fixes only if justified:
- widen gap modestly;
- slow obstacle slightly;
- rebalance flap/gravity;
- fix initial conditions;
- correct collision boundary bugs.

Do not make Flappy trivial.

---

# 5. BUILD A DIRECT-OBSERVATION SOLVABILITY CONTROL FIRST

Before asking the fly network to solve each task, prove that the learning implementation can solve the environment from its engineered observation.

Create:

```text
normalized engineered observation
-> same learner family
-> action
```

with no fly SNN.

Call this the **sensory-only** condition.

During development, sensory-only must robustly beat random on:
- Flappy;
- Pong;
- Breakout;
- Snake.

If sensory-only fails, do **not** blame the fly representation.

Fix:
- environment;
- reward scaling;
- PPO implementation;
- training budget;
- curriculum.

Keep sensory-only as a final control because a reviewer will reasonably ask whether the downstream network, rather than the fly substrate, is doing all useful computation.

---

# 6. IMPLEMENT A SMALL CPU-ONLY PPO LEARNER

A compact project-specific PyTorch implementation is preferred.

Do not import a giant RL framework unless necessary.

If adding PyTorch:
- put it in an optional dependency group such as `learning`;
- record exact version;
- CPU-only for confirmatory reproducibility;
- seed Python, NumPy, and Torch;
- use deterministic algorithms where practical;
- record platform/version metadata.

## 6.1 Stage A2: MLP PPO

Primary input:

```text
1,293 standardized descending-neuron features
```

Recommended starting network:

```text
Linear(1293, 128)
LayerNorm(128)
Tanh
Linear(128, 64)
Tanh
Actor head
Critic head
```

Shared trunk is acceptable.

Use carefully scaled or orthogonal initialization.

## 6.2 Stage A3: GRU PPO

If MLP is not robust:

```text
1293 fly features
-> Linear 128
-> LayerNorm
-> Tanh
-> GRU 64
-> actor head
-> critic head
```

Rules:
- reset hidden state at episode boundary;
- preserve hidden state within episode;
- checkpoint hidden-model/optimizer state;
- greedy deterministic evaluation.

Optional:
- previous action;
- previous reward;

may be added as small recurrent inputs, but document this and apply consistently across topology conditions.

Do **not** feed raw task observation directly into the primary fly policy.

## 6.3 PPO requirements

Implement and test:

- clipped surrogate loss;
- value loss;
- entropy bonus;
- generalized advantage estimation;
- terminal masking;
- advantage normalization;
- gradient clipping;
- minibatches;
- multiple epochs per rollout;
- deterministic evaluation;
- no updates during evaluation;
- optimizer checkpoint/restore;
- rollout checkpoint/resume.

Reasonable development ranges:

```text
gamma:               0.985–0.995
GAE lambda:          0.90–0.98
learning rate:       1e-4–1e-3
PPO clip:            0.1–0.3
entropy coefficient: 0.001–0.02
value coefficient:   0.25–1.0
gradient clip:       0.5–1.0
epochs/rollout:      3–8
MLP width:            64–256
GRU width:            32–128
```

These are development ranges, not final frozen settings.

---

# 7. TRAIN BY TRANSITIONS, NOT JUST EPISODES

Initial development ceilings:

```text
Pong:       100k–300k transitions
Flappy:     200k–500k transitions
Breakout:   200k–500k transitions
Snake:      300k–1M transitions
```

Use these as upper bounds, not automatic targets.

Log:
- transitions;
- episodes;
- PPO updates;
- wall time;
- transitions/sec;
- return;
- task metric;
- entropy;
- state dependence.

Development early stopping is allowed only on development evaluation data.

Confirmatory budget must be frozen.

---

# 8. REWARD SHAPING

Keep the final behavioral metric separate from training reward.

Prefer potential-based shaping:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

This gives dense learning signal without directly encoding the action.

## Flappy

Potential candidate:

```text
Phi = -abs(predicted bird height at pipe crossing - gap center)
```

Event rewards remain:
- pipe pass positive;
- collision negative.

## Pong

Potential:

```text
Phi = -abs(paddle - predicted ball intercept)
```

Events:
- hit positive;
- miss negative.

## Breakout

Potential:

```text
Phi = -intercept error while ball is descending
```

Events:
- brick destroyed positive;
- paddle return small positive;
- miss negative.

## Snake

Potential:

```text
Phi = -normalized Manhattan distance to food
```

Events:
- food positive;
- collision negative;
- small step penalty.

Never directly reward “the correct action.”

Tune shaping only on development data.

---

# 9. CURRICULUM LEARNING

Curriculum is allowed for training.

Final evaluation must always return to the exact frozen target environment.

## Flappy

Possible curriculum:

```text
wider gap / slower obstacle
-> intermediate
-> exact final target
```

Substantial final training must occur on the exact target environment.

## Pong

Try target environment first. Likely no curriculum needed.

## Breakout

Recommended:

```text
Pong-like interception
-> 1 brick
-> 3 bricks
-> final 5-brick target
```

Transfer policy weights between stages.

## Snake

Recommended:

```text
4x4
-> 5x5
-> final 6x6
```

Keep the final relative-action formulation.

Record transition counts at each curriculum level.

Freeze the curriculum schedule before confirmatory runs.

---

# 10. IMPROVE FLY FEATURE EXTRACTION BEFORE ADDING NEURONS

## 10.1 Sweep neural integration duration

Using development data only:

```text
ticks_per_action = 4, 8, 12, 16
```

Measure:
- population variance;
- participation ratio;
- pairwise correlation;
- linear state decoding;
- nonlinear state decoding;
- balanced heuristic-action decoding;
- runtime;
- downstream learning.

Select the smallest duration that materially improves usable information/performance.

## 10.2 Repair representation diagnostics

For discrete labels:
- class counts;
- majority baseline;
- balanced accuracy;
- stratified splitting.

For continuous variables:
- regression R²/correlation;
- held-out test set.

Add small MLP diagnostic probes to distinguish:
- information absent;
- information present but nonlinear.

Do not use confirmatory representation labels to tune the policy.

## 10.3 Feature-scope fallback

Primary model remains descending-neuron features.

Only after serious A2/A3 development failure may you explicitly test:

```text
descending only
descending + central-complex intrinsic
all 2,040 selected neurons
```

as development variants.

If a broader feature population becomes the successful final model, state it explicitly in the paper.

Do not silently change the biological readout population.

---

# 11. DEVELOPMENT SEEDS AND SEARCH PLAN

Before searching, create:

```text
experiments/v13_development_plan.json
```

Use a completely fresh namespace, for example beginning at:

```text
200_000_000
```

Create disjoint blocks for:
- environment validation;
- feature fitting;
- learner screening;
- full learner validation;
- curriculum development;
- representation diagnostics;
- development perturbation smoke tests.

Assert no overlap with:
- v1;
- v1.1;
- historical Snake;
- v1.2;
- future v1.3 confirmatory seeds.

## 11.1 Search ladder

For every task:

1. sensory-only PPO;
2. frozen-core MLP PPO;
3. frozen-core GRU PPO only if MLP is inadequate;
4. curriculum if sparse reward is limiting;
5. feature-integration/readout-scope variants if diagnostics justify them;
6. recurrent biological plasticity only as fallback.

## 11.2 Coarse screening

Use approximately:
- 16–32 deterministic configurations/task;
- 2 development training seeds/config;
- reduced but meaningful transition budget.

Search:
- gamma;
- lambda;
- learning rate;
- entropy;
- PPO clip;
- value coefficient;
- hidden size;
- ticks per action;
- reward-shaping scale;
- rollout length;
- transition budget.

Do not run an unbounded random search.

Store every attempted configuration and result.

## 11.3 Full development validation

Take the top 3 candidates per task.

Evaluate each with:
- full intended transition budget;
- 5 fresh development training seeds;
- at least 60 held-out development evaluation episodes/seed.

Select the **simplest** architecture meeting the development success gate.

Penalize:
- unstable seed behavior;
- constant-action collapse;
- entropy collapse;
- unnecessary complexity.

---

# 12. DEVELOPMENT SUCCESS GATE

Do not expose v1.3 confirmatory seeds until all four tasks pass this development gate.

For each task, across 5 development validation seeds:

1. mean after-training > mean before-training;
2. mean after-training > mean random + 0.05;
3. at least 4/5 seeds individually > matched random + 0.05;
4. fixed-history state-dependence > 0.05;
5. finite model/optimizer values;
6. nonzero policy entropy;
7. no constant-action collapse;
8. close at least 15% of the random-to-heuristic gap:

```text
gap_closure =
    (learned - random) / max(heuristic - random, epsilon)
```

Require:

```text
mean gap_closure >= 0.15
```

Prefer >=0.25 before freeze.

Also report raw metrics:
- Flappy: obstacles passed;
- Pong: returns;
- Breakout: bricks destroyed + paddle returns;
- Snake: food + survival steps.

If a task fails development:
- diagnose;
- use the next allowed rung in the ladder;
- document the change;
- rerun only development;
- update `V13_PROGRESS.md`;
- continue.

---

# 13. RECURRENT FLY PLASTICITY IS A FALLBACK

The repository already supports:

```text
Stage B: plastic synapses terminating on descending neurons
Stage C: additionally plastic synapses onto central-complex intrinsic neurons
```

Do not use this first.

Only consider it after:
- sensory-only PPO works;
- frozen-core MLP/GRU PPO has been properly developed;
- representation diagnostics have been examined.

Important:

The existing recurrent learning signal was designed for a **linear actor**.

Do not incorrectly reuse that linear `readout_projection` with a nonlinear PPO policy.

If Stage B/C is needed, derive a principled per-neuron learning signal, such as a gradient of the policy/value objective with respect to standardized descending features, transformed through the feature standardization and combined with e-prop eligibility.

Because PPO is rollout-based and e-prop is online, design this carefully.

A hybrid approach is acceptable if:
- mathematically documented;
- bounded;
- tested;
- reproducible;
- clearly described as engineered.

Preserve sign constraints and weight bounds.

Do not call this biologically realistic beyond what is justified.

---

# 14. MORE NEURONS ARE LAST RESORT

Do not enlarge the graph just to make a task pass.

Only consider a larger biological subgraph if:
- direct-observation learner works;
- MLP/GRU PPO has been adequately developed;
- longer neural integration has been tested;
- relevant feature scopes have been tested;
- recurrent plasticity has been considered;
- representation diagnostics indicate missing information.

If tested:
- keep original 2,040-neuron model as a condition;
- stay within the existing 4,096-neuron resource guard unless formally versioning the resource policy;
- define a biological selection rule before performance testing;
- do not select neurons based on game scores;
- report retained connectivity;
- build matched rewired controls;
- label it a different model.

---

# 15. REQUIRED FINAL CONTROLS

Every v1.3 task should have:

1. **biological learned**
   - authentic MaleCNS-derived core;
   - selected v1.3 learning architecture.

2. **frozen/untrained**
   - same core/model initialization;
   - no learning.

3. **rewired learned**
   - degree-preserving rewired core;
   - same learner/budget;
   - topology-specific standardization.

4. **random policy**

5. **heuristic/oracle**

6. **sensory-only learned**
   - engineered task observation directly into the same decoder family;
   - no fly SNN.

Optional:
7. matched random reservoir.

Do not claim biological topology is uniquely advantageous unless the data support it.

---

# 16. FREEZE THE v1.3 PROTOCOL

Only after all four pass the development gate:

Create:

```text
experiments/v13_run_plan.json
```

Freeze:
- exact source commit/code hash;
- graph hash;
- exact task versions;
- environment dynamics;
- observations;
- reward functions;
- shaping coefficients;
- curriculum schedule;
- feature fitting;
- neural ticks/action;
- readout population;
- MLP/GRU architecture;
- PPO hyperparameters;
- transition budgets;
- evaluation episodes;
- success criteria;
- controls;
- rewiring;
- perturbations;
- fresh confirmatory seed rules;
- software versions.

Hash the run plan and write a freeze timestamp.

After this point, code that changes numerical outcomes invalidates confirmatory runs.

---

# 17. CONFIRMATORY DESIGN

Use completely untouched seeds.

Minimum:

```text
5 independent training seeds
```

for each learned primary condition.

Prefer:

```text
100 held-out evaluation episodes per training seed
```

Minimum 60 only if runtime accounting proves 100 impractical.

Use matched evaluation seeds across comparable conditions.

## Confirmatory success criterion

Predefine before running.

A task is a v1.3 learning success only if:

1. biological mean after > biological mean before;
2. biological mean after > random mean + 0.05;
3. at least 4/5 biological seeds > matched random + 0.05;
4. mean random-to-heuristic gap closure >= 0.15;
5. fixed-history state-dependence > 0.05;
6. policy is not constant-action;
7. all numerical values are finite.

Report all seed values and descriptive uncertainty.

Do not invent confirmatory p-values.

If confirmatory fails:
- report it;
- do not retune v1.3;
- any further tuning requires a new version.

---

# 18. PERTURBATIONS

After successful training:

Evaluate biological policies under:
- clean;
- sensory Gaussian noise SD 0.1;
- 10% edge ablation;
- 10% neuron ablation.

Use fixed seeded masks.

No retraining.

If GRU is used, define hidden-state reset consistently.

---

# 19. REPRESENTATION ANALYSES

For biological and rewired graphs using matched exogenous states:

Report:
- participation ratio;
- mean pairwise correlation;
- activity variance;
- balanced action decoding;
- majority baseline;
- continuous state-variable decoding;
- nonlinear probe performance;
- class counts.

Use these analyses to explain:
- why some stages succeed;
- why some fail;
- biological vs rewired differences.

Do not overinterpret post-hoc analyses as causal proof.

---

# 20. TEST REQUIREMENTS

Add comprehensive v1.3 tests.

## PPO
- probability normalization;
- finite logits;
- deterministic forward with fixed seed;
- GAE terminal handling;
- clipping objective;
- entropy;
- gradient clipping;
- optimizer save/restore;
- no updates during evaluation.

## GRU
- reset hidden state at episode boundaries;
- hidden state persists within episode;
- checkpoint restore exact enough for deterministic replay;
- no training-state leakage into evaluation.

## Fly interface
- Stage A core unchanged byte-for-byte;
- selected ticks/action deterministic;
- standardizer uses development data only;
- feature dimensions correct;
- no raw task observation leaks into primary fly policy.

## Environment
- deterministic trajectories;
- finite bounded observations;
- valid actions;
- scoring;
- target Pong/Breakout/Snake unchanged unless explicitly versioned;
- Flappy oracle gate.

## Seeds
- historical blocks disjoint;
- v1.3 development and confirmatory disjoint.

## Historical integrity
- protected files unchanged;
- historical checkpoint replays exact.

Run the complete old test suite after each major architecture milestone.

---

# 21. RESOURCE STRATEGY

Keep the project CPU-first.

Do not raise guards just to make experiments easier.

Reuse:
- v1.2 optimized fixed-core sparse execution;
- checkpointing;
- resumable transition counters;
- cached rollouts where mathematically valid.

Critical optimization:

PPO epochs should reuse stored:

```text
fly features
actions
rewards
dones
log probabilities
values
```

instead of rerunning the SNN for every gradient epoch.

Do not cache across states in a way that changes SNN history.

Track:
- wall time;
- RSS;
- transition throughput;
- checkpoint size.

---

# 22. AUTONOMOUS MILESTONE STATE MACHINE

`V13_PROGRESS.md` must use these statuses:

```text
M0_BRANCH_AND_PRESERVATION
M1_ENVIRONMENT_VALIDATION
M2_PPO_IMPLEMENTATION
M3_SENSORY_ONLY_SOLVABILITY
M4_REPRESENTATION_DIAGNOSTICS
M5_FLY_MLP_DEVELOPMENT
M6_FLY_GRU_DEVELOPMENT
M7_OPTIONAL_RECURRENT_PLASTICITY
M8_DEVELOPMENT_VALIDATION
M9_PROTOCOL_FROZEN
M10_CONFIRMATORY
M11_CONTROLS_AND_PERTURBATIONS
M12_ANALYSIS_AND_FIGURES
M13_SCIENTIFIC_AUDIT
M14_MANUSCRIPT_AND_HANDOFF
COMPLETE
```

At every restart:

```text
read this file
read V13_PROGRESS.md
continue at first incomplete milestone
```

If MLP fails:
- tune only allowed development parameters;
- then use GRU.

If GRU fails:
- use curriculum/feature diagnostics;
- then consider recurrent plasticity.

Do not loop forever on one hyperparameter.

Maintain a machine-readable log of all development configurations.

---

# 23. EXPECTED MOST-LIKELY PATH BY TASK

## Pong

Likely easiest.

Priority:
1. sensory-only PPO;
2. MLP PPO on fly features;
3. gamma ~0.99 / lambda ~0.95;
4. larger transition budget;
5. 8–12 SNN ticks if useful.

Likely no GRU required.

## Breakout

Priority:
1. sensory-only PPO;
2. Pong/interception curriculum;
3. MLP PPO;
4. GRU if needed;
5. 8–12 ticks.

## Relative-action Snake

Historical Snake already proves the 2,040-neuron substrate can support meaningful Snake learning in an easier action formulation.

For v1.3:
1. MLP PPO because heading × food-direction interaction is nonlinear;
2. GRU if necessary;
3. transition-based budget much larger than v1.2;
4. gamma/GAE increase;
5. 4x4 -> 5x5 -> 6x6 curriculum.

## Flappy

Do not optimize RL until benchmark validation.

1. MPC/beam-search oracle;
2. prove solvability;
3. minimally repair environment only if necessary;
4. sensory-only PPO;
5. MLP PPO;
6. GRU if timing/memory remains limiting;
7. wider-gap curriculum only if needed.

---

# 24. REQUIRED FILES/ARTIFACTS AT COMPLETION

At minimum:

```text
V13_PROGRESS.md

src/flyarcade/v13/__init__.py
src/flyarcade/v13/ppo.py
src/flyarcade/v13/policy.py
src/flyarcade/v13/controller.py
src/flyarcade/v13/study.py
src/flyarcade/v13/representations.py

scripts/v13_validate_envs.py
scripts/v13_develop.py
scripts/v13_freeze.py
scripts/v13_trial.py
scripts/v13_suite.py
scripts/v13_report.py
scripts/v13_representation.py
scripts/v13_replay.py
scripts/v13_audit.py
scripts/v13_finalize_docs.py

experiments/v13_development_plan.json
experiments/v13_run_plan.json
experiments/METHODS_V13.md

tests/test_v13_ppo.py
tests/test_v13_policy.py
tests/test_v13_environments.py
tests/test_v13_study.py
tests/test_v13_reproducibility.py

artifacts/v13/development/
artifacts/v13/primary.csv
artifacts/v13/contrasts.csv
artifacts/v13/perturbations.csv
artifacts/v13/representations.csv
artifacts/v13/task_specific.csv
artifacts/v13/results_summary.json
artifacts/v13/scientific_audit.json
artifacts/v13/verification.json
artifacts/v13/figures/
```

Names may vary if repository style requires it, but equivalent artifacts must exist.

---

# 25. FIGURES

Generate publication-ready SVG + PNG.

At minimum:

1. six-task performance:
   - historical Catch/Dodge;
   - v1.3 four-task results;
   - biological;
   - rewired;
   - random;
   - heuristic;
   - sensory-only;
   - individual training-seed points.

2. v1.3 learning curves with **transitions** on x-axis.

3. learning-capacity ladder:
   - v1.2 linear;
   - v1.3 MLP;
   - v1.3 GRU;
   - recurrent plasticity if used.

4. biological vs rewired.

5. perturbations.

6. representation diagnostics.

No misleading y-axis truncation.

---

# 26. MANUSCRIPT FRAMING

Never write:

> the fly brain learned six games.

Preferred language:

> a spiking neural network derived from the Drosophila MaleCNS connectome provided a recurrent neural substrate for learned artificial sensorimotor behavior.

If v1.3 succeeds broadly, likely central message:

> The capacity of a fixed connectome-derived neural substrate depends strongly on the learning mechanism used to read out its activity: simple linear reward learning was sufficient for some behaviors, whereas harder tasks required nonlinear and/or temporally recurrent decoding.

If recurrent synapses become plastic, distinguish those results explicitly.

Always distinguish:
- measured connectivity;
- engineered SNN;
- engineered sensory encoding;
- artificial decoder;
- artificial task.

Preserve v1.2 failures in the story because they motivate v1.3.

---

# 27. SCIENTIFIC AUDIT

Before completion, `scripts/v13_audit.py` must verify:

- branch/version;
- graph hash;
- run-plan hash;
- code hash;
- expected trial count;
- all trials complete;
- development/confirmatory seed separation;
- historical seed separation;
- protected historical file hashes;
- Stage A recurrent core unchanged;
- checkpoint tensors finite;
- optimizer state restorable;
- exact checkpoint replay per task;
- perturbation replay on representative runs;
- at least one fresh full-budget deterministic rerun where practical;
- no learning during evaluation;
- random/heuristic/sensory-only reproducibility;
- every reported table regenerated from raw result files;
- every figure traceable to results;
- manuscript numbers traceable to artifacts.

Machine-readable status:

```json
{"status": "PASS"}
```

Do not emit PASS if an invariant fails.

---

# 28. COMPLETION CHECKLIST

Before declaring `COMPLETE`:

- [ ] correct v1.3 branch;
- [ ] v1/v1.1/v1.2 historical files preserved;
- [ ] Flappy solvability gate passed;
- [ ] sensory-only learner solves all four on development data;
- [ ] nonlinear PPO learner implemented;
- [ ] transition-based budgets used;
- [ ] long-horizon gamma/GAE developed;
- [ ] ticks-per-action evaluated;
- [ ] MLP/GRU ladder used as needed;
- [ ] all four tasks pass development gate;
- [ ] protocol frozen before confirmatory;
- [ ] fresh confirmatory seeds run exactly once under frozen protocol;
- [ ] outcomes reported honestly;
- [ ] controls complete;
- [ ] perturbations complete;
- [ ] representation analysis complete;
- [ ] full test suite passes;
- [ ] scientific audit PASS;
- [ ] figures generated;
- [ ] manuscript generated from artifacts;
- [ ] `STATE.md`, `HANDOFF.md`, `RUNBOOK.md`, `CHANGELOG.md`, `DECISIONS.md` updated;
- [ ] reproduction commands tested;
- [ ] no active background jobs;
- [ ] no credentials stored;
- [ ] `V13_PROGRESS.md` = `COMPLETE`.

---

# 29. WHAT COUNTS AS SUCCESS

## Scientific success

Mandatory:
- reproducible;
- honest;
- controlled;
- development/confirmatory separation;
- complete audit;
- no historical corruption.

## Intended behavioral target

Make a serious development effort toward:

```text
Flappy    SUCCESS
Pong      SUCCESS
Breakout  SUCCESS
Snake     SUCCESS
```

But **never manufacture this result**.

Do not say all four succeeded unless frozen confirmatory artifacts meet the predefined criteria.

A clean confirmatory failure is preferable to contaminated success.

---

# 30. FINAL RESPONSE TO THE USER

At completion report:

1. final branch;
2. final commit SHA;
3. what changed from v1.2;
4. selected architecture per task;
5. confirmatory success/failure for Flappy/Pong/Breakout/Snake;
6. biological/frozen/random/rewired/heuristic/sensory-only performance;
7. whether recurrent fly plasticity was required;
8. whether more neurons were required;
9. perturbation results;
10. representation findings;
11. full test count/status;
12. scientific audit status;
13. reproduction commands;
14. important limitations;
15. manuscript-ready files.

---

# 31. START NOW

Immediately:

1. verify the completed v1.2 branch;
2. create/switch to `flyarcade-v1.3-adaptive-learning`;
3. snapshot historical hashes;
4. create `V13_PROGRESS.md`;
5. run the entire current test suite;
6. validate Flappy with a competent oracle;
7. implement sensory-only PPO;
8. implement MLP/GRU fly-feature PPO;
9. continue through the milestone state machine without stopping prematurely;
10. re-read this file after every major milestone and continue until `V13_PROGRESS.md` says `COMPLETE`.
