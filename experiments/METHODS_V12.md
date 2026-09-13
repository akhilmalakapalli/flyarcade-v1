# Four-task extension: methods and interpretation

The canonical biological dataset remains HHMI Janelia MaleCNS `male-cns:v1.0`.
The original 2,040-neuron, 126,676-edge subgraph is reused without reselection.
See DATA.md for anatomical criteria, body IDs, raw query provenance and boundary
loss. This is a limited connectome-derived spiking model, not an intact fly brain.

## Architecture

Stage A freezes every recurrent synaptic magnitude. Visual-projection neurons
receive deterministic round-robin assignments of engineered game features and
their complements. Descending-neuron activity drives the artificial actor–critic.
Four LIF ticks per action, the existing transmitter/sign assumptions, background
current and stimulus scaling are inherited unchanged from the historical v1.1
controller. No recurrent learning is claimed. Actions are artificial outputs.

The v1.2 fixed CSR executor implements the same signed sparse recurrence as the
historical edge-list executor. Synthetic automated tests and authentic biological
and rewired graph checks compare spikes and voltages exactly, with lesions.
No historical source code was changed. The locked environment supplies SciPy.

Each task/topology receives its own standardizer: 24 development-only episodes
under alternating heuristic/random actions. Mean, scale floor 0.02, clipping 4
and division by square root of readout width follow v1.1. Biological statistics
are never substituted for rewired statistics. Fit coverage is limited: 353 Flappy,
3,456 Pong, 3,840 Breakout and 283 Snake states per graph.

## Environments

Exact executable definitions are `src/flyarcade/v12/environments.py`, covered by
the frozen source hash. Machine-readable definitions and reward summaries are in
`artifacts/v12/environment_definitions.json` and `experiments/v12_run_plan.json`.
Observations are bounded engineered variables, not biological visual inputs.
Pong and Breakout use a derived reflected-ball intercept estimate, never a correct
action label. Snake uses immediate relative danger, food direction, heading and
wall distances; it receives no globally optimal route.

| Task | Features | Actions | Horizon | Primary score |
|---|---:|---:|---:|---|
| Flappy | 6 | 2 | 138 | pipes passed / 6 |
| Pong | 7 | 3 | 144 | hits / attempted returns |
| Breakout | 11 | 3 | 160 | bricks destroyed / 5 |
| Snake | 17 | 3 relative | 120 | min(food / 10, 1) |

Raw metrics remain in the episode records and task-specific CSV. Scores share
bounds, not a common unit of competence. New Snake uses a 6×6 board with relative
left/straight/right actions and 40-step starvation. Historical 8×8 absolute-action
Snake is a separate preserved experiment, not replaced or pooled with v1.2.

## Development, freeze and confirmation

The development plan allowed exactly two 300-episode candidates per task, one
training seed and 20 development evaluation episodes. Selection maximized mean
success; exact ties selected candidate zero. Flappy selected zero, the others one.
Neither Flappy candidate scored. No tuning followed confirmatory evaluation.
The run plan freezes all actor–critic parameters, task budgets, source hashes,
seed rules, reward summaries and prospective success criteria.

Three independent training environment/RNG seeds per condition, 40 held-out
episodes per seed, with paired seed blocks across biological, frozen, rewired,
random and heuristic conditions. Initial zero readout and fixed biological core
are intentionally identical; independence comes from training experience and
exploration. Training budgets are 1,200 / 600 / 600 / 2,000 episodes respectively.
The no-learning arm receives the same episode budget; greedy ties choose action
zero. All new scientific seed blocks exceed 100 million and are disjoint from
historical, fitting, development, evaluation and representation blocks.

Success requires every biological seed to improve over its own before-training
score, exceed its paired random score by more than 0.05, and produce action
diversity above 0.05 under fixed-history observation interventions. The latter
resets neural state and RNG identically while changing inputs, distinguishing
input sensitivity from mere trajectory action variability. It is a limited probe,
not a complete characterization of policy state dependence. Failures are retained.

## Controls and perturbations

Directed double-edge swaps use 10 attempts per edge and seeds 700–702. They
preserve neuron IDs, in/out degrees, edge count, outgoing count assignments and
global count distribution. Incoming strength, other biological properties and
uniform graph sampling are not guaranteed. Each graph is calibrated separately.

After learning, all biological policies receive greedy no-learning evaluation
under intact conditions, Gaussian observation noise SD 0.1 clipped to [0,1],
10% Bernoulli edge ablation and 10% Bernoulli neuron ablation. Masks are fixed
per task/training seed and never retrained. These tests are reported even for
failed policies, whose perturbation interpretation is limited.

## Analysis

Primary uncertainty resamples three independent training-seed means 10,000 times
(seed 2026). The 95% intervals are descriptive; no p-values are computed. Contrasts
pair seeds. Historical Catch/Dodge rows are read from unchanged v1.1 results;
they have different tasks and training budgets and are not a newly retuned grid.

Post-hoc neural analyses use 20 separate episodes and matched alternating
heuristic/random actions across all graphs. Up to 24 evenly spaced states per
episode are retained. The first ten whole episodes train a ridge decoder (penalty
1, intercept); the remaining ten test heuristic-action decoding. Majority-action
accuracy is reported. Decoder labels are analysis targets, never controller inputs.
Features use the frozen development standardizer. Raw rates determine participation
ratio and mean population variance; correlations use up to 128 active descending
neurons. State hashes verify matching across topologies. Biological activity is
deterministic and identical across these model-seed replicas, so these are not
three independent biological preparations. Rewired replicas change topology.
No representation result is used for selection or causal claims about task success.

## Reproducibility and resources

The scientific audit verifies all 36 new trials, histories, hashes, seed separation,
frozen weights, degree invariants, all four seed-zero checkpoint evaluations and
perturbations, non-overwrite behavior, and a full-budget Flappy rerun. Historical
files are protected by a 467-file hash snapshot, including existing ignored runs
and data. Published checkpoint/result identifiers are retained in the new archive.

Unchanged guards: 2 GiB RSS snapshot, 120 seconds per operation, 1 GiB free disk,
128 MiB acquisition, 4,096 neurons, 250,000 edges. Trials checkpoint at episode or
evaluation boundaries after 70 seconds and resume in separate processes, at most
20 segments. This follows the historical bounded resumable-run policy. Guards
are cooperative, and RSS values are snapshots rather than measured peaks.

Timing/RSS fields and compressed checkpoint file hashes identify the saved run;
those incidental bytes need not match a fresh execution. Deterministic scientific
reproduction compares complete training/evaluation rows and learned arrays.
Historical Catch/Dodge checkpoint replay is separately recorded for all 18 trials
in artifacts/v12/historical_replay.json, without modifying original artifacts.
The Flappy heuristic itself achieves limited pipe coverage; it is a reference
policy, not evidence that this setup can routinely complete all six obstacles.
