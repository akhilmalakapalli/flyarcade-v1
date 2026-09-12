# Planned v1 acceptance criteria

This protocol is inferred from the supplied partial master. Detailed frozen methods
are in METHODS.md and run_plan.json. M0–M3 are implemented and tested; M4 trials are running.

The primary biological dataset is **HHMI Janelia MaleCNS v1.0**, accessed through
`neuprint-python` at `https://neuprint.janelia.org`, dataset `male-cns:v1.0`.
FlyWire is superseded. The authenticated 2,040-neuron subgraph is acquired and
audited. See `DATA.md` for provenance and reproduction.
Target 1,000–4,096 neurons (the intersection of the requested approximate
1,000–5,000 range and the unchanged hard neuron cap). Use annotation-guided
selection and induced adjacency queries, not a bulk graph download.

| Milestone | Acceptance evidence | Status |
| --- | --- | --- |
| M0 bootstrap | Isolated install, resource guards, passing health/lint/tests | Complete |
| M1 connectome | Exact IDs, provenance, sparse import, topology controls, real subnetwork | Complete: 2,040 neurons, 126,676 edges |
| M2 spiking controller | Tested LIF dynamics, reset/refractory behavior, bounded reward-modulated eligibility updates, sensory/motor interfaces | Implemented; validation/results below |
| M3 arcade tasks | At least two deterministic headless tasks, action/observation contracts, random and heuristic baselines | Implemented; validation/results below |
| M4 experiments | Train/eval separation, matched controls, multiple seeds, transfer and robustness protocols, resumable checkpoints | Not started |
| M5 scientific report | Measured learning and uncertainty, resource benchmarks, reproducibility audit, completed manuscript | Not started |

The primary biological comparison should hold neuron count, input/output
assignments, weight initialization rule, training budget and random seeds fixed
between the derived topology and directed degree-preserving rewires. Include
frozen-plasticity and random-action controls. The implemented rewiring preserves
outgoing weighted strength and both directed degree sequences, but not incoming
weighted strength; measure/report that imbalance. Report swap acceptance and
graph overlap; zero accepted swaps is not a randomized control.

Predeclare separate environment seeds for training, development and held-out
evaluation. Evaluation must freeze learning and reset episode state. Pair seeds
across conditions, aggregate uncertainty at the independent training-seed level,
and retain negative results. Do not tune on test seeds. Transfer must compare
source pretraining against equal-budget target-only training and fresh starts.
Robustness conditions should include sensory noise and neuron/edge ablation with
fixed masks independent of reward. These designs still need concrete parameters
before experiments begin.
