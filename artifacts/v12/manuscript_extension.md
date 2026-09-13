

---

# v1.2 extension: four further artificial sensorimotor tasks

This extension preserves the historical report above verbatim. Catch and Dodge are historical v1.1 comparisons, not retuned experiments. The new relative-action 6×6 Snake differs from the historical absolute-action 8×8 Snake.

## Methods

The same authentic 2,040-neuron MaleCNS v1.0 subnetwork and frozen recurrent spiking substrate drive a learned artificial actor–critic readout. Engineered observations and game actions are not biological sensory/motor interfaces. No recurrent synapse learns. The source graph was not reselected.

A bounded two-candidate development search preceded the frozen v1.2 protocol. Three independent training experience/exploration seeds per condition and 40 held-out episodes per seed were used. The task-specific budgets, reward definitions, complete seed rules and source hashes are in `experiments/v12_run_plan.json`; detailed methods are in `experiments/METHODS_V12.md`. Every topology has its own development-fitted standardizer. No confirmatory score was used for tuning.

Success was defined prospectively: every biological seed must improve over its initial held-out score, beat paired random by more than 0.05, and show fixed-history observation-intervention action diversity above 0.05. Weight changes alone do not establish learning success.

## Held-out results

| Task | Biological | Frozen | Rewired | Random | Heuristic | Change | Bio−random | Bio−rewired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catch | 0.661 | 0.206 | 0.817 | 0.197 | 1.000 | 0.456 | 0.464 | -0.156 |
| dodge | 0.926 | 0.794 | 0.938 | 0.803 | 1.000 | 0.132 | 0.124 | -0.011 |
| flappy | 0.000 | 0.000 | 0.000 | 0.000 | 0.087 | 0.000 | 0.000 | 0.000 |
| pong | 0.350 | 0.301 | 0.489 | 0.253 | 1.000 | 0.049 | 0.097 | -0.139 |
| breakout | 0.368 | 0.365 | 0.402 | 0.468 | 0.617 | 0.003 | -0.100 | -0.033 |
| snake | 0.043 | 0.005 | 0.062 | 0.025 | 0.928 | 0.037 | 0.017 | -0.019 |


Entries are means across three training seeds. The task-defined scores share [0,1] bounds, not common behavioral units: pipe fraction, return fraction, brick fraction, and capped food/10 for new Snake. Catch/Dodge retain their original success fractions. Descriptive 95% bootstrap intervals and paired contrasts are available in `artifacts/v12/results_summary.json`; no p-values or population-level claims are made.

**Flappy did not meet the frozen success criteria** (0/3 seeds met all components). Before 0.000, after 0.000 [0.000, 0.000]; seed scores 0.000, 0.000, 0.000. Fixed-history state-probe diversity averaged 0.000.

Mean raw biological outcomes: obstacles passed 0.000, steps 20.375.

**Pong did not meet the frozen success criteria** (2/3 seeds met all components). Before 0.301, after 0.350 [0.308, 0.392]; seed scores 0.308, 0.350, 0.392. Fixed-history state-probe diversity averaged 0.283.

Mean raw biological outcomes: .

**Breakout did not meet the frozen success criteria** (0/3 seeds met all components). Before 0.365, after 0.368 [0.365, 0.370]; seed scores 0.370, 0.370, 0.365. Fixed-history state-probe diversity averaged 0.192.

Mean raw biological outcomes: bricks destroyed 1.842, misses 3.000.

**Snake did not meet the frozen success criteria** (0/3 seeds met all components). Before 0.005, after 0.043 [0.035, 0.053]; seed scores 0.040, 0.053, 0.035. Fixed-history state-probe diversity averaged 0.433.

Mean raw biological outcomes: food 0.425, steps 39.017.

## Topology and perturbations

The degree-preserving rewired control preserves in/out degree and outgoing count assignments, but not all biological properties or incoming strength; it is not a uniformly sampled random graph. These comparisons concern the artificial task, input mapping and learned readout used here.

| Task | Bio−rewired | Intact | Noise SD 0.1 | Edge 10% | Neuron 10% |
|---|---:|---:|---:|---:|---:|
| catch | -0.156 | 0.661 | 0.453 | 0.617 | 0.329 |
| dodge | -0.011 | 0.926 | 0.914 | 0.924 | 0.939 |
| flappy | 0.000 | 0.000 | 0.001 | 0.004 | 0.017 |
| pong | -0.139 | 0.350 | 0.356 | 0.340 | 0.340 |
| breakout | -0.033 | 0.368 | 0.367 | 0.442 | 0.382 |
| snake | -0.019 | 0.043 | 0.044 | 0.020 | 0.018 |

Perturbations used fixed masks with no retraining. Their interpretation is limited when the intact policy fails the behavioral criteria. Neither resilience of a constant policy nor lesions that improve a poor policy demonstrate biologically meaningful robustness.

## Post-hoc neural representations

Matched alternating heuristic/random behavior produced identical sampled states across topologies. A ridge probe predicted heuristic-action labels using separate whole episodes for training and testing. The labels were never controller inputs. Raw activity supplied participation ratio, correlation and variance. Biological feature replicas are deterministic duplicates across these model seeds; they are not independent biological samples. Rewired replicas change graph topology.

| Task | Bio probe | Rewired probe | Majority | Bio PR | Rewired PR |
|---|---:|---:|---:|---:|---:|
| flappy | 0.955 | 0.955 | 0.955 | 19.554 | 32.145 |
| pong | 0.442 | 0.475 | 0.379 | 35.531 | 66.413 |
| breakout | 0.362 | 0.419 | 0.346 | 34.865 | 69.064 |
| snake | 0.491 | 0.497 | 0.509 | 9.700 | 18.857 |

Probe accuracy must be interpreted against the majority baseline. These post-hoc summaries do not establish a causal explanation of learning or a general ranking of biological and rewired circuitry.

## Limitations and reproducibility

Only about 15.05% of incoming and 7.12% of outgoing synaptic weight is retained in this selected circuit. Artificial background drive, simplified LIF dynamics, transmitter-sign assumptions and engineered inputs constrain interpretation. Development was deliberately limited to two candidates; Flappy and Snake feature-fit state coverage was small. Failure at these budgets is a result about this protocol, not proof of impossibility.

The scientific audit verified 36 new trials and preserved 467 historical file hashes. It replayed clean and all three perturbation evaluations from representative checkpoints for every new task and repeated one full-budget Flappy training history exactly. All Stage A recurrent and frozen-control invariants passed. Maximum trial segment duration was 46.89 seconds; maximum recorded trial RSS snapshot was 136.5 MiB. These are cooperative guards and memory snapshots, not measured peaks.

Reproduction commands are in `RUNBOOK.md`. Machine-readable primary, contrast, perturbation, raw-task and representation tables, complete result archives and five PNG/SVG figures are under `artifacts/v12/`. Checkpoints and the biological data remain in the local ignored run/data directories.
