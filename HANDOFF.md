# Handoff — branch `flyarcade-v1.1-learning`

**v1 is frozen at commit `ec27ad0` and must not be altered, rewritten or
reinterpreted.** It remains the negative baseline. v1.1 changes the learning rule
only and has a confirmatory positive result. Do not edit README.md. Do not raise any
resource limit. No background process, download or training job is running.

## Read first
`FLYARCADE_MASTER.md`, `STATE.md`, `DECISIONS.md` (now through D022),
`experiments/METHODS.md`, `experiments/v11_run_plan.json`, then `paper/manuscript.md`
(Part I is v1, Part II is v1.1).

## What v1.1 did
Replaced v1's global reward-modulated STDP with an actor-critic e-prop three-factor
rule over the **same** MaleCNS topology, circuit, LIF dynamics, sign rule, sensory
encoding, tasks and reward. `tests/test_v11.py` asserts a frozen v1.1 core
reproduces v1's spike trains bit-for-bit, so behavioural differences are
attributable to credit assignment, not dynamics.

Two conditioning repairs were necessary and are frozen preprocessing, not tuned
parameters: removing the descending population's state-independent common mode, and
dividing standardised features by sqrt(1293) so learning rates are not effectively
a thousand times too large.

## Confirmatory result (seeds never used for tuning)
| Task | Before | After | Random | Over random | Seeds |
|---|---|---|---|---|---|
| catch | 0.206 | **0.661** | 0.197 | **+0.464** [0.408, 0.563] | 3/3 |
| dodge | 0.794 | **0.926** | 0.803 | **+0.124** [0.100, 0.158] | 3/3 |

All three preregistered success criteria met. Policy stays state-dependent (0.44 /
0.40); the frozen arm reproduced its initial score exactly. **Stage A only** — the
recurrent MaleCNS core stayed entirely frozen, so per the protocol the search
stopped at the simplest model that genuinely learns. Stages B and C are implemented
and unit-tested but unused and support no claim.

## The topology result went against the biological graph
With identical rule, budget, neuron count, readout and per-graph standardisation:

- catch: e-prop − rewired = **-0.156** [-0.325, -0.021], rewired better on 3/3 seeds.
- dodge: e-prop − rewired = -0.011 [-0.079, 0.071], indistinguishable.

Post-hoc explanation (`artifacts/v11_topology_diagnostic.json`): the rewired
descending code has participation ratio 54.5 vs 29.5, mean absolute pairwise
correlation 0.112 vs 0.152, and linear probe accuracy 0.901 vs 0.774. Decorrelation
gives a linear readout more independent directions.

**Report this in the direction it came out.** It is not a claim that the biology is
worse: the readout is an artificial linear decoder on a circuit missing 85% of its
incoming synaptic weight, and correlated structure may serve functions this task
cannot measure. Do not re-run or reframe it to soften the direction.

## Lesions
Repeated only once state-dependent learning existed. catch: 0.661 unperturbed →
0.453 sensory noise → 0.617 edge ablation → **0.329** neuron ablation. dodge is
near-ceiling and unaffected. **Do not call perturbation insensitivity robustness**;
v1 showed the same lesions on a collapsed policy meaning the opposite thing.

## Verification

```sh
sh scripts/verify.sh                                   # 68 tests, lint, format, health
.venv/bin/python scripts/replay_all.py                 # v1 only: must report 36/36 MATCH
.venv/bin/python scripts/scientific_audit.py           # v1: must report PASS
.venv/bin/python scripts/v11_fit_features.py           # refits the frozen standardiser
.venv/bin/python scripts/v11_run_suite.py              # skips existing results
.venv/bin/python scripts/v11_summarize.py              # tables + 3 figures
.venv/bin/python scripts/v11_topology_diagnostic.py
shasum -a 256 README.md                                # must stay a4657288...caaa7769
```

v1.1 determinism was demonstrated by deleting `runs/v11-*` and re-running the whole
suite: every confirmatory number reproduced exactly. `replay_all.py` is scoped to v1
trials because v1.1 checkpoints use a different format.

## Resource posture
Unchanged guards throughout: 4,096 neurons, 250,000 edges, 2 GiB RSS, 120 s per
independently guarded trial. v1.1 confirmatory trials peak at ~34 s; development
trials at ~22 s. **No limit was raised at any point.** Background runs are
descheduled by the host, so wall-clock far exceeds guarded CPU time; trust the
guard-measured `elapsed_seconds` in each artifact, not wall-clock.

## If you continue
1. Stage B/C are implemented but untested in a confirmatory setting. Enabling
   recurrent plasticity would need a **new** preregistration, not edits to
   `experiments/v11_run_plan.json`.
2. The topology finding deserves a dedicated study: vary the rewiring null model
   (preserve incoming strength too, preserve clustering) to isolate which graph
   property costs effective dimensionality.
3. More seeds. Three support descriptive intervals only.
4. A nonlinear readout would test whether the biological code's lower linear
   dimensionality is a real capacity limit or an artefact of linear decoding.

## Working tree
Branch `flyarcade-v1.1-learning` off `ec27ad0`. `data/`, `runs/`, `cache/`, `.venv`
ignored; `.secrets/` ignored and never staged. LICENSE still reserves rights.
