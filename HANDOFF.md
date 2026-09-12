# Handoff — FlyArcade-v1 complete (branch `flyarcade-v1.1-learning`)

**All three studies are finished and the final manuscript is written.** v1 is frozen
at commit `ec27ad0` and must not be altered, rewritten or reinterpreted. No
background process, download or training job is running. Do not edit README.md. Do
not raise any resource limit.

## Read first
`FLYARCADE_MASTER.md`, `STATE.md`, `DECISIONS.md` (now through D027),
`experiments/METHODS.md`, the three preregistrations (`experiments/run_plan.json`,
`v11_run_plan.json`, `snake_run_plan.json`), then `paper/manuscript.md`.

## The three studies
1. **v1 (negative, frozen).** Naive reward-modulated plasticity did not improve
   held-out performance on catch or dodge and collapsed the policy onto a constant
   action, while the circuit demonstrably carried decodable task information.
2. **v1.1 (positive).** A reward-guided actor-critic mechanism plus two feature
   conditioning fixes: catch 0.206 → 0.661 vs 0.197 random, dodge 0.794 → 0.926 vs
   0.803, 3/3 seeds.
3. **Snake (positive).** Same architecture, four documented task-specific parameters.

## Snake headline (confirmatory seeds, never used for tuning)
| Condition | Food | Steps | Return | State-dep. |
|---|---|---|---|---|
| actor-critic | **1.483** (1.150, 1.750, 1.550) | 29.85 | +0.202 | 0.712 |
| frozen | 0.100 | 5.00 | -0.950 | 0.000 |
| rewired | 1.933 (1.875, 1.200, 2.725) | 33.37 | +0.600 | 0.691 |
| random | 0.150 | 10.57 | — | — |
| heuristic | 17.258 | 112.32 | — | — |

Contrasts: vs frozen **+1.383** [1.050, 1.650]; vs random **+1.333** [1.050, 1.625],
3/3 seeds; vs rewired -0.450 [-1.175, 0.550]. All three preregistered criteria met.

**Competence is modest and must stay stated that way:** 0.078 of the way from random
to heuristic, with the learning curve still rising at the frozen 4,000-episode
budget. Do not extend the budget and re-report it as the same experiment.

## Two things a successor must not undo

**1. Per-graph standardisation.** An early Snake run standardised the rewired arm
with the *biological* circuit's statistics and made it look as though rewiring
abolished Snake learning entirely (0.100 / 0.075 / 0.100 food). Per-graph calibration
raised the same arm to 1.875 / 1.200 / 2.725. The whole apparent topology effect was
readout calibration. Every topology comparison must calibrate each graph by the same
procedure applied to its own activity. This is D025 and it is reported in the paper,
not quietly corrected.

**2. No single topology conclusion.** Biological never exceeds rewired on any task,
but only catch resolves from zero. Normalised biological/rewired: catch 0.578/0.772,
dodge 0.627/0.683, snake 0.078/0.104. Report the consistent direction; do not claim a
uniform effect, and do not reframe it in either direction.

## Cross-task findings
- One frozen MaleCNS recurrent core supports learning on all three tasks (stage A
  throughout; the connectome-derived weights never learned).
- Lesion severity tracks task difficulty. Fraction of learned gain retained under 10%
  neuron ablation: dodge 1.10, catch 0.28, Snake 0.12. Snake shows a graded
  dose-response (5/10/25% → 0.442/0.308/0.100 food) down to its random baseline.
- Firing statistics barely differ across tasks: 0.2167, 0.2173, 0.2085 spikes per
  neuron per tick.
- **Do not call perturbation insensitivity robustness.** Dodge sits near ceiling; v1
  showed the same lesions on a collapsed policy meaning the opposite thing.

## Verification

```sh
sh scripts/verify.sh                          # health, lint, format, 80 tests, benchmark
.venv/bin/python scripts/replay_all.py        # v1 only: must report 36/36 MATCH
.venv/bin/python scripts/scientific_audit.py  # v1: must report PASS
.venv/bin/python scripts/v11_summarize.py
.venv/bin/python scripts/snake_run_suite.py   # skips completed trials
.venv/bin/python scripts/snake_summarize.py
.venv/bin/python scripts/cross_task_analysis.py
.venv/bin/python scripts/snake_visualize.py --trial runs/snake-eprop-1 --gif
shasum -a 256 README.md                       # must stay a4657288...caaa7769
```

`replay_all.py` matches the v1 naming convention positively; v1.1 and Snake use
different checkpoint formats and were verified by delete-and-re-run instead.

## Resource posture
Unchanged guards throughout all three studies: 4,096 neurons, 250,000 edges, 2 GiB
RSS, ≥1 GiB free disk, 120 s per independently guarded process. Snake trials
checkpoint at episode boundaries and resume in a **fresh** process rather than
resetting a guard mid-operation (D024). **No limit was raised at any point.**
Background processes are descheduled by the host, so trust the guard-measured
`elapsed_seconds` in each artifact, not wall-clock.

## If you continue — all require a NEW preregistration
1. **Stage B/C recurrent plasticity.** Implemented and unit-tested, never used in a
   confirmatory setting. This is the most interesting open question: every positive
   result here comes from an artificial readout over a *frozen* connectome circuit.
2. **A longer Snake budget.** The curve was still rising; a larger frozen budget is a
   new experiment, not a continuation of this one.
3. **Better rewiring null models** — preserve incoming strength, or preserve
   clustering — to isolate which graph property costs effective dimensionality.
4. **A nonlinear readout**, to test whether the biological code's lower *linear*
   dimensionality is a real capacity limit or an artefact of linear decoding.
5. **More seeds.** Three support descriptive intervals only.

## Working tree
Branch `flyarcade-v1.1-learning` off `ec27ad0`. `data/`, `runs/`, `cache/` and
`.venv` are ignored; `.secrets/` is ignored and has never been staged. LICENSE still
reserves rights pending the owner's choice.
