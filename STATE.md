# flyarcade-v1 State

## Overall status
COMPLETE on branch `flyarcade-v1.1-learning`. Three studies finished: v1 (negative
baseline, frozen at commit ec27ad0), v1.1 catch/dodge (positive), Snake (positive).
The final manuscript is written. v1 must not be altered.

## Current milestone
None open. Snake was the final major experiment. Remaining work is optional
follow-up listed in HANDOFF.md.

## Completed
- v1 (ec27ad0): 36-trial preregistered grid, negative result, full-grid bit-identical
  replay, scientific audit PASS. Untouched.
- v1.1 catch/dodge: reward-guided actor-critic over the unchanged MaleCNS circuit.
  catch 0.206 -> 0.661 vs 0.197 random; dodge 0.794 -> 0.926 vs 0.803. 3/3 seeds.
- Snake: 8x8 grid, food, wall/self collision, growing body, 4 actions, reversal
  refused, 21 hand-specified channels, bounded reward (+1 food, -1 death, -0.01 step).
  Development on new 7,000,000-block seeds; frozen plan; confirmatory on 8,000,000 /
  9,000,000 blocks.
  food 0.100 -> 1.483 vs 0.150 random (+1.333 [1.050, 1.625], 3/3 seeds);
  steps 5.0 -> 29.85; state-dependence 0.712; entropy 0.425 of a 1.386 maximum.
  Heuristic reference 17.258, so competence is 0.078 of the way from random to
  heuristic and the learning curve was still rising at the frozen budget.
- Snake perturbations: food 1.483 intact, 1.242 sensory noise, 0.417 10% edge,
  0.442 / 0.308 / 0.100 at 5 / 10 / 25 percent neuron ablation. Graded dose-response.
- Topology across all three tasks: biological never exceeds degree-rewired.
  Normalised biological/rewired: catch 0.578/0.772, dodge 0.627/0.683, snake
  0.078/0.104. Only catch resolves from zero. No single conclusion is forced.
- Caught and reported a preprocessing artefact: standardising the rewired Snake arm
  with biological statistics made it look as though rewiring abolished learning.
- Cross-task analysis, Snake figures/tables, demonstration figure and GIF.
- paper/manuscript.md rewritten as the final paper.

## In progress
- Nothing. No background process, download or training job remains.

## Next
1. Optional: owner selects a LICENSE; it still reserves rights.
2. Optional follow-ups are listed in HANDOFF.md and all require new preregistrations.

## Tests
- sh scripts/verify.sh: PASS (health, Ruff lint/format, 80 tests, benchmark).
- scripts/replay_all.py: v1 36/36 MATCH. scripts/scientific_audit.py: PASS.
- v1.1 and Snake determinism verified by delete-and-re-run.
- README SHA256 unchanged: a46572881f00b6963a0bbddcf2cc439d7acfd1ca46158711ad1a6fe8caaa7769.

## Latest benchmark
- Final firing rate is near 0.21 spikes per neuron per tick on all three tasks
  (catch 0.2167, dodge 0.2173, snake 0.2085).
- Lesion sensitivity tracks task difficulty: fraction of learned gain retained under
  10% neuron ablation is 1.10 dodge, 0.28 catch, 0.12 snake.

## Resource status
- RAM: 16 GiB physical; unchanged 2 GiB RSS guard; peak trial snapshot well under it.
- disk: ~340 GiB free; unchanged 1 GiB reserve.
- runtime: unchanged 120 s per independently guarded process. Snake trials checkpoint
  and resume in fresh processes rather than raising the guard. No limit was raised in
  any study.
- graph/download: unchanged 4,096 neurons / 250,000 edges / 128 MiB / 32 MiB NPZ.

## Known issues
- The recurrent MaleCNS core never learned: every positive result uses stage A with
  connectome-derived weights frozen. Learning lives in an artificial readout.
- Snake competence is modest (0.078 normalised) and budget-bound.
- Boundary truncation: 85% incoming / 93% outgoing synaptic weight is missing.
- Three seeds support descriptive intervals only; no confirmatory significance.
- Only acetylcholine and GABA have fast effects.
- The rewired control does not preserve incoming strength; mixing is unvalidated.
- Cooperative resource checks do not provide OS-level isolation.
- LICENSE reserves rights pending owner choice.

## Last agent
Claude Code

## Timestamp
2026-09-12T23:35:48.827119+00:00
