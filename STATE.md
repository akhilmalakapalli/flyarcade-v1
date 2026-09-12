# flyarcade-v1 State

## Overall status
COMPLETE — v1 finished with a negative primary result, fully documented.

## Current milestone
M5 complete. Acquisition, dynamics, games, the frozen experiment grid, controls,
ablations, robustness, transfer, replay verification, figures, CSV tables and the
manuscript are all done. No milestone remains open.

## Completed
- M0-M3 retained and reverified; no bootstrap or acquisition restart.
- Authentic MaleCNS v1.0 subgraph: 2,040 neurons, 126,676 directed edges,
  1,054,402 synapses, no isolates; 218 visual-projection, 1,293 descending,
  529 central-complex intrinsic. All 1,293 descending neurons are reachable from
  visual-projection neurons within 1-2 directed hops (735 at one, 558 at two).
- Boundary loss documented everywhere it matters: 15.05% incoming and 7.12%
  outgoing synaptic weight retained.
- All 36 preregistered trials COMPLETE: 2 tasks x 4 conditions x 3 seeds, plus
  100-episode and both transfer directions.
- Controls finished: frozen, degree-rewired, readout-only.
- Robustness finished: sensory noise sd0.1, 10% edge ablation, 10% neuron ablation.
- Transfer finished: both directions against equal-budget and equal-exposure arms.
- Reproducibility verified: all 36 trials replayed from saved checkpoints,
  144 evaluation sets, bit-identical, zero mismatches (artifacts/replay_check.json).
- Scientific audit PASS: 36 result files hash-verified, seed disjointness and
  frozen/readout-only invariants confirmed.
- Post-hoc mechanism diagnostics added for both tasks (matched state distribution).
- Figures: learning_curves, heldout_results, robustness, transfer, policy_collapse,
  connectome (PNG + SVG). CSV tables: 8 files in artifacts/tables/.
- paper/manuscript.md rewritten around the measured outcome.
- README.md untouched; no resource limit raised.

## In progress
- Nothing. No background process, download or training job remains.

## Next
1. Optional: owner selects a LICENSE; it still reserves rights.
2. Optional: commit the working tree (nothing is committed yet beyond 05b0ee7).
3. Optional future work, out of v1 scope: an exploration-preserving variant
   (entropy regularisation or a non-collapsing readout) to test whether the
   available descending signal becomes usable. This would be a NEW preregistration,
   not a retune of the frozen v1 protocol.

## Tests
- sh scripts/verify.sh: PASS; health, Ruff lint/format, 60 tests (0.75 s), benchmark.
- .venv/bin/python -m pip check: PASS.
- scripts/replay_all.py: 36/36 trials MATCH.
- scripts/scientific_audit.py: PASS.
- README SHA256 unchanged: a46572881f00b6963a0bbddcf2cc439d7acfd1ca46158711ad1a6fe8caaa7769.

## Latest benchmark
- Primary result is NEGATIVE and preserved as such.
- Mean paired held-out change: catch -0.0139 [-0.100, 0.108]; dodge -0.0167
  [-0.058, 0.025]. Every condition's interval spans zero.
- Trained held-out success: catch 0.175 vs random 0.194; dodge 0.794 vs random
  0.806. Heuristic scores 1.000 on both, so the tasks are solvable.
- Frozen controls reproduced their pre-training scores exactly (change 0.0000).
- Contrasts all near zero: learning-minus-frozen -0.014/-0.017,
  learning-minus-rewired +0.017/-0.025, learning-minus-readout_only +0.011/-0.039.
- Mechanism: policy entropy collapses 1.098 -> 0.361 (catch) / 0.128 (dodge) nats
  while linear probe accuracy on descending rates holds at 0.654 -> 0.658/0.713
  against a 0.421 majority rate. Constant-action policies score catch 0.13-0.23 and
  dodge 0.77-0.87, bracketing the trained results.
- Perturbations change nothing (<=0.006), which is degeneracy, not robustness.

## Resource status
- RAM: 16 GiB physical; max trial RSS snapshot 183 MiB; unchanged 2 GiB guard.
- disk: ~341 GiB free; unchanged 1 GiB reserve.
- runtime: max single trial 28.7 s against the unchanged 120 s guard;
  632 s total across 36 sequential trials. No guard raised at any point.
- graph/download: unchanged 4,096 neurons / 250,000 edges / 128 MiB / 32 MiB NPZ.

## Known issues
- The primary scientific result is negative. It is intentionally preserved.
- Boundary truncation (85% incoming / 93% outgoing weight missing) is the largest
  threat to any dynamical conclusion.
- Three seeds support descriptive intervals only; no confirmatory claim is made.
- Only acetylcholine and GABA have fast effects; glutamate, dopamine, octopamine,
  serotonin and unclear neurons are structurally present but functionally silent.
- The rewired control does not preserve incoming strength and its mixing is
  unvalidated, bounding the "topology did not matter" reading.
- Cooperative resource checks do not provide OS-level isolation.
- LICENSE reserves rights pending owner choice.
- Nothing is committed; the working tree is untracked and reviewable.

## Last agent
Claude Code

## Timestamp
2026-09-12T19:46:02.700368+00:00
