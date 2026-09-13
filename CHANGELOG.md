## v1.4 — prospective downstream architecture development

Completed matched Linear/MLP-PPO/GRU-PPO development on five unchanged tasks, bounded secondary analyses and fresh validation. Historical data and dashboard remain frozen; no confirmatory testing. 241 tests and the scientific audit pass. See artifacts/v14/final_summary.md.

# Changelog

## 0.4.0 — Snake, cross-task analysis and the final manuscript
- Add Snake (src/flyarcade/v11/snake.py): 8x8 grid, food, wall and self collision,
  growing body, four actions, immediate 180-degree reversal refused, 200-step cap and
  64-step starvation cap. Bounded reward: +1 food, -1 death, -0.01 per step, no
  distance shaping.
- Add 21 hand-specified sensory channels (food direction and distance, heading,
  immediate danger, 3x3 local occupancy). No pixel or CNN input.
- Generalise V11Controller with a pluggable encoder and action count; verified
  bit-for-bit against the frozen v1.1 catch result.
- Add resumable, independently guarded Snake trials that checkpoint at episode
  boundaries and resume in a fresh process rather than raising the 120 s guard.
- Snake development on new 7,000,000-block seeds across three seeds, with rejection
  criteria applied before any performance comparison; freeze
  experiments/snake_run_plan.json with a success criterion stated in advance.
- Snake confirmatory result on untouched seeds: food 0.100 -> 1.483 against a 0.150
  random baseline (+1.333 [1.050, 1.625], 3/3 seeds), survival 5.0 -> 29.85 steps,
  state-dependence 0.712. Heuristic reference 17.258.
- Snake perturbations on frozen weights: graded dose-response to neuron ablation
  (0.442 / 0.308 / 0.100 food at 5 / 10 / 25 percent).
- Fix and report a preprocessing artefact: standardising the rewired Snake arm with
  biological statistics made it appear to abolish learning entirely; per-graph
  calibration restored it to 1.933 food.
- Add cross-task analysis over catch, dodge and Snake; add Snake tables, figures, a
  gameplay/sensory/neural/policy demonstration figure and an animated GIF.
- Add tests/test_snake.py (80 tests total), including danger-channel validity and
  Snake/v1/v1.1 seed disjointness.
- Rewrite paper/manuscript.md as the final paper.

## 0.3.0 — v1.1 credit-assignment rebuild, branch flyarcade-v1.1-learning
- Add `src/flyarcade/v11/`: actor-critic e-prop three-factor learning over the
  unchanged MaleCNS topology. Local eligibility traces, a learned critic, TD error,
  neuron-specific learning signals, separate actor/critic rates, entropy
  regularisation, an exploration floor, bounded weights and clipped updates.
- Verify the frozen v1.1 core reproduces v1's LIF spike trains bit-for-bit.
- Diagnose and fix two conditioning failures: descending common mode and readout
  scaling over 1,293 features. Both are frozen preprocessing, not tuned parameters.
- Add a bounded coordinate-wise development search on new seed blocks, with
  rejection criteria applied before any performance comparison, and a combined
  validation on five development seeds including two the sweeps never used.
- Freeze `experiments/v11_run_plan.json` with a success criterion stated in advance.
- Confirmatory positive result on untouched seeds: catch 0.206 -> 0.661 vs 0.197
  random; dodge 0.794 -> 0.926 vs 0.803. 3/3 seeds on both; policy stays
  state-dependent. Stage A alone sufficed; the recurrent core stayed frozen.
- Topology comparison came out against the biological graph: rewired beat biological
  on catch (0.817 vs 0.661) and tied on dodge. Post-hoc analysis attributes it to a
  higher-dimensional, less correlated descending code in the rewired circuit.
- Repeat lesions now that a state-dependent policy exists; catch degrades to 0.329
  under 10% neuron ablation, in contrast to v1 where nothing changed.
- Add tests/test_v11.py including a v1/v1.1 seed-disjointness assertion.
- v1 (commit ec27ad0) is preserved unchanged as the negative baseline.

## 0.2.0 — v1 complete with a negative primary result, 2026-09-12
- Finish the frozen 36-trial grid: 2 tasks x 4 conditions x 3 seeds, plus
  100-episode and both transfer directions. All controls, sensory-noise robustness
  and 10% edge/neuron ablations complete.
- Add full-grid reproducibility verification (scripts/replay_trial.py,
  scripts/replay_all.py): 36 trials and 144 evaluation sets replay bit-identically
  from saved checkpoints, each in its own process under the unchanged 120 s guard.
- Add post-hoc mechanism diagnostics (scripts/diagnostics.py) on frozen weights
  under a matched state distribution, with a shared-covariance linear probe and
  constant-action reference policies. No retuning; no feedback into any model.
- Add scripts/export_tables.py: 8 manuscript CSV tables and the robustness,
  transfer and connectome figures; adopt the validated categorical palette.
- Record the negative primary result: no held-out improvement in any condition, on
  any seed, on either task; no trained condition beats the uniform-random policy.
  Identify policy collapse with intact decodable descending signal as the cause.
- Rewrite paper/manuscript.md around the measured outcome and pivot the framing to
  what does and does not emerge from reward-modulated plasticity.
- Add tests/test_analysis.py (60 tests total). README.md unchanged; no resource
  limit raised.

## 0.1.0 — MaleCNS source correction, 2026-09-12
- Make HHMI Janelia MaleCNS v1.0 canonical; supersede FlyWire without fallback.
- Install pinned neuprint-python 0.6.3 optional dependency and record environment versions.
- Add pinned client, bounded transport/query builders, annotation-preserving record adapter and 20 regression cases.
- Switch source preflight to a clean missing-token exit (3); no population acquired.
- Document token setup, unchanged guards, anatomical selection plan and exact resumption steps.
- Preserve M0 and README.md; M2–M5 remain gated on authentic acquisition/validation.

## 0.1.0 — 2026-09-12 checkpoint
- Complete M0: CPU package, environment snapshot, health checks and guards.
- Implement M1 sparse import, exact IDs, provenance, topology controls and storage.
- Add 27 passing tests, synthetic benchmark and fixed-source data preflight.
- Record M1 download guard and complete resumption instructions; later milestones pending.
- Add data/protocol docs and an explicitly incomplete manuscript scaffold.
- Preserve README.md and the supplied master unchanged.
