# Changelog

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
