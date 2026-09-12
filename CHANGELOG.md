# Changelog

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
