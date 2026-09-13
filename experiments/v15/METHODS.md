# v1.5 methods (development and fresh validation only)

* Plan: `experiments/v15/development_plan.json` (committed before development training).
* Diagnosis that motivated it: `artifacts/v15/diagnosis.md` (committed first).
* Engine: `src_v15/flyarcade_v15` (kept outside `src/` so the frozen v1.4 provenance hash
  stays reproducible). It imports, unchanged, the MaleCNS LIF core (`flyarcade_v14.features`),
  environment adapters (`flyarcade_v14.environments`), and PPO/GAE/actor-critic
  (`flyarcade_v13.ppo`, `flyarcade_v13.policy`, `flyarcade_v14.policy`).
* Readout populations: descending (1,293), nonvisual = descending + central-brain intrinsic
  (1,822), cb_intrinsic (529), pooled = mean rate per cell type over nonvisual neurons (605);
  leakage controls: visual inputs (218), all (2,040), sensory encoding (no SNN).
* Execution: `scripts/v15_suite.py` → `scripts/v15_run.py` (8 single-thread workers) →
  `scripts/v15_trial.py` (≤120 s, 2 GiB per process; checkpoint at 70 s; bit-identical resume).
* Every run records config, per-update PPO diagnostics (entropy, value loss, explained
  variance, KL, clip fraction, gradient norm, action counts), checkpoint curve, chosen
  checkpoint, dev score, state probe, action distribution, entropy, random/reference
  baselines, wall time, peak RSS, checkpoint and policy hashes.
* Validation: `scripts/v15_validate.py` after the locked selections are committed.
