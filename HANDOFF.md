# Handoff — v1 complete, primary result negative

**All milestones are finished.** MaleCNS v1.0 acquisition, the spiking controller,
both games, the frozen 36-trial grid, all controls, ablations, robustness, transfer,
full-grid replay verification, figures, CSV tables and the manuscript are done.
There is no open milestone, no blocker, and no background process, download or
training job running. **Do not edit README.md.** Do not raise any resource limit.

## Read first
`FLYARCADE_MASTER.md`, `STATE.md`, `DECISIONS.md` (now through D017),
`experiments/METHODS.md`, `experiments/PROTOCOL.md`, `experiments/DATA.md`,
`games/TASKS.md`, then `paper/manuscript.md`. Do not restart completed work and do
not re-run acquisition; the graph and its response manifest are already validated.

## The result, stated plainly
Reward-modulated plasticity produced **no held-out improvement** on either task, in
any of the four conditions, on any of the three seeds. Mean paired held-out change:
catch -0.0139 [-0.100, 0.108], dodge -0.0167 [-0.058, 0.025]. No trained condition
beat the uniform-random policy (catch 0.175 vs 0.194; dodge 0.794 vs 0.806). The
deterministic heuristic scores 1.000, so the tasks are solvable.

**This negative result is preserved deliberately. Do not retune anything to make it
positive.** Held-out results were frozen against retuning by the preregistration in
`experiments/run_plan.json`, and nothing was changed in response to them.

The mechanism is identified, not hand-waved: the policy collapses to a
near-constant action (entropy 1.098 -> 0.361 catch / 0.128 dodge nats; most likely
action reaching p=0.90 and 0.98) while the descending population continues to carry
linearly decodable task information (probe 0.654 -> 0.658/0.713 against a 0.421
majority rate). Constant-action reference policies score catch 0.13-0.23 and dodge
0.77-0.87, which brackets the trained controllers entirely. The failure is credit
assignment and lost exploration, not the connectome-derived front end.

Perturbation insensitivity (all changes <=0.006) is reported as **degeneracy, not
robustness** — a policy that ignores observations cannot be degraded by ablating
them. The +0.025 transfer differences are reported as noise, since no arm learned a
source task to transfer from.

## What exists now
- `artifacts/results_summary.json`, `trial_metrics.json` — seed-level aggregates.
- `artifacts/tables/*.csv` — 8 manuscript tables (primary, contrasts, robustness,
  transfer, trial index, diagnostics, constant-action reference, connectome).
- `artifacts/*.png` and `*.svg` — learning_curves, heldout_results, robustness,
  transfer, policy_collapse, connectome.
- `artifacts/replay_check.json` — 36/36 trials, 144 evaluation sets, bit-identical.
- `artifacts/scientific_audit.json` — PASS, 36 result files hash-verified.
- `artifacts/diagnostics_catch.json`, `diagnostics_dodge.json` — mechanism analysis.
- `paper/manuscript.md` — full study written around the measured outcome.
- New code: `scripts/replay_trial.py`, `scripts/replay_all.py`,
  `scripts/diagnostics.py`, `scripts/export_tables.py`, `tests/test_analysis.py`.

## Verification and reproduction

```sh
sh scripts/verify.sh                               # health, lint, format, 60 tests, benchmark
.venv/bin/python -m pip check
.venv/bin/python scripts/summarize_results.py      # seed aggregation + 2 figures
.venv/bin/python scripts/export_tables.py          # 8 CSVs + 3 figures
.venv/bin/python scripts/replay_all.py             # ~2 min, 36 subprocess replays
.venv/bin/python scripts/scientific_audit.py
.venv/bin/python scripts/diagnostics.py --task catch
.venv/bin/python scripts/diagnostics.py --task dodge
shasum -a 256 README.md                            # must stay a4657288...caaa7769
git status --short
```

Every script is deterministic and rereads only committed artifacts and `runs/`.
Re-running them overwrites artifacts with identical content. `scripts/run_suite.py`
skips any trial whose `result.json` exists, so it is safe but unnecessary to run.

## Resource posture
All 36 trials ran under the unchanged guards: 4,096 neurons, 250,000 edges, 2 GiB
RSS, >=1 GiB free disk, 120 s per trial, 128 MiB cumulative API payload, 32 MiB
expanded NPZ. Peak observed: 28.7 s and 183 MiB for a single trial; 632 s total
sequential. Replays run one subprocess per trial precisely so the per-trial guard
still applies. **No limit was raised at any point, including to finish runs.**
Guards are cooperative, not OS-level isolation.

## If you continue this work
Legitimate next steps, all of which require a **new preregistration** rather than
edits to the frozen v1 protocol:

1. Exploration-preserving variants — entropy regularisation, a temperature floor,
   or a readout that cannot saturate — to test whether the demonstrably available
   descending signal becomes usable. This is the single most promising direction,
   because v1 shows the signal is there and the rule discards it.
2. Sparse terminal-only reward, to test whether dense shaping drives the collapse.
3. A larger seed count, since three seeds support descriptive intervals only.
4. A boundary-drive model better than a uniform 0.18 background current, given that
   85% of incoming synaptic weight is missing.

Do not present any of these as v1 results, and do not fold them into the existing
manuscript's preregistered claims.

## Working tree
Still no commit; the repository remains at 05b0ee7 with only README tracked.
Everything else is untracked and reviewable. `data/`, `runs/`, `cache/` and `.venv`
are ignored; `.secrets/` is ignored and must never be staged. LICENSE still
reserves rights pending the owner's choice.
