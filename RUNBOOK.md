# flyarcade-v1 operation and reproduction

The authoritative scope is FLYARCADE_MASTER.md. The canonical data are HHMI
Janelia MaleCNS v1.0; README.md is intentionally preserved. Start with STATE.md
and HANDOFF.md for the current checkpoint. Methods and limitations are in
experiments/METHODS.md; preregistration is experiments/run_plan.json.

## Existing workspace

```sh
sh scripts/verify.sh
.venv/bin/python scripts/audit_malecns.py
```

The authenticated graph and raw responses are in ignored data/malecns-v1.0/.
No network or token is needed for simulation once they are present.

## Environment on a new machine

Use Python 3.11+; the measured environment was Python 3.14.6 on macOS arm64.
Create .venv and install `requirements-lock.txt`, then `pip install -e . --no-deps`.
Alternatively install `.[dev,connectome,report]`. The version snapshot records the
measured environment without wheel hashes; it is not a guaranteed cross-platform
lock. Follow experiments/DATA.md to configure a neuPrint token and acquire the
fixed subnetwork under the existing guards. Never commit credentials.

## Run or resume experiments

Run from the repository root, one numerical-library thread:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/run_suite.py
```

The suite executes 36 trials sequentially. Each has its own 120-second guard and
an independent checkpoint. Existing completed results are preserved; incomplete
trials restore the last complete 25-episode boundary after checking graph/config/
model-source hashes. A partial episode never overwrites a valid checkpoint.
A changed model needs a deliberate new experimental protocol and output set;
do not mix it into this frozen pilot or tune on its held-out seeds.

For one trial:

```sh
.venv/bin/python scripts/run_trial.py --task catch --seed 0 --condition learning
```

Conditions: learning, frozen, rewired, readout_only. Seeds: 0,1,2. Budgets: 100 or
200 episodes. Transfer uses `--source dodge --task catch` (or the reverse), with
200 total episodes. The suite includes 100-target and 200-target reference trials.

## Reports and checks

```sh
.venv/bin/python scripts/summarize_results.py   # seed aggregation, 2 figures
.venv/bin/python scripts/export_tables.py       # 8 CSV tables, 3 figures
.venv/bin/python scripts/replay_all.py          # ~2 min, 36 subprocess replays
.venv/bin/python scripts/scientific_audit.py
.venv/bin/python scripts/diagnostics.py --task catch
.venv/bin/python scripts/diagnostics.py --task dodge
.venv/bin/python scripts/benchmark_neural.py
```

The summary requires the full trial grid and checks that all trials share the
same model source hash. It writes compact JSON metrics and PNG/SVG figures to
artifacts/. The scientific audit validates trial hashes, frozen/readout-only
invariants, seed separation and exact checkpoint replay for both tasks.
Raw trial histories and checkpoint files remain in ignored runs/; compact
per-seed metrics are retained in artifacts/trial_metrics.json. `export_tables.py`
writes the eight manuscript CSV tables to artifacts/tables/ plus the robustness,
transfer and connectome figures; it reads existing artifacts only.

`replay_all.py` restores every completed trial from its saved final checkpoint and
re-runs its held-out, sensory-noise and ablation evaluations, one subprocess per
trial so the per-trial 120-second guard still applies. It must report 36/36 MATCH.
Pre-training evaluations predate the surviving checkpoint and are not replayed.

`diagnostics.py` is post-hoc analysis of frozen saved weights. It trains nothing
and selects nothing; its probe decoders are analyst instruments that measure how
much task information the descending population carries, and their accuracy is
never a behavioural result. Comparisons use a matched uniform-random behaviour
policy so that a collapsed policy's degenerate state distribution cannot inflate
them. Do not feed any diagnostic output back into the model or the protocol.

paper/manuscript.md is the human-readable scientific report. It must distinguish
implemented reward updates from evidence of improved performance. Negative
results are valid; do not claim biological-topology superiority or successful
learning from weight changes alone.

## Resource and data boundaries

No GPU/cloud computation or whole-connectome graph download. Keep graph size
<=4,096 neurons and <=250,000 edges, RSS <=2 GiB, >=1 GiB disk reserve, API payload
<=128 MiB, expanded graph archive <=32 MiB. Guards are cooperative, not an OS
sandbox. Each experiment has a bounded budget; the complete sequential suite
naturally takes longer than one trial. Do not reset a guard to continue an
oversized operation. Raw annotations are referenced through checksummed files,
not discarded to shrink data or duplicated into an oversized archive.
