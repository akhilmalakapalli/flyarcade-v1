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

## v1.2 four-task extension (historical studies remain frozen)

Read `experiments/METHODS_V12.md`. The new Snake is relative-action 6×6;
`experiments/snake_run_plan.json` remains the separate historical absolute-action
study. Never delete historical artifacts or run historical artifact writers just
to validate this extension. README.md remains unchanged.

Install the frozen environment from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

The locked scientific environment includes SciPy, matplotlib and neuprint-python.
The canonical acquired graph and existing historical `runs/` are local ignored
scientific inputs. Preserve/restore these from the original workspace or backup;
see `experiments/DATA.md` for authenticated acquisition provenance. A source-only
clone does not include the raw connectome or checkpoints. The compact v1.2 result
archive is retained evidence; it is not a replacement for checkpoint replay.

Restore reproducible rewired caches (standardizers already ship as frozen files):

```sh
.venv/bin/python scripts/v12_restore_graphs.py
```

Development preparation and bounded candidate validation, preserving existing
outputs on rerun:

```sh
for task in flappy pong breakout snake; do
  .venv/bin/python scripts/v12_prepare.py --task "$task"
  .venv/bin/python scripts/v12_develop.py --task "$task" --candidate 0
  .venv/bin/python scripts/v12_develop.py --task "$task" --candidate 1
done
.venv/bin/python scripts/v12_freeze.py
```

Freeze deliberately refuses to overwrite an existing plan. The delivered plan
is already frozen; do not regenerate it or tune using confirmatory results.
Fresh-development replication belongs in a separate copy with only the new
v1.2 outputs removed, never historical outputs. Confirmatory reproduction uses
the delivered plan and standardizer files so their byte hashes remain exact.

Run one trial or the entire matched grid:

```sh
.venv/bin/python scripts/v12_trial.py --task pong --condition biological --seed 0
.venv/bin/python scripts/v12_suite.py
```

A single trial exits 7 after a safe checkpoint when another guarded segment is
needed; repeat the same command. The suite resumes automatically, at most 20
segments per trial. Completed results are preserved. To regenerate completed
v1.2 trials, use a separate working copy without its `runs/v12-*` directories.
Do not delete historical `runs/v1*`, v1.1, or Snake directories.

Representation analysis, tables, figures and audit:

```sh
for task in flappy pong breakout snake; do
  .venv/bin/python scripts/v12_representation.py --task "$task"
done
.venv/bin/python scripts/v12_report.py
.venv/bin/python scripts/v12_audit.py
.venv/bin/python scripts/v12_manuscript.py
```

Representation scripts preserve completed outputs. Reports regenerate only
`artifacts/v12/`; manuscript generation requires a passing matching audit.
For clean and perturbation checkpoint replay (no training, exact-row comparison):

```sh
for task in flappy pong breakout snake; do
  for seed in 0 1 2; do
    .venv/bin/python scripts/v12_replay.py --task "$task" --seed "$seed"
  done
done
```

Validation without rewriting the historical timing artifact:

```sh
.venv/bin/python scripts/health_check.py
.venv/bin/python -m pip check
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/v12_audit.py
shasum -a 256 README.md
```

All training histories, evaluation rows, action counts, fixed-history state probes,
resource snapshots, graph/code/plan/checkpoint hashes and seed lists are retained
in `runs/v12-*/result.json` and `artifacts/v12/results_archive.json.gz`. Full
checkpoint NPZ files remain in ignored run directories. CSVs and five paired
PNG/SVG figures are under `artifacts/v12/`. Catch/Dodge aggregation requires the
unchanged historical result files; the new archive also records those source rows.

Additional read-only preservation and executor checks:

```sh
.venv/bin/python scripts/v12_historical_replay.py
.venv/bin/python scripts/v12_core_equivalence.py
```

## v1.4 performance development (no confirmatory testing)

Use branch `flyarcade-v1.4-performance`. The plan is already committed at `ee7d012`;
read `experiments/v14/development_plan.json` and `experiments/v14/METHODS.md`.
The original dashboard and all historical results remain frozen. Do not invoke
historical training, evaluation, rescoring or artifact-writing scripts.

Existing locked environment and graph are reused. For a fresh Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

The source graph and full checkpoints live in ignored data/run directories and
must be backed up. Preserve the canonical graph's plan-recorded SHA256. No full
connectome download is required, and no limit may be raised to acquire one.

Development and progress reporting:

```sh
.venv/bin/python scripts/v14_suite.py
.venv/bin/python scripts/v14_report.py
```

The suite prepares fresh feature fits, runs all primary comparisons, locks primary
winners, then applies only declared secondary/rescue rules. It preserves completed
trials and terminal failure records. Trial processes checkpoint after 70 seconds
and return 7; the suite resumes up to the plan's 20-segment cap. Each process keeps
the original 120-second and 2-GiB guards. To resume one existing spec:

```sh
.venv/bin/python scripts/v14_trial.py --spec experiments/v14/specs/catch-primary-linear-s0.json
```

After `artifacts/v14/selected_configs.json` is locked, validation and final reports:

```sh
.venv/bin/python scripts/v14_validate.py
.venv/bin/python scripts/v14_report.py
.venv/bin/python scripts/v14_audit.py
```

Validation preserves completed files and does not rescore them. Never delete a
validation result to enable more tuning or repeat its use for selection. Any
new training configuration requires a new development protocol/version and an
untouched validation block. No v1.4 confirmatory execution is implemented or
permitted by this mission.

Verification that does not run historical scientific evaluation:

```sh
.venv/bin/python scripts/health_check.py
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python scripts/v14_audit.py
```

The dashboard software tests use their dedicated demo seeds. The v1.4 audit uses
fresh software-test seeds and a development-only training replay; it never reruns
historical confirmatory or fresh validation episodes. Tables, summaries, progress,
resource records and compressed development result archive are under artifacts/v14.
The primary comparison CSV excludes secondary and rescue trials; their results
have separate files. Review raw per-seed outcomes before interpreting any mean.
