# FlyArcade state — v1.2 complete

## Status and milestones
Implementation, development, protocol freeze, 36 confirmatory trials, matched
controls, perturbations, post-hoc representations, six-task aggregation, figures,
audit and generated manuscript extension are complete. No milestone remains open.
M0 and biological acquisition were reused, not restarted. Canonical dataset:
HHMI Janelia MaleCNS `male-cns:v1.0`, authentic 2,040-neuron subgraph.

## Exact new outcomes
- flappy: before 0.000000, biological 0.000000, random 0.000000, rewired 0.000000; criteria FAIL (0/3 seeds meet all components).
- pong: before 0.300926, biological 0.350000, random 0.252778, rewired 0.488889; criteria FAIL (2/3 seeds meet all components).
- breakout: before 0.365000, biological 0.368333, random 0.468333, rewired 0.401667; criteria FAIL (0/3 seeds meet all components).
- snake: before 0.005000, biological 0.042500, random 0.025000, rewired 0.061667; criteria FAIL (0/3 seeds meet all components).

Success is the frozen conjunction of improvement, random +0.05 margin and
fixed-history state sensitivity on every seed. Failure does not mean weights
failed to change or prove the task impossible. Partial improvements are reported.
New Snake is relative-action 6×6, separate from historical absolute-action Snake.

## Preservation and validation
- 105 tests pass; health, pip consistency, Ruff lint and format checks pass.
- Scientific audit PASS: 36/36 new trials; seed separation and weight invariants.
- All four seed-zero checkpoints replay intact and all three perturbations exactly.
- Fresh full-budget Flappy seed-zero training history and actor/critic match exactly.
- All 18 historical Catch/Dodge checkpoint evaluations match exactly.
- All 467 protected historical files remain byte-identical.
- README.md unchanged; historical paper preserved in manuscript_v11_historical.md.
- Frozen source hash: `9736ec692fe7d488f67ead1041f473d322ab924c3a1c3152c78ca13e273836b7`.

## Resources and processes
No background process, download or training job remains. No guard was increased.
Maximum trial segment 46.894 seconds;
maximum recorded trial RSS snapshot 136.52 MiB (not peak).
Unchanged guards: 120 seconds/operation, 2 GiB RSS, 1 GiB disk reserve,
128 MiB download, 4,096 neurons, 250,000 edges. No new download was needed.

## Artifacts and next action
See artifacts/v12/results_summary.json, scientific_audit.json, verification.json,
results_archive.json.gz, CSV tables and figures/. Paper text is generated from
these audited results. RUNBOOK.md contains exact reproduction commands.
Changes are uncommitted and reviewable on the existing branch. No routine user
input is needed. Any additional tuning requires a new development plan and version.
Preserve ignored data/ and runs/ in backups; they contain authentic source responses
and full checkpoints. No credential value is included in artifacts.
