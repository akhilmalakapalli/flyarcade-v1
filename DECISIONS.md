# Decisions

## D001 — authoritative filename and inferred milestones
The supplied `FLYARCADE_MASTER.md` is authoritative. Do not duplicate or rename it merely to match the alternate name in its text. Its milestone and scientific methods sections are absent. Use M0 bootstrap, M1 connectome, M2 neural controller, M3 games, M4 experiments, M5 audit/manuscript. These are engineering decisions, not requirements quoted from the master. Preserve README.md byte for byte.

## D002 — conservative CPU and scientific defaults
Use Python, NumPy and sparse edge lists, one process, one numerical-library thread, small seeded experiments, no GPU/cloud. Enforce graph, disk, resident-memory and elapsed-time bounds. Synthetic fixtures must always be labeled synthetic and never support biological conclusions. Actual connectome acquisition needs a public source, fixed release, checksum and extraction record.

## D003 — reproducibility and ownership
Keep generated data/runs out of Git; preserve machine-readable metadata and compact benchmark reports in artifacts. Do not choose a copyright license on the owner's behalf: LICENSE will explicitly reserve rights until the owner selects terms. Do not automatically commit the user's untracked master document or other user changes.


## D004 — explicit resource budgets
Initial defaults: 4,096 neurons, 250,000 edges, 2,048 MiB RSS, 1,024 MiB minimum free disk, 120 seconds per guarded operation, and 128 MiB per input/download file. These are agent-chosen defaults, not user-specified limits or physical hardware shortages. Graph has a hard default-size ceiling; import may impose stricter limits. Expanded NPZ payload is capped at 32 MiB. Guards are cooperative, not an OS sandbox. scripts/verify.sh sets numerical-library thread counts to one.

## D005 — anatomical and control semantics
Keep exact IDs and unsigned anatomical counts. Aggregate duplicates before thresholding, remove autapses, retain selected isolates. A manifest/hash check validates declared bytes, not independent biological origin. No physiological signs or weights are inferred. Double-edge swaps preserve in/out degree, global counts and outgoing strength, but not incoming strength. Record swap acceptance; do not claim uniform sampling or adequate mixing without validation.

## D006 — historical FlyWire stopping condition (superseded by D008)
On 2026-09-12 the fixed FlyWire v783 archive metadata confirmed proofread_connections_783.feather is 852,022,274 bytes. The preflight script saved evidence and exited 2 because this exceeds the 134,217,728-byte cap; no bulk data downloaded. The smaller Codex CSV route needs an account API token unavailable in this session. Stop milestone execution at the configured guard as requested, rather than silently raising it. A future session can deliberately review the budget and implement bounded Feather acquisition/extraction or import a smaller provenance-preserving export. RAM/disk are otherwise healthy. This is not a routine permission request.

## D007 — checkpoint and environment scope
M0 complete, M1 software tested/data incomplete, M2–M5 not started. Manuscript is a scaffold, not scientific completion. requirements-lock.txt records the tested Mac/Python environment versions without hashes or cross-platform guarantees; build tools still resolve through pyproject.toml. No commit made; original untracked master and new files remain reviewable. README checksum is unchanged.


## D008 — user-directed canonical source correction
HHMI Janelia MaleCNS v1.0 is now the only canonical biological source: https://male-cns.janelia.org/, neuPrint server https://neuprint.janelia.org, dataset male-cns:v1.0. This supersedes D006's FlyWire recovery plan. The default preflight was replaced; legacy FlyWire code remains explicitly noncanonical to preserve existing tests. The user authorized updating project documentation, including the master; README.md remains unchanged. No full MaleCNS graph download and no increase to resource limits. Target 1,000–4,096 neurons, the intersection of the user's approximate range and the existing cap.

## D009 — token checkpoint without overstating server restrictions
No NEUPRINT_APPLICATION_CREDENTIALS or project-local token exists. Installed neuprint-python 0.6.3 rejects construction without a token. A direct HTTP constant query and tiny Meta schema query succeeded without credentials, so we do not claim that the entire API denied anonymous access. Stop the preferred supported client workflow at its missing-token requirement rather than fabricating credentials or substituting an unrequested acquisition route. The canonical preflight exits 3 and records AUTHENTICATION_REQUIRED. No population/adjacency acquisition was performed. The user can configure the official environment variable or use the hidden-input helper to create an ignored owner-readable token file.

## D010 — metadata-preserving, bounded query path
Use exact MaleCNS body IDs (not FlyWire root IDs), total directed ConnectsTo.weight, raw neuron properties and raw edge ROI annotations. Do not add overlapping ROI counts to get total counts. Preserve type/class, sides/regions and all available transmitter/confidence annotations without interpreting physiological sign. Count and preview candidate anatomical populations first, then fix criteria and all IDs before any game outcomes. Candidate LAL/GNG premotor/descending circuitry is a hypothesis to validate after authentication, not a chosen or validated subnetwork. Record adapters reject missing selected neurons, outside endpoints, duplicate pairs and inexact IDs/counts. Synthetic regression records remain labeled synthetic.

## D011 — bounded client and scope of implementation
Install optional neuprint-python==0.6.3 and snapshot dependencies. The supported client subclass bounds streamed response bytes cumulatively to 128 MiB and uses socket timeouts plus cooperative runtime/RSS/disk guards. Query builders constrain metadata to <=64 IDs per query and adjacency to <=16 selected sources with both endpoints inside <=4,096 selected IDs. No full acquisition orchestrator/selection was completed before the token checkpoint; that remains the next M1 task. Existing 27 tests preserved, 20 new regression cases pass (47 total). M2–M5 stay gated on authentic biological acquisition and validation. No local commit made.


## D012 — fixed anatomical selection and validated acquisition
Select all Traced visual_projection, descending_neuron, and cb_intrinsic/class=CX neurons innervating LAL(L), LAL(R), or GNG. Discovery counts fixed this anatomical criterion before any game outcome: 2,040 neurons (218 visual, 529 CX, 1,293 descending). 163 authenticated neuprint-python queries downloaded 29,506,842 bytes in 37.65 s. The initial packaging guard rejected duplicated embedded annotations; retain exact checksummed API responses and reference an immutable response manifest instead, without raising a limit. Final graph has 126,676 directed edges and 1,054,402 synapses. All descending cells are reachable from visual cells; retained input/output weight is only 15.05%/7.12%. This truncated circuit is not a whole-animal physiological simulation.

## D013 — model and experiment design before game results
Use discrete leaky integrate-and-fire neurons, sparse edge arrays, one-tick transmission delay, fixed neuronal threshold/reset/refractory parameters and bounded reward-modulated pair eligibility updates. Anatomical counts set relative weight magnitudes, not conductances. Model acetylcholine as positive and GABA as negative; leave other/unclear transmitter effects disabled in the primary model because receptor context is absent. Preserve their structural edges and annotations. Inject engineered task features only into visual-projection neurons; decode descending-neuron population rates with an artificial softmax motor readout trained by reward-modulated policy eligibility. External encoding/readout are not biological claims. Compare learning, frozen learning, degree-rewired learning, and readout-only learning under matched seeds/budgets. Use two headless lane tasks, separate training/development/test seeds, frozen evaluation, noise/ablation, and equal-budget transfer comparisons. No claim that STDP alone learns the games or topology improves learning unless demonstrated.


## D014 — development and frozen test protocol
Two readout designs were tested only on seed99 and development environment seeds (1,900,000 training; 2,000,000 evaluation). Pooled16 descending rates showed no useful development improvement; use per-neuron descending four-tick spike rates centered at0.25, readout step0.01, recurrent step0.0002. This is an artificial decoder, not a biological motor reconstruction. The full experiment protocol is frozen in experiments/run_plan.json before held-out runs. Three independent seeds and two tasks, 200 training episodes, 20 shared held-out evaluation episodes; four learning/control conditions; fixed noise/edge/neuron perturbations. Transfer is100 source+100 target versus200 target and100 target fresh. Resource limits apply per independently resumable trial. Store negative/mixed results without test-driven retuning. Checkpoints only represent complete episode boundaries; preserve the last boundary on interruption.

## D015 — full-grid replay instead of a two-trial spot check
The original audit replayed two held-out evaluations. Reproducibility is a stated
deliverable, so `scripts/replay_trial.py` and `scripts/replay_all.py` now restore
every completed trial from its saved final checkpoint and re-run its held-out,
sensory-noise, edge-ablation and neuron-ablation evaluations. All 36 trials and
144 evaluation sets matched bit-identically. Each replay runs in its own
subprocess so the unchanged 120 s per-trial guard still applies; no limit was
raised, and the sequential total is not treated as one guarded operation.
Pre-training evaluations predate the surviving checkpoint and are explicitly not
replayable from it. This is an implementation-consistency proof, not evidence of
physiological validity.

## D016 — post-hoc mechanism diagnostics, not retuning
The preregistered primary result is negative, so `scripts/diagnostics.py` asks why
without changing anything. It reads frozen saved weights only; it trains no
controller, selects no model, and no diagnostic output feeds back into the model,
the protocol or the reported outcomes. Probe decoders are analyst instruments
whose accuracy is a measurement of available information, never a behavioural
result. Two design choices were needed for honesty. First, a trained controller
collapses onto one action and therefore visits a degenerate state distribution
that inflates every probe and discriminability statistic; all reported comparisons
use a matched uniform-random behaviour policy so conditions see identical states.
Second, a one-vs-rest least-squares probe cannot place an ordinal middle class at
the argmax and understated decodable information (0.735 on a planted signal whose
Bayes limit is about 0.86); it was replaced with a shared-covariance regularised
linear discriminant (0.825 on the same signal). Being linear, the probe remains a
lower bound on the information present. Constant-action reference policies were
added because they, not the controller, explain the observed held-out scores.

## D017 — negative result retained and framing pivoted
Reward-modulated plasticity produced no held-out improvement in any condition, on
any seed, on either task, and no trained condition beat the uniform-random policy.
No hyperparameter, rule or protocol change was made in response. The manuscript
was rewritten around the measured outcome under the title "What does and does not
emerge from reward-modulated plasticity in a MaleCNS-derived visuomotor
controller?", reporting the negative primary result together with the topology
measurements, the policy-collapse mechanism, the lesion and robustness results and
the transfer comparisons. Perturbation insensitivity is reported as evidence of
behavioural degeneracy, not as robustness, because a policy that ignores its
observations cannot be degraded by ablating them. The +0.025 transfer differences
are reported as noise because no arm learned a source task to transfer from.
Figures use the documented validated categorical palette slots 1-4 in fixed order
with direct value labels.

## D018 — v1.1 changes the learning rule only, on a branch
Branch `flyarcade-v1.1-learning` from the frozen v1 commit ec27ad0. v1 is never
altered, rewritten or reinterpreted. v1.1 code lives in `src/flyarcade/v11/` and
reuses the v1 connectome, graph, LIF membrane equations, transmitter sign rule,
sensory encoding, tasks and reward function unchanged. `tests/test_v11.py` asserts
that a frozen `EpropCore` reproduces v1's `LIF` spike trains bit-for-bit, so any
behavioural difference is attributable to credit assignment and not to dynamics.
Adding files under `src/` changes the `model_code_sha256` that v1's `run_trial.py`
computes, so a *future* re-run of v1 from scratch would record a different code
hash. The frozen v1 artifacts keep the original hash, and v1's verification path
(`replay_all.py`, `scientific_audit.py`) restores checkpoints and re-evaluates
rather than recomputing that hash, so v1 remains verifiable on this branch.

## D019 — two conditioning repairs were necessary, and are preprocessing not tuning
Development exposed two failures that no hyperparameter could fix. First, the
descending population carries a large state-independent common mode (mean absolute
per-neuron mean 0.091 against a per-neuron sd of 0.109), which lets a
state-independent direction dominate the policy gradient — the exact mechanism of
v1's collapse. Second, with 1,293 features `||phi||^2` is of order 1,293, making
every learning rate about a thousand times larger than it appears: the first
development run saturated the critic inside one episode (|delta| pinned at its clip
of 10) and drove entropy to 0.055 nats before episode two. Both are fixed by a
frozen preprocessing step — per-neuron standardisation estimated once on
development rollouts of the frozen core under a uniform-random behaviour policy,
divided by sqrt(width). These are fixed constants of the readout, not searched
parameters, and the statistics are never re-estimated from confirmatory data.

## D020 — staged strategy stopped at stage A; tie broken on anti-collapse margin
Stage A (recurrent core entirely frozen, only the actor-critic plastic) learned on
development seeds, so per the protocol's "prefer the simplest model that genuinely
learns" the search stopped there. Stages B (79,275 synapses onto descending
neurons) and C (additionally 123,434 including central-complex intrinsic) are
implemented and unit-tested but were not needed and support no claim.

The search was coordinate-wise rather than a full grid, which is unaffordable on
CPU; every configuration including rejected ones is recorded in
`artifacts/v11_development/`. Rejection criteria were applied before any
performance comparison. Because coordinate optima need not compose, the combined
configuration was re-validated on five development seeds, two of which the sweeps
never used, and compared head to head with the runner-up. The two tied on
performance (0.736 vs 0.741 averaged over both tasks). The tie was broken on a
pre-specified criterion rather than on score: the chosen configuration keeps far
more state-dependence on dodge (0.30 vs 0.12, against a 0.05 rejection threshold)
and higher entropy (0.90 vs 0.51). Preventing collapse is the point of v1.1, so the
larger anti-collapse margin wins.

## D021 — the topology result came out against the biological graph, and is reported that way
With an identical rule, budget, neuron count, readout and per-graph standardisation,
the degree-preserving rewired control learned catch *better* than the biological
graph (0.817 vs 0.661; paired difference -0.156, interval [-0.325, -0.021], all
three seeds) and was indistinguishable on dodge. The rewired arm is standardised
against its own rates; reusing the biological statistics would have given the
biological arm a calibrated readout and the control an uncalibrated one, which
would have manufactured the opposite conclusion.

A post-hoc analysis explains the direction: the rewired descending code has a
participation ratio of 54.5 against 29.5, lower mean absolute pairwise correlation
(0.112 vs 0.152), and higher linear probe accuracy (0.901 vs 0.774). Decorrelation
gives a linear readout more independent directions. This is not a claim that the
biology is worse: the readout is an artificial linear decoder on a circuit missing
85% of its incoming synaptic weight, and correlated population structure may serve
functions this task cannot measure. No result was reframed or re-run to soften the
direction it came out.

## D022 — lesions repeated only after state-dependent learning existed
v1 reported perturbation insensitivity and refused to call it robustness. v1.1
repeats the identical lesions against a genuinely state-dependent policy, where they
degrade catch substantially (neuron ablation -0.332, sensory noise -0.208, edge
ablation -0.044) and leave near-ceiling dodge unchanged. The same numbers mean
opposite things depending on whether the policy uses its observations, which is
itself recorded as a methodological finding rather than folded into a robustness
claim.

## D023 — Snake reuses the v1.1 architecture; four parameters are task-specific
Snake preserves the MaleCNS subgraph, LIF dynamics, the reward-guided actor-critic
rule, common-mode removal, sqrt(N) scaling, stage A scope, guards, deterministic
seeding and greedy evaluation. Four parameters differ, each forced by a measured
failure rather than chosen to improve a result. At the unchanged v1.1 settings Snake
did not learn (development food 0.15 -> 0.10 after 300 episodes). Snake's mean
absolute TD error is about 0.17 against 0.3-0.5 on the lane tasks, so identical
learning rates produce far smaller updates; measured development food at actor_lr
0.4 / 1.5 / 4.0 / 8.0 was 0.35 / 0.65 / 1.35 / 1.95. With TD errors that small the
entropy term at 0.03 dominated the reward term and the policy never differentiated
(entropy stuck at 1.22 of a 1.386 maximum), so it was lowered to 0.003. A 10 percent
exploration floor is survivable in catch and dodge, which have no terminal failure
state, but frequently fatal in Snake, so it was lowered to 0.02. Snake episodes begin
at about five steps, so 600 episodes supply roughly a twentieth of the updates the
lane tasks receive, and the budget was raised to 4,000. critic_lr, discount and
trace_decay are unchanged. All four are recorded in experiments/snake_run_plan.json
with their justifications.

## D024 — Snake trials resume across processes instead of raising the guard
Snake episodes lengthen as the policy improves, so a 4,000-episode budget cannot fit
one 120-second guarded process. Rather than raise the limit, trials checkpoint at
episode boundaries and exit asking to be resumed; the suite re-invokes them in a
fresh independently guarded process, exactly as v1 handled resumable trials. A guard
is never reset mid-operation to continue an oversized computation. A partial episode
never overwrites a checkpoint.

## D025 — an apparent Snake topology effect was a preprocessing artefact
The first Snake confirmatory run standardised the rewired arm with the biological
circuit's statistics. Under that mismatch the rewired arm appeared to fail outright
(0.100 / 0.075 / 0.100 food, 5.0-7.0 steps), which would have supported the
conclusion a connectome paper is most tempted to draw. Refitting the standardisation
per graph by the identical frozen procedure raised the same arm to 1.875 / 1.200 /
2.725 food. The entire apparent effect was readout calibration, not topology. The
rewired arm was re-run and the artefact is reported in the manuscript rather than
quietly corrected. Any future topology comparison must calibrate each graph by the
same procedure applied to its own activity.

When the plan file was extended with the per-graph standardiser path, the eprop and
frozen arms were re-run so that every Snake trial shares one plan hash. Their results
were verified field-by-field to be identical to the pre-rerun results; this was a
provenance fix, not a re-roll.

## D026 — no single topology conclusion is forced across tasks
Biological topology never exceeded the degree-rewired control on any task, but only
catch resolves the difference from zero (-0.156 [-0.325, -0.021]); dodge (-0.011) and
Snake (-0.450 [-1.175, 0.550]) do not. The manuscript reports the consistent
direction and explicitly declines to claim a uniform effect. The mechanism analysis
(rewired participation ratio 54.5 vs 29.5, correlation 0.112 vs 0.152, probe 0.901 vs
0.774) is offered as an explanation of the direction, not as proof of a general
principle, and is bounded by the fact that the readout is an artificial linear
decoder on a circuit missing 85 percent of its incoming weight.

## D027 — the demonstration recording is the median episode, not the best
scripts/snake_visualize.py defaults to replaying the median-performing evaluation
episode and records its rank among the 40 in artifacts/snake_demo.json. Selecting the
best episode for a demonstration figure would misrepresent typical behaviour even
though the figure is not itself a result.
