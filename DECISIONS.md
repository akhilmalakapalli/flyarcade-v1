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
