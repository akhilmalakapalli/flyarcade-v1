# MaleCNS v1.0: acquired biological subnetwork

**HHMI Janelia MaleCNS v1.0 is the canonical dataset.** The authenticated
`neuprint-python==0.6.3` client uses `https://neuprint.janelia.org`, dataset
`male-cns:v1.0`. See the [official access instructions](https://male-cns.janelia.org/download/)
and [release notes](https://male-cns.janelia.org/release/). The official project
links a CC-BY license; attribution and license URL are retained in the manifest.
FlyWire is superseded and is never a fallback. Its legacy module only preserves
old regression tests. Synthetic graphs remain explicitly labeled test fixtures.

## Acquired selection

Select **all** Traced neurons innervating `LAL(L)`, `LAL(R)` or `GNG`, annotated as
visual_projection, descending_neuron, or cb_intrinsic with class=CX. This criterion
was fixed from annotation counts before any game outcome, without taking the
first N IDs or selecting neurons for task performance.

The population has **2,040 neurons**: 218 visual-projection, 529 central-complex,
and 1,293 descending neurons. The induced graph has **126,676 directed edges**,
**1,054,402 synapses**, and no isolates. All descending neurons are reachable
from visual neurons in one or two hops. This is a truncated visual/navigation/
descending-command model, not direct muscle control. Only about **15.05% of
incoming** and **7.12% of outgoing** synaptic weight is retained, relative to
neuPrint upstream/downstream totals. The model replaces omitted external input
with an artificial background current; that is a major limitation.

Acquisition used 163 bounded queries and **29,506,842 response bytes** (~28.1 MiB),
within the unchanged 128 MiB cap. No full connection graph was downloaded. The
first packaging attempt hit an annotation-size guard because raw annotations
were embedded twice. Final storage keeps exact checksummed responses separately
and references their immutable manifest, without relaxing any resource limit.

## Artifacts and provenance

- `data/malecns-v1.0/graph.npz`: sparse graph, original body IDs and neuron metadata.
- `data/malecns-v1.0/response-*.json`: raw API responses, including exact edge ROI annotations.
- `data/malecns-v1.0/responses_manifest.json`: immutable selection/IDs/queries/response SHA256s.
- `artifacts/malecns_manifest.json`: acquisition journal, graph checksum and resource record.
- `artifacts/malecns_discovery.json`: initial anatomical count/schema queries.
- `artifacts/malecns_audit.json`: response-hash checks, topology and boundary-coverage audit.

Original neuPrint body IDs are exact integers. Preserve total ConnectsTo.weight
once per directed pair; never sum overlapping ROI counts into total weights.
Raw type/class, side, region, transmitter and confidence fields are retained.
The neural model uses only documented consensusNt assumptions; missing/unclear
annotations are never invented. Generic CSV column labels pre_root_id/post_root_id
can hold MaleCNS body IDs but do not change their namespace or preserve rich metadata
alone. Use the MaleCNS response bundle and adapter for complete reproduction.

Data and checkpoints are Git-ignored; compact manifests, metrics and figures are
retained. Graph loading disables pickle and caps expanded NPZ at 32 MiB. Embedded
annotation metadata remains capped at 24 MiB. API acquisition retains all original
guards: 4,096 neurons, 250,000 edges, 128 MiB cumulative responses, 2 GiB RSS,
1 GiB free-disk reserve and 120 seconds per guarded operation. Guards are cooperative.

## Reuse and reacquisition

Existing data need no token or network for modeling and audit:

```sh
.venv/bin/python scripts/audit_malecns.py
```

For a new local checkout, configure credentials with the hidden-input helper,
then acquire the fixed selection:

```sh
.venv/bin/python scripts/set_neuprint_token.py
.venv/bin/python scripts/check_data_source.py
.venv/bin/python scripts/acquire_malecns.py
.venv/bin/python scripts/audit_malecns.py
```

Acquisition preserves an existing graph and does not silently overwrite it. If
all responses were acquired but finalization was interrupted, use the offline
`finalize_malecns.py` after checking the journal. It verifies every cached response
and requires all expected metadata and adjacency batches. Do not overwrite partial
responses with a new live acquisition without an explicit recovery decision.

## Token setup for future environments

At https://neuprint.janelia.org/ sign in, open the upper-right **Account** menu,
and copy the entire auth token. Run `scripts/set_neuprint_token.py` and paste at
the hidden prompt. It stores an owner-readable `.secrets/neuprint-token`, ignored
by Git. Alternatively set `NEUPRINT_APPLICATION_CREDENTIALS` in the agent's process.
Never paste tokens into chat, tracked files or literal shell commands. See the
[official quickstart](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html).
The client preflight reports readiness, not graph acquisition. Exit 3 means
missing credentials; 2 a resource guard; 4 initialization failure. The current
session's user configured a token and authenticated acquisition succeeded.
