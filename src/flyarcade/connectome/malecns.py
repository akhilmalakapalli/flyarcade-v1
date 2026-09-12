"""Pinned MaleCNS neuPrint access and bounded query/record adapters.

No bulk download, dataset fallback, token invention or automatic circuit choice.
Population acquisition must wait for authenticated annotation review.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from flyarcade.config import Limits
from flyarcade.connectome.graph import Graph
from flyarcade.resources import ResourceGuard, ResourceLimit

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
SOURCE_URL = "https://male-cns.janelia.org/"
TOKEN_ENV = "NEUPRINT_APPLICATION_CREDENTIALS"
TOKEN_FILE = Path(".secrets/neuprint-token")
SCHEMA_QUERY = (
    "MATCH (m:Meta) RETURN m.dataset AS dataset, m.primaryRois AS primaryRois, "
    "m.neuronProperties AS neuronProperties LIMIT 1"
)


class AuthenticationRequired(RuntimeError):
    """The supported client requires user-supplied neuPrint credentials."""


def read_token(environ=None, token_file=TOKEN_FILE):
    """Read only the documented environment key or project-local secret file."""
    environ = os.environ if environ is None else environ
    token = environ.get(TOKEN_ENV, "").strip()
    if not token and token_file is not None and Path(token_file).is_file():
        if Path(token_file).stat().st_size > 16384:
            raise ValueError("neuPrint credential file exceeds 16 KiB")
        token = Path(token_file).read_text().strip()
    if not token:
        raise AuthenticationRequired(
            "neuprint-python requires a token. Sign in at https://neuprint.janelia.org, "
            "open Account (upper right), copy the full auth token, and set "
            "NEUPRINT_APPLICATION_CREDENTIALS or save it in .secrets/neuprint-token."
        )
    return token


def create_client(*, environ=None, token_file=TOKEN_FILE, factory=None):
    token = read_token(environ, token_file)
    if factory is None:
        from neuprint import Client

        class BoundedClient(Client):
            """Use the supported client with bounded buffered HTTP responses."""

            def __init__(self, *args, **kwargs):
                self.guard = ResourceGuard()
                self.received_bytes = 0
                super().__init__(*args, **kwargs)

            def _fetch(self, url, json=None, ispost=False):
                self.guard.check()
                # All requests, including constructor dataset discovery, use this
                # hook in pinned neuprint-python 0.6.3. Keep normal TLS/auth handling.
                with self.session.request(
                    "POST" if ispost else "GET",
                    url,
                    json=json,
                    verify=self.verify,
                    timeout=(5, 15),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    chunks = []
                    for chunk in response.iter_content(chunk_size=65536):
                        self.guard.check()
                        self.received_bytes += len(chunk)
                        if self.received_bytes > self.guard.limits.max_download_mb * 2**20:
                            raise ResourceLimit("cumulative neuPrint download exceeds 128 MiB")
                        chunks.append(chunk)
                    response._content = b"".join(chunks)
                    response._content_consumed = True
                    return response

        factory = BoundedClient
    client = factory(SERVER, dataset=DATASET, token=token, verify=True, progress=False)
    if client.dataset != DATASET or client.server.rstrip("/") != SERVER:
        raise ValueError("neuPrint client did not retain the exact canonical server/dataset")
    return client


def _ids(values, maximum=4096):
    values = list(values)
    if not values or len(values) > maximum:
        raise ValueError(f"ID selection must contain 1–{maximum} IDs")
    if any(type(value) is not int or not 0 < value <= 2**64 - 1 for value in values):
        raise ValueError("body IDs must be exact positive integers, never floats or Cypher")
    if len(set(values)) != len(values):
        raise ValueError("duplicate body IDs")
    return sorted(values)


def selection_queries(rois, *, known_rois):
    """Count an anatomical candidate and preview <=64 records; never truncate-select.

    Caller must review the pinned dataset's annotations and justify the circuit
    before fetching a full selected population. ROI names must come from Meta.
    """
    rois = sorted(set(rois))
    if not rois or len(rois) > 16:
        raise ValueError("provide 1–16 reviewed ROI names")
    if any(not isinstance(r, str) or r not in known_rois or "`" in r for r in rois):
        raise ValueError("ROI must match the pinned dataset schema")
    predicate = " OR ".join(f"coalesce(n.`{r}`, false)" for r in rois)
    match = f"MATCH (n:Neuron) WHERE n.status = 'Traced' AND ({predicate})"
    return {
        "count": match + " RETURN count(n) AS count",
        "preview": match + " RETURN properties(n) AS neuron ORDER BY n.bodyId LIMIT 64",
    }


def metadata_query(body_ids):
    ids = _ids(body_ids, maximum=64)
    return (
        f"MATCH (n:Neuron) WHERE n.bodyId IN {json.dumps(ids)} "
        "RETURN properties(n) AS neuron ORDER BY n.bodyId"
    )


def adjacency_query(source_batch, selected_ids):
    """Induced connectivity: both endpoints constrained, <=16 source neurons.

    Return total relationship weight once, plus raw ROI metadata separately.
    Never sum hierarchical/overlapping ROI counts into the total weight.
    """
    selected = _ids(selected_ids)
    sources = _ids(source_batch, maximum=16)
    if not set(sources) <= set(selected):
        raise ValueError("source batch must belong to selected population")
    return (
        "MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) "
        f"WHERE a.bodyId IN {json.dumps(sources)} AND b.bodyId IN {json.dumps(selected)} "
        "RETURN a.bodyId AS pre, b.bodyId AS post, e.weight AS weight, "
        "e.roiInfo AS roiInfo ORDER BY a.bodyId, b.bodyId"
    )


def records_from_response(response, expected_columns, max_rows):
    """Validate JSON shape without a DataFrame conversion that could coerce IDs."""
    if response.get("columns") != list(expected_columns):
        raise ValueError("unexpected neuPrint response columns")
    rows = response.get("data")
    if not isinstance(rows, list) or len(rows) > max_rows:
        raise ResourceLimit("neuPrint response row limit exceeded")
    if any(not isinstance(row, list) or len(row) != len(expected_columns) for row in rows):
        raise ValueError("malformed neuPrint response row")
    return [dict(zip(expected_columns, row, strict=True)) for row in rows]


def graph_from_records(neurons, edges, provenance, *, limits=Limits(), annotation_bundle=None):
    """Validate selected records and retain raw neuron/edge annotation metadata.

    This adapter alone does not establish origin; provenance must include exact
    executed queries and IDs, and callers must persist checksummed API responses.
    Tests explicitly pass kind=synthetic. No physiological NT signs are inferred.
    """
    if provenance.get("dataset") != DATASET or provenance.get("server") != SERVER:
        raise ValueError("records must be declared for the exact MaleCNS server/dataset")
    if provenance.get("kind") not in ("connectome", "synthetic"):
        raise ValueError("explicit data kind required")
    if not provenance.get("queries") or not provenance.get("selection_criteria"):
        raise ValueError("executed queries and anatomical selection criteria required")
    limits.check_graph(len(neurons), len(edges))
    ids = _ids([n.get("bodyId") for n in neurons], maximum=limits.max_neurons)
    if _ids(provenance.get("selected_ids", [])) != ids:
        raise ValueError("metadata must match every selected ID, including isolates")
    index = {body: i for i, body in enumerate(ids)}
    seen = set()
    kept = []
    autapses = 0
    for edge in edges:
        a, b, weight = (edge.get(key) for key in ("pre", "post", "weight"))
        if any(type(v) is not int for v in (a, b, weight)):
            raise ValueError("edge endpoints/counts must be exact integers")
        if a not in index or b not in index or not 0 < weight <= 2**63 - 1:
            raise ValueError("invalid induced endpoint or synapse count")
        if (a, b) in seen:
            raise ValueError("duplicate pair: do not double count neuPrint ROI rows")
        seen.add((a, b))
        if a == b:
            autapses += 1
        else:
            kept.append(edge)
    kept.sort(key=lambda edge: (edge["pre"], edge["post"]))
    if annotation_bundle is not None:
        if set(annotation_bundle) != {"path", "sha256"}:
            raise ValueError("external annotation bundle requires path and SHA256")
        path = Path(annotation_bundle["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != annotation_bundle["sha256"]:
            raise ValueError("external annotation manifest checksum mismatch")
    metadata = {
        **provenance,
        "source_url": SOURCE_URL,
        "release": "1.0",
        "id_namespace": "MaleCNS neuPrint bodyId (not FlyWire root IDs)",
        "neurons": sorted(neurons, key=lambda n: n["bodyId"]),
        "edge_annotations": edges
        if annotation_bundle is None
        else {
            "storage": "checksummed-api-response-bundle",
            **annotation_bundle,
        },
        "autapses_removed": autapses,
        "count_semantics": "ConnectsTo.weight, not sum of ROI counts",
        "synaptic_signs": "not-inferred",
    }
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    # Leave room for arrays/NPY headers under the existing expanded-archive cap.
    if len(encoded) > 24 * 2**20:
        raise ResourceLimit("graph annotation metadata exceeds 24 MiB")
    metadata["records_sha256"] = hashlib.sha256(encoded).hexdigest()
    return Graph(
        np.array(ids, dtype=np.uint64),
        np.array([index[e["pre"]] for e in kept], dtype=np.int64),
        np.array([index[e["post"]] for e in kept], dtype=np.int64),
        np.array([e["weight"] for e in kept], dtype=np.int64),
        metadata,
    )
