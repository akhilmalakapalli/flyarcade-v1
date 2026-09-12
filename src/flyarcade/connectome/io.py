"""Bounded CSV import with exact 64-bit IDs, checksums and extraction history."""

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from flyarcade.config import Limits
from flyarcade.connectome.graph import Graph
from flyarcade.resources import ResourceGuard, ResourceLimit


def sha256(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(2**20):
            if guard:
                guard.check()
            digest.update(chunk)
    return digest.hexdigest()


def import_csv(path, manifest, selected_ids=None, min_synapses=1, limits=Limits()):
    """Import pre_root_id,post_root_id,syn_count columns.

    Extra columns are allowed. Aggregate counts across rows (e.g. neuropils),
    discard autapses, then apply the pair-level threshold. An explicit ID set
    selects an induced subgraph; never silently truncate a graph to fit limits.
    A manifest is a provenance declaration, not independent biological validation.
    """
    path = Path(path)
    guard = ResourceGuard(limits, path.parent)
    guard.check()
    if path.stat().st_size > limits.max_download_mb * 2**20:
        raise ResourceLimit("input-file size guard exceeded")
    if not isinstance(min_synapses, int) or min_synapses < 1:
        raise ValueError("min_synapses must be a positive integer")
    for key in ("kind", "source_url", "release", "license", "sha256"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"manifest requires {key}")
    if manifest["kind"] not in ("connectome", "synthetic"):
        raise ValueError("manifest kind must be connectome or synthetic")
    actual_hash = sha256(path, guard)
    if actual_hash != manifest["sha256"]:
        raise ValueError("source SHA256 mismatch")
    selected = None if selected_ids is None else set(int(x) for x in selected_ids)
    if selected is not None:
        limits.check_graph(len(selected), 0)
        if any(x <= 0 or x > np.iinfo(np.uint64).max for x in selected):
            raise ValueError("selected IDs must be positive uint64 values")
    ids = set() if selected is None else selected.copy()
    pairs = {}
    rows = autapses = excluded = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"pre_root_id", "post_root_id", "syn_count"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires pre_root_id,post_root_id,syn_count")
        for rows, row in enumerate(reader, start=1):
            if rows % 4096 == 0:
                guard.check()
            try:
                a, b, count = (
                    int(row[k])
                    for k in (
                        "pre_root_id",
                        "post_root_id",
                        "syn_count",
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid integer at CSV row {rows + 1}") from exc
            if not 0 < a <= 2**64 - 1 or not 0 < b <= 2**64 - 1 or count <= 0:
                raise ValueError(f"invalid ID/count at CSV row {rows + 1}")
            if selected is not None and (a not in selected or b not in selected):
                excluded += 1
                continue
            ids.update((a, b))
            limits.check_graph(len(ids), len(pairs))
            if a == b:
                autapses += 1
                continue
            total = pairs.get((a, b), 0) + count
            if total > np.iinfo(np.int64).max:
                raise ValueError("aggregated synapse count exceeds int64")
            pairs[(a, b)] = total
            limits.check_graph(len(ids), len(pairs))
    if not ids:
        raise ValueError("empty neuron selection")
    ordered_ids = sorted(ids)
    index = {root: i for i, root in enumerate(ordered_ids)}
    kept = sorted((a, b, c) for (a, b), c in pairs.items() if c >= min_synapses)
    provenance = dict(manifest)
    provenance["extraction"] = {
        "method": "induced" if selected is not None else "all-rows",
        "selected_ids": None if selected is None else ordered_ids,
        "min_pair_synapses": min_synapses,
        "rows_read": rows,
        "autapse_rows_removed": autapses,
        "outside_selection_rows": excluded,
        "pairs_below_threshold": len(pairs) - len(kept),
        "id_order": "ascending",
        "synaptic_signs": "not-inferred",
    }
    guard.check()
    return Graph(
        np.array(ordered_ids, dtype=np.uint64),
        np.array([index[a] for a, _, _ in kept], dtype=np.int64),
        np.array([index[b] for _, b, _ in kept], dtype=np.int64),
        np.array([c for _, _, c in kept], dtype=np.int64),
        provenance,
    )


def save_graph(graph, path):
    """Atomic single-file graph artifact including provenance (no pickle)."""
    path = Path(path)
    ResourceGuard(directory=path.parent).check()
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                neuron_ids=graph.neuron_ids,
                pre=graph.pre,
                post=graph.post,
                counts=graph.counts,
                provenance=np.frombuffer(
                    json.dumps(graph.provenance, sort_keys=True).encode(), dtype=np.uint8
                ),
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_graph(path, limits=Limits()):
    guard = ResourceGuard(limits, Path(path).parent)
    guard.check()
    with zipfile.ZipFile(path) as archive:
        # Bound expanded payload before NumPy can allocate from an external file.
        if sum(info.file_size for info in archive.infolist()) > 32 * 2**20:
            raise ResourceLimit("expanded graph archive exceeds 32 MiB")
    with np.load(path, allow_pickle=False) as data:
        graph = Graph(
            data["neuron_ids"],
            data["pre"],
            data["post"],
            data["counts"],
            json.loads(
                data["provenance"].tobytes().decode()
                if data["provenance"].dtype == np.uint8
                else str(data["provenance"])
            ),
        )
    limits.check_graph(graph.n, graph.e)
    guard.check()
    return graph
