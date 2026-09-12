"""Finalize complete cached acquisition without any network access or raised limits."""

import json
from pathlib import Path

from acquire_malecns import PREDICATE

from flyarcade.connectome import save_graph
from flyarcade.connectome.io import sha256
from flyarcade.connectome.malecns import (
    adjacency_query,
    graph_from_records,
    metadata_query,
    records_from_response,
)
from flyarcade.resources import ResourceGuard


def main():
    guard = ResourceGuard()
    path = Path("artifacts/malecns_manifest.json")
    manifest = json.loads(path.read_text())
    if manifest["selection_criteria"] != PREDICATE:
        raise ValueError("selection changed")
    ids = manifest["selected_ids"]
    by_query = {}
    for query in manifest["queries"]:
        guard.check()
        if sha256(query["path"]) != query["sha256"]:
            raise ValueError("cached API response checksum mismatch")
        by_query[query["cypher"]] = query
    neurons, edges = [], []
    for offset in range(0, len(ids), 64):
        response = json.loads(
            Path(by_query[metadata_query(ids[offset : offset + 64])]["path"]).read_text()
        )
        neurons.extend(r["neuron"] for r in records_from_response(response, ["neuron"], 64))
    for offset in range(0, len(ids), 16):
        query = adjacency_query(ids[offset : offset + 16], ids)
        response = json.loads(Path(by_query[query]["path"]).read_text())
        edges.extend(
            records_from_response(response, ["pre", "post", "weight", "roiInfo"], 16 * len(ids))
        )
    # Immutable response manifest avoids a graph/manifest checksum cycle.
    bundle = Path("data/malecns-v1.0/responses_manifest.json")
    bundle.write_text(
        json.dumps(
            {
                k: manifest[k]
                for k in ("server", "dataset", "selection_criteria", "selected_ids", "queries")
            },
            sort_keys=True,
        )
        + "\n"
    )
    provenance = {
        k: manifest[k]
        for k in (
            "kind",
            "server",
            "dataset",
            "selection_criteria",
            "selected_ids",
            "license_url",
            "attribution",
        )
    }
    provenance["queries"] = {"manifest": str(bundle), "sha256": sha256(bundle)}
    graph = graph_from_records(
        neurons,
        edges,
        provenance,
        annotation_bundle={"path": str(bundle), "sha256": sha256(bundle)},
    )
    graph_path = Path("data/malecns-v1.0/graph.npz")
    save_graph(graph, graph_path)
    manifest.update(
        status="ACQUIRED_PENDING_AUDIT",
        graph_sha256=sha256(graph_path),
        graph_summary=graph.summary(),
        annotation_manifest_sha256=sha256(bundle),
        finalization_resources=guard.check(),
        storage_decision="raw edge ROI annotations in verified response bundle",
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(graph.summary(), indent=2))


if __name__ == "__main__":
    main()
