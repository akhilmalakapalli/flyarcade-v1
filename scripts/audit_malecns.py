"""Verify saved API hashes, graph lineage, and sensory-to-descending reachability."""

import json
from pathlib import Path

from flyarcade.connectome import load_graph
from flyarcade.connectome.audit import audit_graph
from flyarcade.connectome.io import sha256


def main():
    path = Path("data/malecns-v1.0/graph.npz")
    manifest = json.loads(Path("artifacts/malecns_manifest.json").read_text())
    if sha256(path) != manifest["graph_sha256"]:
        raise ValueError("graph checksum mismatch")
    for query in manifest["queries"]:
        if sha256(query["path"]) != query["sha256"]:
            raise ValueError("response checksum mismatch")
    graph = load_graph(path)
    if graph.neuron_ids.tolist() != manifest["selected_ids"]:
        raise ValueError("graph IDs differ from fixed selection")
    report = audit_graph(graph)
    report["graph_sha256"] = manifest["graph_sha256"]
    report["api_responses_verified"] = len(manifest["queries"])
    report["usable_pathway"] = report["reachable_descending_count"] > 0
    Path("artifacts/malecns_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("indices")}, indent=2))
    if not report["usable_pathway"] or not report["boundary_validation"]:
        raise ValueError("biological pathway/boundary audit failed")


if __name__ == "__main__":
    main()
