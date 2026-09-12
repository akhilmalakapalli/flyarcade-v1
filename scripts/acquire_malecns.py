"""Acquire the predeclared MaleCNS visual/CX/descending induced population."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from flyarcade.connectome import save_graph
from flyarcade.connectome.io import sha256
from flyarcade.connectome.malecns import (
    DATASET,
    SCHEMA_QUERY,
    SERVER,
    adjacency_query,
    create_client,
    graph_from_records,
    metadata_query,
    records_from_response,
)

PREDICATE = (
    "n.status='Traced' AND (n.`LAL(L)` OR n.`LAL(R)` OR n.GNG) AND "
    "(n.superclass IN ['visual_projection','descending_neuron'] OR "
    "(n.superclass='cb_intrinsic' AND n.class='CX'))"
)


def main():
    root = Path("data/malecns-v1.0")
    root.mkdir(exist_ok=True)
    if (root / "graph.npz").exists():
        raise SystemExit("Existing graph preserved; validate/reuse it instead of reacquiring.")
    client = create_client()
    journal = {
        "kind": "connectome",
        "server": SERVER,
        "dataset": DATASET,
        "selection_criteria": PREDICATE,
        "selected_ids": [],
        "queries": [],
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "HHMI Janelia MaleCNS project and collaborators",
        "status": "ACQUIRING",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def checkpoint():
        Path("artifacts/malecns_manifest.json").write_text(json.dumps(journal, indent=2) + "\n")

    def query(cypher):
        response = client.fetch_custom(cypher, format="json")
        payload = json.dumps(response, sort_keys=True, allow_nan=False).encode()
        path = root / f"response-{len(journal['queries']):04d}.json"
        path.write_bytes(payload)
        journal["queries"].append(
            {
                "cypher": cypher,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        checkpoint()
        return response

    try:
        journal["schema"] = query(SCHEMA_QUERY)
        count = query(f"MATCH (n:Neuron) WHERE {PREDICATE} RETURN count(n) AS count")
        count = records_from_response(count, ["count"], 1)[0]["count"]
        if not 1000 <= count <= 4096:
            raise ValueError("predeclared population outside 1000–4096; refine anatomy explicitly")
        selected = query(
            f"MATCH (n:Neuron) WHERE {PREDICATE} RETURN n.bodyId AS bodyId ORDER BY n.bodyId"
        )
        ids = [r["bodyId"] for r in records_from_response(selected, ["bodyId"], 4096)]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("population count/ID mismatch")
        journal["selected_ids"] = ids
        neurons, edges = [], []
        for offset in range(0, len(ids), 64):
            response = query(metadata_query(ids[offset : offset + 64]))
            neurons.extend(r["neuron"] for r in records_from_response(response, ["neuron"], 64))
        print(f"Metadata complete: {len(neurons)} neurons", flush=True)
        for offset in range(0, len(ids), 16):
            response = query(adjacency_query(ids[offset : offset + 16], ids))
            edges.extend(
                records_from_response(response, ["pre", "post", "weight", "roiInfo"], 16 * len(ids))
            )
            client.guard.limits.check_graph(len(ids), len(edges))
            if offset % 256 == 0:
                print(f"Connectivity sources {offset}/{len(ids)}; {len(edges)} edges", flush=True)
        bundle = root / "responses_manifest.json"
        bundle.write_text(
            json.dumps(
                {
                    k: journal[k]
                    for k in (
                        "server",
                        "dataset",
                        "selection_criteria",
                        "selected_ids",
                        "queries",
                    )
                },
                sort_keys=True,
            )
            + "\n"
        )
        provenance = {
            k: journal[k]
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
        client.guard.check()
        save_graph(graph, root / "graph.npz")
        journal.update(
            status="ACQUIRED_PENDING_AUDIT",
            graph_sha256=sha256(root / "graph.npz"),
            graph_summary=graph.summary(),
            downloaded_bytes=client.received_bytes,
            resources=client.guard.check(),
        )
        checkpoint()
        print(
            json.dumps(
                {
                    k: journal[k]
                    for k in (
                        "status",
                        "graph_summary",
                        "downloaded_bytes",
                        "resources",
                    )
                },
                indent=2,
            )
        )
    except Exception as exc:
        journal.update(
            status="STOPPED",
            error_type=type(exc).__name__,
            resources=client.guard.snapshot(),
            downloaded_bytes=client.received_bytes,
        )
        checkpoint()
        # No raw HTTP exception text or credentials in output.
        print(f"Acquisition stopped: {type(exc).__name__}; see manifest/checkpoint.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
