"""Topology and boundary-coverage audit, without whole-connectome queries."""

from collections import Counter, deque

import numpy as np


def audit_graph(graph):
    neurons = graph.provenance["neurons"]
    by_id = {n["bodyId"]: n for n in neurons}
    neurons = [by_id[int(i)] for i in graph.neuron_ids]
    inputs = [i for i, n in enumerate(neurons) if n.get("superclass") == "visual_projection"]
    outputs = [i for i, n in enumerate(neurons) if n.get("superclass") == "descending_neuron"]
    adjacency = [[] for _ in range(graph.n)]
    for a, b in zip(graph.pre, graph.post, strict=True):
        adjacency[a].append(int(b))
    distance = np.full(graph.n, -1, dtype=int)
    distance[inputs] = 0
    queue = deque(inputs)
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if distance[b] == -1:
                distance[b] = distance[a] + 1
                queue.append(b)
    incoming = np.bincount(graph.post, weights=graph.counts, minlength=graph.n)
    outgoing = np.bincount(graph.pre, weights=graph.counts, minlength=graph.n)
    total_in = np.array([n.get("upstream", 0) for n in neurons], dtype=float)
    total_out = np.array([n.get("downstream", 0) for n in neurons], dtype=float)
    return {
        **graph.summary(),
        "classes": dict(Counter(n.get("superclass", "missing") for n in neurons)),
        "consensus_nt": dict(Counter(n.get("consensusNt") or "missing" for n in neurons)),
        "visual_input_count": len(inputs),
        "descending_output_count": len(outputs),
        "reachable_descending_count": int(np.sum(distance[outputs] >= 0)),
        "descending_shortest_hops": dict(Counter(map(str, distance[outputs].tolist()))),
        "incoming_weight_retained_fraction": float(incoming.sum() / total_in.sum()),
        "outgoing_weight_retained_fraction": float(outgoing.sum() / total_out.sum()),
        "boundary_denominators": "neuPrint upstream/downstream synaptic totals",
        "boundary_validation": bool(np.all(incoming <= total_in) and np.all(outgoing <= total_out)),
        "input_indices": inputs,
        "output_indices": outputs,
    }
