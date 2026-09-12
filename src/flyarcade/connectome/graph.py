"""O(N + E) graph storage and topology controls without dense adjacency matrices."""

from dataclasses import dataclass

import numpy as np

from flyarcade.config import Limits
from flyarcade.resources import ResourceGuard


@dataclass(frozen=True)
class Graph:
    neuron_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    counts: np.ndarray
    provenance: dict

    def __post_init__(self):
        # Own the arrays so callers cannot invalidate a checked graph through aliases.
        for name in ("neuron_ids", "pre", "post", "counts"):
            array = np.array(getattr(self, name), copy=True)
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a one-dimensional integer array")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        n, e = len(self.neuron_ids), len(self.pre)
        Limits().check_graph(n, e)
        if len(np.unique(self.neuron_ids)) != n or np.any(self.neuron_ids <= 0):
            raise ValueError("neuron IDs must be unique positive integers")
        if len(self.post) != e or len(self.counts) != e:
            raise ValueError("edge arrays must have equal length")
        if np.any(self.counts <= 0):
            raise ValueError("synapse counts must be positive")
        if any(np.any(a < 0) or np.any(a >= n) for a in (self.pre, self.post)):
            raise ValueError("edge endpoint out of range")
        if np.any(self.pre == self.post):
            raise ValueError("autapses must be removed during import")
        pairs = self.pre.astype(np.int64) * n + self.post
        if len(np.unique(pairs)) != e:
            raise ValueError("parallel edges must be aggregated during import")
        if self.provenance.get("kind") not in ("synthetic", "connectome", "control"):
            raise ValueError("explicit provenance kind required")

    @property
    def n(self):
        return len(self.neuron_ids)

    @property
    def e(self):
        return len(self.pre)

    def summary(self):
        indegree = np.bincount(self.post.astype(np.int64), minlength=self.n)
        outdegree = np.bincount(self.pre.astype(np.int64), minlength=self.n)
        return {
            "neurons": self.n,
            "edges": self.e,
            "synapses": sum(int(x) for x in self.counts),
            "isolates": int(np.sum((indegree + outdegree) == 0)),
            "array_bytes": sum(
                a.nbytes
                for a in (
                    self.neuron_ids,
                    self.pre,
                    self.post,
                    self.counts,
                )
            ),
            "kind": self.provenance["kind"],
        }


def synthetic_graph(n=128, edges=1024, seed=0):
    """Uniform directed G(n,m), explicitly a software fixture, never biological."""
    Limits().check_graph(n, edges)
    guard = ResourceGuard()
    guard.check()
    if edges > n * (n - 1):
        raise ValueError("too many edges for a simple directed graph")
    rng = np.random.default_rng(seed)
    # NumPy choice may allocate proportional to the population. Instead sample
    # integer codes with a bounded set (at most 16M possible under graph limits).
    codes = set()
    while len(codes) < edges:
        guard.check()
        codes.update(int(x) for x in rng.integers(0, n * (n - 1), edges - len(codes)))
    codes = np.array(sorted(codes), dtype=np.int64)
    pre = codes // max(n - 1, 1)
    post = codes % max(n - 1, 1)
    post += post >= pre
    return Graph(
        np.arange(1, n + 1, dtype=np.uint64),
        pre,
        post,
        rng.integers(1, 11, edges),
        {"kind": "synthetic", "generator": "directed-uniform-gnm", "seed": seed},
    )


def degree_preserving_control(graph, seed=0, swaps_per_edge=10):
    """Directed double-edge swaps preserve each node's in/out degree.

    Keep counts attached to presynaptic edge slots: outgoing strength and the
    global count distribution are preserved, incoming strength is not. No claim
    of independent or uniformly sampled random graphs is made.
    """
    if not isinstance(swaps_per_edge, int) or not 0 <= swaps_per_edge <= 100:
        raise ValueError("swaps_per_edge must be an integer between 0 and 100")
    guard = ResourceGuard()
    guard.check()
    rng = np.random.default_rng(seed)
    post = graph.post.copy()
    occupied = set(zip(graph.pre.tolist(), post.tolist()))
    accepted = 0
    attempts = swaps_per_edge * graph.e if graph.e >= 2 else 0
    for _ in range(attempts):
        if _ % 4096 == 0:
            guard.check()
        i, j = rng.integers(0, graph.e, 2)
        a, b = int(graph.pre[i]), int(post[i])
        c, d = int(graph.pre[j]), int(post[j])
        if a == c or b == d or a == d or c == b:
            continue
        if (a, d) in occupied or (c, b) in occupied:
            continue
        occupied.remove((a, b))
        occupied.remove((c, d))
        occupied.update(((a, d), (c, b)))
        post[i], post[j] = d, b
        accepted += 1
    return Graph(
        graph.neuron_ids,
        graph.pre,
        post,
        graph.counts,
        {
            "kind": "control",
            "method": "directed-double-edge-swap",
            "seed": seed,
            "attempts": attempts,
            "accepted": accepted,
            "parent": graph.provenance,
        },
    )
