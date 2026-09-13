"""Fit development-only feature standardizers and build extra rewired controls.

Rewired graphs 0-2 are the preserved v1.2 caches; 3-4 use the identical
degree-preserving procedure with the next seeds (703, 704). Existing outputs are kept.
"""

import argparse
import json
from pathlib import Path

from flyarcade.connectome import degree_preserving_control, load_graph, save_graph
from flyarcade.resources import ResourceGuard
from flyarcade_v13.study import fit_standardizer, graph_hash, load_topology, standardizer_path


def ensure_rewired():
    biological = load_graph("data/malecns-v1.0/graph.npz")
    Path("data/v13").mkdir(parents=True, exist_ok=True)
    for seed in (3, 4):
        path = Path(f"data/v13/rewired-{seed}.npz")
        if not path.exists():
            save_graph(
                degree_preserving_control(biological, seed=700 + seed, swaps_per_edge=10), path
            )
            print(path, "built", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--topology", default="biological")
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--readout", default="descending")
    parser.add_argument("--rewired", action="store_true", help="only build rewired graphs 3-4")
    args = parser.parse_args()
    if args.rewired:
        ensure_rewired()
        return
    output = standardizer_path(args.task, args.topology, args.ticks, args.readout)
    if output.exists():
        print(f"{output}: preserved", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = load_topology(args.topology)
    std, meta = fit_standardizer(graph, args.task, ticks=args.ticks, readout=args.readout)
    std.save(output)
    meta.update(task=args.task, topology=args.topology, graph_sha256=graph_hash(graph))
    meta["resources"] = ResourceGuard().snapshot()
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{output}: {meta['samples']} states from {len(meta['seeds'])} episodes", flush=True)


if __name__ == "__main__":
    main()
