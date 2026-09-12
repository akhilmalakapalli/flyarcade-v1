"""Import a small, provenance-declared CSV; see experiments/DATA.md."""

import argparse
import json
from pathlib import Path

from flyarcade.connectome import import_csv, save_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ids", type=Path, help="one exact neuron ID per line")
    parser.add_argument("--min-synapses", type=int, default=5)
    args = parser.parse_args()
    graph = import_csv(
        args.csv,
        json.loads(args.manifest.read_text()),
        None if args.ids is None else args.ids.read_text().split(),
        args.min_synapses,
    )
    save_graph(graph, args.output)
    print(json.dumps(graph.summary(), indent=2))


if __name__ == "__main__":
    main()
