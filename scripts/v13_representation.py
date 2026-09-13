"""Representation diagnostics: tick sweep (development) and biological vs rewired.

    v13_representation.py --task pong --ticks 4 8 12 16 --topologies biological rewired-0

Writes artifacts/v13/representation-<task>.json (one row per topology x ticks, plus a
sensory-observation reference row). Completed rows are preserved on rerun.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flyarcade_v13.representations import (
    activity_statistics,
    analyse,
    collect_states,
    features_for_states,
)
from flyarcade_v13.study import graph_hash, load_topology, standardizer_path

from flyarcade.v11.features import Standardizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--ticks", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--topologies", nargs="+", default=["biological"])
    parser.add_argument("--readout", default="descending")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output or f"artifacts/v13/representation-{args.task}.json")
    report = json.loads(output.read_text()) if output.exists() else {"task": args.task, "rows": {}}
    states = collect_states(args.task)
    report["state_sha256"] = states["state_sha256"]
    report["episode_seeds"] = states["seeds"]
    report["post_hoc"] = True
    if "sensory-observation" not in report["rows"]:
        report["rows"]["sensory-observation"] = analyse(states["observations"], states, args.task)
    for topology in args.topologies:
        graph = load_topology(topology)
        for ticks in args.ticks:
            key = f"{topology}-t{ticks}-{args.readout}"
            if key in report["rows"]:
                continue
            started = time.monotonic()
            raw = features_for_states(graph, args.task, states, ticks=ticks, readout=args.readout)
            seconds = time.monotonic() - started
            path = standardizer_path(args.task, topology, ticks, args.readout)
            std = Standardizer.load(path)
            standardized = np.clip((raw - std.mean) / std.scale, -std.clip, std.clip)
            row = {
                "topology": topology,
                "ticks": ticks,
                "readout": args.readout,
                "graph_sha256": graph_hash(graph),
                "feature_seconds": seconds,
                "seconds_per_state": seconds / len(raw),
                **activity_statistics(raw),
                **analyse(standardized, states, args.task),
                "state_sha256": states["state_sha256"],
            }
            report["rows"][key] = row
            output.write_text(json.dumps(report, indent=1) + "\n")
            print(
                f"{args.task} {key}: PR={row['participation_ratio']:.1f} "
                f"corr={row['mean_pairwise_correlation']:.3f} "
                f"lin_bal={row['linear_action_balanced_accuracy']:.3f} "
                f"mlp_bal={row['mlp_action_balanced_accuracy']:.3f} "
                f"maj_bal={row['majority_balanced_accuracy']:.3f} "
                f"linR2={row['linear_state_r2_mean']:.3f} mlpR2={row['mlp_state_r2_mean']:.3f} "
                f"{seconds:.1f}s",
                flush=True,
            )
    output.write_text(json.dumps(report, indent=1) + "\n")
    s = report["rows"]["sensory-observation"]
    print(
        f"{args.task} sensory: lin_bal={s['linear_action_balanced_accuracy']:.3f} "
        f"mlp_bal={s['mlp_action_balanced_accuracy']:.3f} linR2={s['linear_state_r2_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
