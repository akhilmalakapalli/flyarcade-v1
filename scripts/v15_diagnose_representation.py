"""Phase-1 representation diagnosis for one task and tick count (bounded process)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import v15_path  # noqa: F401
from flyarcade_v15.diagnosis import (
    activity,
    collect_states,
    decode,
    replay_all_rates,
    sensory_features,
)
from flyarcade_v15.features import READOUTS

from flyarcade.resources import ResourceGuard
from flyarcade_v14.study import load_topology

OUT = Path("artifacts/v15/diagnosis/representation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--ticks", type=int, required=True)
    parser.add_argument("--states", type=int, default=4000)
    args = parser.parse_args()
    guard = ResourceGuard()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    states = collect_states(args.task, min_states=args.states)
    graph = load_topology("biological")
    raw_a, source = replay_all_rates(graph, args.task, states, args.ticks, repeat=0)
    guard.check()
    raw_b, _ = replay_all_rates(graph, args.task, states, args.ticks, repeat=1)
    guard.check()
    report = {
        "task": args.task,
        "ticks": args.ticks,
        "states": int(len(raw_a)),
        "episodes": int(len(states["seeds"])),
        "seed_purpose": "v1.5 diagnosis",
        "reference_action_counts": np.bincount(states["labels"]).tolist(),
        "readouts": {},
    }

    for readout in READOUTS:
        from flyarcade_v15.features import Fly15

        view = Fly15.__new__(Fly15)
        view.select, view.pool = _selection(source, graph, readout)
        xa, xb = view_project(view, raw_a), view_project(view, raw_b)
        entry = decode(xa, states, args.task)
        entry["activity"] = activity(xa, xb, states["episodes"])
        report["readouts"][readout] = entry
        guard.check()
    sensory = sensory_features(args.task, states)
    report["readouts"]["sensory_encoding"] = decode(sensory, states, args.task)
    report["elapsed_seconds"] = time.monotonic() - started
    report["resources"] = guard.check()
    path = OUT / f"{args.task}-t{args.ticks}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    short = {
        r: (
            round(v["linear_action_balanced_accuracy"], 3),
            round(v["linear_state_r2_mean"], 3),
            round(v.get("mlp_action_balanced_accuracy", float("nan")), 3),
        )
        for r, v in report["readouts"].items()
    }
    print(args.task, args.ticks, f"{report['elapsed_seconds']:.1f}s", short, flush=True)


def _selection(source, graph, readout):
    from flyarcade_v15.features import Fly15

    probe = Fly15(graph, source.task, 1, ticks=1, readout=readout)
    return probe.select, probe.pool


def view_project(view, raw):
    x = raw[:, view.select]
    return x @ view.pool.T if view.pool is not None else x


if __name__ == "__main__":
    main()
