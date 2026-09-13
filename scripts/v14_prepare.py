"""Fit only fresh v1.4 development features; preserve existing standardizers."""

import argparse
import json

from flyarcade.resources import ResourceGuard
from flyarcade_v14.environments import TARGETS
from flyarcade_v14.study import digest, fit_standardizer, load_topology, standardizer_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TARGETS, required=True)
    parser.add_argument("--ticks", type=int, choices=[4, 8], default=4)
    parser.add_argument("--readout", choices=["descending", "all"], default="descending")
    args = parser.parse_args()
    guard = ResourceGuard()
    path = standardizer_path(args.task, "biological", args.ticks, args.readout)
    meta_path = path.with_suffix(".meta.json")
    if path.exists():
        meta = json.loads(meta_path.read_text())
        assert digest(path) == meta["standardizer_sha256"]
        print(path, "preserved")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    std, meta = fit_standardizer(
        load_topology("biological"), args.task, ticks=args.ticks, readout=args.readout, guard=guard
    )
    std.save(path)
    meta.update(
        standardizer_sha256=digest(path),
        resources=guard.check(),
        plan_sha256=digest("experiments/v14/development_plan.json"),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(args.task, args.ticks, args.readout, meta["samples"], "states", flush=True)


if __name__ == "__main__":
    main()
