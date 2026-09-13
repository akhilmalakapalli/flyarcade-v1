"""Fit (once) the v1.5 development standardizers needed by a set of specs."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import v15_path  # noqa: F401
from flyarcade_v15.trainer import DEFAULTS, ensure_standardizer

from flyarcade.resources import ResourceGuard

ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def keys(paths):
    out = set()
    for p in paths:
        c = {**DEFAULTS, **json.load(open(p))["config"]}
        if c["source"] == "fly":
            out.add((c["task"], c["ticks"], c["readout"], c["std_floor"]))
    return sorted(out)


def main():
    if sys.argv[1] == "--one":
        task, ticks, readout, floor = sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
        print(ensure_standardizer(task, ticks, readout, floor, guard=ResourceGuard()), flush=True)
        return
    todo = keys(sys.argv[1:])

    def fit(k):
        cmd = [sys.executable, __file__, "--one", k[0], str(k[1]), k[2], str(k[3])]
        return subprocess.run(
            cmd, check=True, timeout=125, capture_output=True, text=True, env=ENV
        ).stdout

    with ThreadPoolExecutor(8) as pool:
        for line in pool.map(fit, todo):
            print(line.strip())


if __name__ == "__main__":
    main()
