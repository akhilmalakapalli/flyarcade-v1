"""Flappy rescue trial: installs the opt-in extensions, then runs the unchanged v13 trial.

Identical to ``scripts/v13_trial.py`` (same checkpointing, resume, evaluation and
result format) except that configs may set ``encoding`` and ``potential`` and the
recorded ``code_sha256`` also covers the rescue package and this wrapper.
"""

import argparse
import json
import sys
from pathlib import Path

import flyarcade_v13_flappy as rescue

rescue.install()

import v13_trial  # noqa: E402  (binds the installed dispatchers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args, _ = parser.parse_known_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    rescue.activate(spec["config"])
    return v13_trial.main(["--spec", args.spec])


if __name__ == "__main__":
    sys.exit(main())
