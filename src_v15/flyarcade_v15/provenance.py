"""v1.5 provenance: code hash over the frozen engine it imports plus the v1.5 files."""

import hashlib
from pathlib import Path


def code_files():
    return (
        sorted(Path("src").rglob("*.py"))
        + sorted(Path("src_v15").rglob("*.py"))
        + [Path("scripts/v15_trial.py")]
    )


def code_hash():
    h = hashlib.sha256()
    for path in code_files():
        h.update(str(path).encode() + path.read_bytes())
    return h.hexdigest()
