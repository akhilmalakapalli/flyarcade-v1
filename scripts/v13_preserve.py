"""M0: snapshot every protected historical file before any v1.3 model change.

Protected = every tracked file at the v1.2 source commit plus every existing file
under the ignored ``runs/`` and ``data/`` trees and the existing ``artifacts/``
tree, except the living documents v1.3 is required to update. Refuses to overwrite
an existing snapshot. Also archives the v1.2 manuscript verbatim.
"""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LIVING = {
    "STATE.md",
    "HANDOFF.md",
    "RUNBOOK.md",
    "CHANGELOG.md",
    "DECISIONS.md",
    "paper/manuscript.md",
}
OUTPUT = Path("artifacts/v13/historical_hashes.json")
V12_COMMIT = "00861e0e5cb7e08584836395c41f2df1d5234da5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        print(f"{OUTPUT}: preserved (refusing to overwrite)")
        return
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    files = {
        p for p in tracked if p not in LIVING and not p.startswith(("artifacts/v13", "runs/v13"))
    }
    for root in ("runs", "data", "artifacts"):
        for path in Path(root).rglob("*"):
            if path.is_file() and not str(path).startswith(
                ("artifacts/v13", "runs/v13", "data/v13")
            ):
                files.add(str(path))
    files = sorted(p for p in files if Path(p).is_file() and "__pycache__" not in p)
    archive = Path("paper/manuscript_v12_historical.md")
    if not archive.exists():
        shutil.copyfile("paper/manuscript.md", archive)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "v12_commit": V12_COMMIT,
        "living_documents_excluded": sorted(LIVING),
        "manuscript_v12_archive_sha256": digest(archive),
        "manuscript_v12_source_sha256": digest("paper/manuscript.md"),
        "files": {p: digest(p) for p in files},
    }
    assert payload["manuscript_v12_archive_sha256"] == payload["manuscript_v12_source_sha256"]
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{OUTPUT}: {len(files)} protected files at {head}")


if __name__ == "__main__":
    main()
