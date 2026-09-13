"""Run repository checks without rewriting any historical benchmark artifact."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    environment = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    commands = [
        [sys.executable, "scripts/health_check.py"],
        [sys.executable, "-m", "pip", "check"],
        [".venv/bin/ruff", "check", "."],
        [".venv/bin/ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, env=environment, timeout=120
        )
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(result.returncode)
    digest = hashlib.sha256(Path("README.md").read_bytes()).hexdigest()
    assert digest == "a46572881f00b6963a0bbddcf2cc439d7acfd1ca46158711ad1a6fe8caaa7769"
    Path("artifacts/v12/verification.json").write_text(
        json.dumps({"status": "PASS", "checks": rows, "readme_sha256": digest}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
