"""Check new code and full suite, explicitly recording immutable legacy style failures."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from v14_audit import file_hash


def main():
    environment = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    new_paths = [
        "src/flyarcade_v14",
        "tests/test_v14.py",
        *[str(p) for p in sorted(Path("scripts").glob("v14_*.py"))],
    ]
    commands = {
        "health": [sys.executable, "scripts/health_check.py"],
        "pip": [sys.executable, "-m", "pip", "check"],
        "new_lint": [".venv/bin/ruff", "check", *new_paths],
        "new_format": [".venv/bin/ruff", "format", "--check", *new_paths],
        "full_lint": [".venv/bin/ruff", "check", ".", "--output-format", "json"],
        "full_format": [".venv/bin/ruff", "format", "--check", "."],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
    }
    rows = {}
    for name, command in commands.items():
        p = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
        rows[name] = {
            "command": command,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
        print(name, "PASS" if p.returncode == 0 else "FAILED", flush=True)
        if name == "pytest":
            print(p.stdout, flush=True)
    historical = json.loads(Path("artifacts/v14/historical_hashes.json").read_text())
    diagnostics = json.loads(rows["full_lint"]["stdout"])
    lint_files = sorted({str(Path(r["filename"]).relative_to(Path.cwd())) for r in diagnostics})
    format_files = sorted(set(re.findall(r"--> ([^:\n]+):\d+", rows["full_format"]["stdout"])))
    legacy = set(lint_files + format_files)
    inherited = all(p in historical and file_hash(p) == historical[p] for p in legacy)
    passed = all(
        rows[name]["returncode"] == 0
        for name in ("health", "pip", "new_lint", "new_format", "pytest")
    )
    if rows["full_format"]["returncode"] and not format_files:
        inherited = False
    if rows["full_lint"]["returncode"] and not diagnostics:
        inherited = False
    status = (
        ("PASS_WITH_PREEXISTING_STYLE_FAILURES" if legacy else "PASS")
        if passed and inherited
        else "FAIL"
    )
    report = {
        "status": status,
        **rows,
        "preexisting_style": {
            "lint_diagnostics": len(diagnostics),
            "lint_files": lint_files,
            "format_files": format_files,
            "all_match_frozen_hashes": inherited,
        },
    }
    Path("artifacts/v14/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(status, "; inherited lint diagnostics:", len(diagnostics), "in", len(lint_files), "files")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
