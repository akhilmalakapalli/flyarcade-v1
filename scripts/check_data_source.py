"""Preflight the canonical MaleCNS neuprint-python client; never acquire a graph."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from flyarcade.connectome.malecns import (
    DATASET,
    SERVER,
    SOURCE_URL,
    AuthenticationRequired,
    create_client,
)
from flyarcade.resources import ResourceGuard, ResourceLimit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_source_check.json"))
    args = parser.parse_args()
    guard = ResourceGuard(directory=args.output.parent)
    report = {
        "source_url": SOURCE_URL,
        "server": SERVER,
        "dataset": DATASET,
        "release": "1.0",
        "access_method": "neuprint-python==0.6.3",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "population_acquired": False,
        "bulk_downloaded": False,
        "download_limit_bytes": guard.limits.max_download_mb * 2**20,
        "max_neurons": guard.limits.max_neurons,
    }
    code = 0
    try:
        guard.check()
        create_client()
        report["status"] = "CLIENT_READY_SELECTION_PENDING"
    except AuthenticationRequired as exc:
        report.update(status="AUTHENTICATION_REQUIRED", reason=str(exc))
        code = 3
    except ResourceLimit:
        report.update(status="RESOURCE_GUARD", reason="neuPrint resource budget exceeded")
        code = 2
    except Exception as exc:
        # Do not persist exception text/requests that might contain credentials.
        report.update(
            status="CLIENT_ERROR",
            error_type=type(exc).__name__,
            reason="Client initialization failed; check token validity and access.",
        )
        code = 4
    report["resources"] = guard.snapshot()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
