"""Legacy FlyWire preflight retained for regression tests, not canonical acquisition.

Use flyarcade.connectome.malecns for the user-selected MaleCNS v1.0 source.
"""

import json
import urllib.request

from flyarcade.config import Limits
from flyarcade.resources import ResourceLimit

RECORD_URL = "https://zenodo.org/api/records/10676866"
FILENAME = "proofread_connections_783.feather"
EXPECTED_MD5 = "f48f972d262323a102aed49af1396b8a"
EXPECTED_SIZE = 852_022_274


def inspect_record(record, limits=Limits()):
    """Return an actionable report; never download the bulk data as a side effect."""
    candidates = [f for f in record["files"] if f["key"] == FILENAME]
    if len(candidates) != 1:
        raise ValueError("pinned connectivity file missing or ambiguous")
    item = candidates[0]
    if item["checksum"] != f"md5:{EXPECTED_MD5}" or item["size"] != EXPECTED_SIZE:
        raise ValueError("archive metadata differs from the inspected fixed release")
    cap = limits.max_download_mb * 2**20
    return {
        "source_url": "https://zenodo.org/records/10676866",
        "metadata_url": RECORD_URL,
        "release": "783.0",
        "filename": FILENAME,
        "license": record["metadata"]["license"]["id"],
        "bytes": item["size"],
        "checksum": item["checksum"],
        "download_url": item["links"]["self"],
        "download_limit_bytes": cap,
        "status": "RESOURCE_GUARD" if item["size"] > cap else "WITHIN_SIZE_BUDGET",
        "downloaded": False,
    }


def check_download_budget(report):
    if report["bytes"] > report["download_limit_bytes"]:
        raise ResourceLimit(
            f"{report['filename']}: {report['bytes']} bytes exceeds "
            f"download budget {report['download_limit_bytes']} bytes"
        )


def fetch_record():
    request = urllib.request.Request(RECORD_URL, headers={"User-Agent": "flyarcade-v1/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read(2**20 + 1)
        if len(payload) > 2**20:
            raise ResourceLimit("archive metadata exceeds 1 MiB")
    return json.loads(payload)
