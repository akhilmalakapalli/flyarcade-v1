"""Run from the repository root after installing the package."""

import json
import platform
import sys

from flyarcade import __version__
from flyarcade.resources import ResourceGuard

if __name__ == "__main__":
    print(
        json.dumps(
            {
                "status": "ok",
                "package": __version__,
                "python": sys.version,
                "platform": platform.platform(),
                "resources": ResourceGuard().check(),
            },
            indent=2,
        )
    )
