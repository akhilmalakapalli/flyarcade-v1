"""Store a user-pasted neuPrint token locally without echoing it or logging it."""

import getpass
import os
import tempfile
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[1] / ".secrets"
    token = getpass.getpass("Paste neuPrint Account auth token (hidden): ").strip()
    if not token or len(token.encode()) > 16384:
        raise SystemExit("Token must be nonempty and at most 16 KiB; nothing saved.")
    directory.mkdir(mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(token + "\n")
        os.replace(temporary, directory / "neuprint-token")
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Saved .secrets/neuprint-token with owner-only file permissions. Token not displayed.")


if __name__ == "__main__":
    main()
