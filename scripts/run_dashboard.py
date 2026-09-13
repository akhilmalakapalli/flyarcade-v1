"""Launch the FlyArcade visualization dashboard (evaluation only, no training).

python scripts/run_dashboard.py [--port 8000] [--no-browser]
"""

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")  # one core is plenty; keep the laptop cool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from flyarcade_dashboard.server import serve

    print("Loading the 2,040-neuron MaleCNS-derived network ...", flush=True)
    server, _ = serve(args.host, args.port)
    url = f"http://localhost:{args.port}"
    print(f"FlyArcade dashboard running at:\n{url}\n(Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
