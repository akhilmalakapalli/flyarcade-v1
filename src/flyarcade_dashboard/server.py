"""Standard-library HTTP server: static page + a tiny JSON API (polling, no sockets).

No third-party web framework is needed, so the dashboard adds no dependencies.
"""

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flyarcade_dashboard import replay
from flyarcade_dashboard.session import Dashboard

STATIC = Path(__file__).resolve().parent / "static"


def make_handler(dashboard):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FlyArcadeDashboard/1.0"

        def log_message(self, *_):  # keep the terminal quiet during playback
            pass

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):  # noqa: N802
            try:
                path = self.path.split("?", 1)[0]
                if path == "/api/tasks":
                    return self._json(dashboard.tasks())
                if path == "/api/network":
                    return self._json(dashboard.network.payload())
                if path == "/api/session/state":
                    return self._json(dashboard.state())
                if path == "/api/replays":
                    return self._json(replay.list_replays())
                name = (
                    "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
                )
                target = (STATIC / name).resolve()
                if STATIC not in target.parents or not target.is_file():
                    return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                data = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type", mimetypes.guess_type(target.name)[0] or "text/plain"
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:  # noqa: BLE001
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):  # noqa: N802
            try:
                body = self._body()
                path = self.path
                if path == "/api/session/new":
                    return self._json(
                        dashboard.new_session(
                            body["task"], body.get("model_seed"), body.get("demo_index", 0)
                        )
                    )
                if path == "/api/session/step":
                    return self._json(dashboard.step())
                if path == "/api/session/reset":
                    return self._json(dashboard.reset())
                if path == "/api/session/new_demo":
                    return self._json(dashboard.new_demo())
                if path == "/api/session/save":
                    return self._json({"name": replay.save(dashboard)})
                if path == "/api/replay/load":
                    return self._json(replay.load(body["name"]))
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, FileNotFoundError) as error:
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # noqa: BLE001
                return self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(host="127.0.0.1", port=8000, dashboard=None):
    dashboard = dashboard or Dashboard()
    server = ThreadingHTTPServer((host, port), make_handler(dashboard))
    server.daemon_threads = True
    return server, dashboard
