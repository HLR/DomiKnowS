"""Loopback HTTP compiler for normalized plan graphs."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .compiler import VERSION, compile_monitor


def make_handler(semantic_validator=None):
    """Return a handler; a trusted full-DomiKnowS sidecar may supply a validator."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"status": "ok", "compilerVersion": VERSION})
            else:
                self._reply(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path != "/compile":
                self._reply(404, {"error": "unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 2_000_000:
                    raise ValueError("compiler request must be 1..2000000 bytes")
                body = json.loads(self.rfile.read(size))
                facts = body.get("requestFacts")
                if facts is not None:
                    if semantic_validator is None:
                        raise ValueError("request facts require an installed DomiKnowS semantic profile")
                    semantic_validator(facts)
                artifact = compile_monitor(body["spec"])
            except (ValueError, KeyError, TypeError) as exc:
                self._reply(422, {"error": str(exc)})
                return
            self._reply(200, artifact)

        def _reply(self, status, body):
            encoded = json.dumps(body, sort_keys=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    ThreadingHTTPServer((args.host, args.port), make_handler()).serve_forever()


if __name__ == "__main__":
    main()
