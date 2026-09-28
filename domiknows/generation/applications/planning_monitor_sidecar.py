"""Local HTTP compiler for portable planning monitor artifacts.

Run with ``python -m domiknows.generation.applications.planning_monitor_sidecar``.
Bind to loopback (the default); callers must authenticate the planning input
through their own local service boundary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .planning_monitor import compile_local_monitor, compile_mission_monitor
except ImportError:  # A standalone sidecar needs no DomiKnowS runtime dependencies.
    from planning_monitor import compile_local_monitor, compile_mission_monitor


SEMANTIC_PROFILES = {}


def _validate_request_facts(run):
    facts = run.get('requestFacts')
    if facts is None:
        return
    _validate_facts(facts)


def _validate_facts(facts):
    if not isinstance(facts, dict) or facts.get('profile') not in SEMANTIC_PROFILES:
        raise ValueError('request facts name no installed semantic profile')
    from domiknows.graph.dataNode import DataNode
    try:
        from .planning_monitor import validate_action_candidate
    except ImportError:
        from planning_monitor import validate_action_candidate
    nodes = [DataNode(instanceID=item['instanceID'], attributes=item.get('attributes') or {})
             for item in facts['dataNodes']]
    validate_action_candidate(
        SEMANTIC_PROFILES[facts['profile']], facts['actions'], nodes,
        expected=facts.get('expected'), request_values=facts.get('requestValues'),
    )


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"status": "ok", "compilerVersion": 1})
        else:
            self._reply(404, {"error": "unknown route"})

    def do_POST(self):
        if self.path not in {"/compile", "/compile/local", "/compile/mission"}:
            self._reply(404, {"error": "unknown route"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 2_000_000:
                raise ValueError("compiler request must be 1..2000000 bytes")
            body = json.loads(self.rfile.read(size))
            if self.path == "/compile":
                from domiknows_planning_monitor import compile_monitor
                if body.get("requestFacts") is not None:
                    _validate_facts(body["requestFacts"])
                artifact = compile_monitor(body["spec"])
            elif self.path == "/compile/local":
                _validate_request_facts(body["run"])
                artifact = compile_local_monitor(
                    body["run"], domain_fingerprint=body["domainFingerprint"],
                    award_key=body["awardKey"], epoch=body["epoch"],
                )
            else:
                artifact = compile_mission_monitor(
                    body["board"], domain_fingerprint=body["domainFingerprint"],
                )
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--semantic-profiles", help="trusted Python module exporting PROFILES by name")
    args = parser.parse_args(argv)
    if args.semantic_profiles:
        spec = importlib.util.spec_from_file_location('planning_semantic_profiles', args.semantic_profiles)
        if spec is None or spec.loader is None:
            raise ValueError('semantic profile module cannot be loaded')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SEMANTIC_PROFILES.update(module.PROFILES)
    ThreadingHTTPServer((args.host, args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
