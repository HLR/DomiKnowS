"""Wheel-level contracts without importing full DomiKnowS or KAoS."""
from __future__ import annotations

import copy
import json
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from domiknows_planning_monitor import (MonitorEngine, MonitorRefused,
                                         compile_monitor, validate)
from domiknows_planning_monitor.server import make_handler


class MemoryStore:
    def __init__(self):
        self.records = {}

    def get(self, kind, key):
        return copy.deepcopy(self.records.get((kind, key)))

    def immutable(self, kind, key, value):
        old = self.get(kind, key)
        if old is not None and old != value:
            raise ValueError("immutable record changed")
        self.records[(kind, key)] = copy.deepcopy(value)
        return self.get(kind, key)

    @contextmanager
    def edit(self, kind, key, event):
        value = self.get(kind, key)
        if value is None:
            raise KeyError(key)
        yield value
        self.records[(kind, key)] = copy.deepcopy(value)


def spec(scope="mission", nodes=None, groups=(), directives=()):
    return dict(scope=scope, identity=dict(
        mission="search", revision=1, domainFingerprint="domain", planDigest="plan",
        actor="drone" if scope == "local" else None,
        awardKey="survey|0" if scope == "local" else None,
        epoch=0 if scope == "local" else None),
        nodes=nodes or [dict(id="A", actionClass="survey",
                             actor="drone" if scope == "local" else None, dependsOn=[]),
                        dict(id="report", actionClass="report",
                             actor="drone" if scope == "local" else None, dependsOn=["A"])],
        groups=groups, directives=directives)


class MonitorContracts(unittest.TestCase):
    def test_mission_dependency_reaward_and_outcome(self):
        artifact = compile_monitor(spec())
        self.assertEqual(validate(artifact), artifact)
        store = MemoryStore()
        engine = MonitorEngine(store)
        engine.install("run", artifact, mission="search", revision=1,
                       domain_fingerprint="domain")
        engine.award("run", node_id="A", actor="drone", award_key="A|0", epoch=0)
        engine.award("run", node_id="report", actor="drone", award_key="report|0", epoch=0)
        with self.assertRaisesRegex(MonitorRefused, "dependencies"):
            engine.reserve("run", node_id="report", dispatch="early", action_class="report",
                           actor="drone", award_key="report|0", epoch=0)
        engine.reserve("run", node_id="A", dispatch="first", action_class="survey",
                       actor="drone", award_key="A|0", epoch=0)
        with self.assertRaisesRegex(MonitorRefused, "reconciliation"):
            engine.award("run", node_id="A", actor="drone-2", award_key="A|1", epoch=1)
        engine.settle("run", node_id="A", dispatch="first", succeeded=False, evidence="journal:first")
        engine.award("run", node_id="A", actor="drone-2", award_key="A|1", epoch=1)
        engine.reserve("run", node_id="A", dispatch="second", action_class="survey",
                       actor="drone-2", award_key="A|1", epoch=1)
        engine.settle("run", node_id="A", dispatch="second", succeeded=True,
                      evidence="journal:second")
        engine.reserve("run", node_id="report", dispatch="report", action_class="report",
                       actor="drone", award_key="report|0", epoch=0)

    def test_local_choice_repair_and_obligation(self):
        nodes = [dict(id="A", actionClass="survey", actor="drone", dependsOn=[]),
                 dict(id="B", actionClass="survey", actor="drone", dependsOn=[])]
        artifact = compile_monitor(spec("local", nodes, groups=[
            dict(id="choice", kind="case_or", members=["A", "B"], k=1)]))
        engine = MonitorEngine(MemoryStore())
        engine.install("run", artifact, mission="search", revision=1,
                       domain_fingerprint="domain", actor="drone",
                       award_key="survey|0", epoch=0)
        engine.reserve("run", node_id="A", dispatch="a", action_class="survey",
                       actor="drone", award_key="survey|0", epoch=0)
        with self.assertRaisesRegex(MonitorRefused, "alternative"):
            engine.reserve("run", node_id="B", dispatch="b", action_class="survey",
                           actor="drone", award_key="survey|0", epoch=0)
        engine.settle("run", node_id="A", dispatch="a", succeeded=True, evidence="journal:a")
        engine.add_obligation("run", node_id="notify", action_class="notify",
                              actor="drone", incurred_by="A")
        engine.reserve("run", node_id="notify", dispatch="n", action_class="notify",
                       actor="drone", award_key="survey|0", epoch=0)
        with self.assertRaisesRegex(MonitorRefused, "in-flight"):
            engine.replace("run", artifact, mission="search", revision=1,
                           domain_fingerprint="domain", actor="drone",
                           award_key="survey|0", epoch=0)

    def test_small_server_rejects_unbound_request_facts(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/compile"
            def post(body):
                return urllib.request.urlopen(urllib.request.Request(
                    url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}))
            with post({"spec": spec()}) as response:
                self.assertEqual(json.load(response)["version"], 1)
            with self.assertRaises(urllib.error.HTTPError) as failure:
                post({"spec": spec(), "requestFacts": {"profile": "missing"}})
            self.assertEqual(failure.exception.code, 422)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
