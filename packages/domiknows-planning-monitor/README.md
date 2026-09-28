# DomiKnowS planning monitor

This is the small, standard-library-only execution monitor extracted from
DomiKnowS. It compiles a normalized finite plan graph to a portable version-1
JSON artifact and verifies execution transitions without importing the full
DomiKnowS package.

```python
from domiknows_planning_monitor import compile_monitor, MonitorEngine

artifact = compile_monitor({
    "scope": "local",
    "identity": {"mission": "search-1", "revision": 1,
                 "domainFingerprint": "domain-sha256", "planDigest": "plan-sha256",
                 "actor": "drone-1", "awardKey": "survey-A|0", "epoch": 0},
    "nodes": [{"id": "survey", "actionClass": "survey-area", "actor": "drone-1",
               "dependsOn": []}],
})
```

Nodes may additionally have `dependsOnDirectives`, `maxAttempts`, and
`stepDigest`. `groups` can contain `case_or` or `k_of_n` choices; `directives`
names local control nodes. `MonitorEngine(store)` requires a store with `get`,
`immutable`, and a transactional `edit` context manager. The caller is
responsible for authenticating awards and Guard outcome evidence. The artifact
digest detects changes to the artifact; it is not a signature.

Run `domiknows-planning-monitor serve --port 8765` to serve `POST /compile`
with `{"spec": ...}` and `GET /health` on loopback. Request facts require a
full DomiKnowS sidecar with an installed semantic profile; the small server
rejects them. The artifact schema version is independent of the wheel version.

Build a private wheel with
`uv build --wheel packages/domiknows-planning-monitor` from the DomiKnowS
repository. Install that wheel without installing full DomiKnowS.
