"""Portable monitor compiler checks independent of the KAoS runtime."""
import pytest

from domiknows.generation.applications.planning_monitor import (
    artifact_digest, compile_local_monitor, compile_mission_monitor,
    validate_action_candidate,
)


def _leaf(task, index):
    return dict(task=task, actionClass=task, kind="leaf", index=index)


def test_sar_mission_monitor_keeps_parallel_survey_parts_unordered():
    tree = dict(kind="goal", connector="AND", precedence="DIRECTED", children=[
        _leaf("plan", 0),
        dict(task="survey", actionClass="survey", kind="goal", connector="AND",
             precedence="UNDIRECTED", children=[_leaf("A", 1), _leaf("B", 2), _leaf("C", 3)]),
        _leaf("localize", 4), _leaf("confirm", 5), _leaf("report", 6),
    ])
    parts = [dict(task="plan", status="delivered"), dict(task="survey", status="split")]
    parts.extend(dict(task=name, parent="survey", status="delivered") for name in "ABC")
    parts.extend(dict(task=name, status="delivered") for name in ("localize", "confirm", "report"))
    artifact = compile_mission_monitor(
        dict(tree=tree, parts=parts, mission="m", revision=1), domain_fingerprint="d",
    )
    nodes = {node["id"]: node for node in artifact["nodes"]}
    assert all(nodes[name]["dependsOn"] == ["plan"] for name in "ABC")
    assert nodes["localize"]["dependsOn"] == ["A", "B", "C"]
    assert nodes["report"]["dependsOn"] == ["confirm"]
    assert artifact["digest"] == artifact_digest(artifact)


def test_local_monitor_preserves_actor_dependencies_and_choice():
    run = dict(id="run", mission="m", revision=1, actor="drone", steps=[
        dict(id="choose", directive="case_or", children=["x", "y"]),
        dict(id="x", actor="drone", actionClass="survey-x"),
        dict(id="y", actor="drone", actionClass="survey-y"),
        dict(id="report", actor="drone", actionClass="report", dependsOn=["choose"]),
    ])
    artifact = compile_local_monitor(run, domain_fingerprint="d", award_key="a", epoch=0)
    assert artifact["groups"] == [dict(id="choose", kind="case_or", members=["x", "y"], k=1)]
    report = next(n for n in artifact["nodes"] if n["id"] == "report")
    assert report["dependsOnDirectives"] == ["choose"]
    assert artifact["identity"]["awardKey"] == "a"
    assert artifact["digest"] == "c6a7da3df3092616078d2219b3c60a98f7f7715d28d74709823220974fff0833"


def test_version_one_mission_contract_digest():
    board = dict(mission="m", revision=1, tree=dict(kind="goal", connector="AND",
        precedence="DIRECTED", children=[_leaf("A", 0), _leaf("B", 1)]),
        parts=[dict(task="A", status="delivered"), dict(task="B", status="delivered")])
    artifact = compile_mission_monitor(board, domain_fingerprint="d")
    # Shared golden digest with KAoS's adapter contract test.
    assert artifact["digest"] == "7b2484db6876a4cdedb87444b5f3b00dcaa9e498a49100d1588f4f2f4b0d5f6a"


def test_local_monitor_rejects_unknown_dependencies():
    run = dict(id="run", mission="m", revision=1, actor="drone", steps=[
        dict(id="x", actor="drone", actionClass="survey", dependsOn=["missing"]),
    ])
    with pytest.raises(ValueError, match="unknown local dependency"):
        compile_local_monitor(run, domain_fingerprint="d", award_key="a", epoch=0)


def test_bounded_until_compiles_a_finite_attempt_limit():
    run = dict(id="run", mission="m", revision=1, actor="drone", steps=[
        dict(id="repeat", directive="until", maxIterations=2, children=["scan"]),
        dict(id="scan", actor="drone", actionClass="survey"),
    ])
    artifact = compile_local_monitor(run, domain_fingerprint="d", award_key="a", epoch=0)
    assert artifact["nodes"][0]["maxAttempts"] == 2


def test_request_candidate_requires_position_facts_and_expected_answer():
    from domiknows.generation import GenerationEncoder, compile_generation_constraint_plan
    from domiknows.graph.logicalConstrain import andL, eqL, iotaL, queryL
    from domiknows.graph.dataNode import DataNode

    class Tokenizer:
        def encode(self, token):
            return {"<eos>": [0], "A": [1], "B": [2]}[token]

    graph, item = GenerationEncoder(
        ["<eos>", "A", "B"], eos_token="<eos>", tokenizer=Tokenizer(),
    ).build_graph()
    with graph:
        answer = queryL(item.generated_token, iotaL(andL(
            item.context.token_value("A", "x"),
            eqL(item.generated_token, "instanceID", {"chosen"}),
        )))
    plan = compile_generation_constraint_plan(graph, item)
    nodes = [DataNode(instanceID="other", attributes={}),
             DataNode(instanceID="chosen", attributes={})]
    assert validate_action_candidate(plan, ["B", "A"], nodes,
                                     expected={answer.lcName: "A"})
    with pytest.raises(ValueError, match="violate"):
        validate_action_candidate(plan, ["A", "B"], nodes,
                                  expected={answer.lcName: "A"})
    with pytest.raises(ValueError, match="one ordered DataNode"):
        validate_action_candidate(plan, ["B", "A"], nodes[:1])
    with pytest.raises(ValueError, match="answer is missing"):
        validate_action_candidate(plan, ["B", "A"], nodes)


def test_sidecar_compiles_local_artifact_and_refuses_missing_semantic_profile():
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from domiknows.generation.applications.planning_monitor_sidecar import _Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/compile/local"
        run = dict(id="run", mission="m", revision=1, actor="drone",
                   steps=[dict(id="a", actor="drone", actionClass="survey")])
        body = dict(run=run, domainFingerprint="d", awardKey="a|0", epoch=0)
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            artifact = json.load(response)
        assert artifact["identity"]["awardKey"] == "a|0"
        body["run"] = dict(run, requestFacts=dict(profile="missing", actions=["survey"], dataNodes=[]))
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request)
        assert refused.value.code == 422
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_full_sidecar_binds_request_facts_on_generic_compile_route():
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from domiknows.generation import GenerationEncoder, compile_generation_constraint_plan
    from domiknows.graph.logicalConstrain import andL, eqL, iotaL, queryL
    from domiknows.generation.applications.planning_monitor_sidecar import _Handler, SEMANTIC_PROFILES

    class Tokenizer:
        def encode(self, token):
            return {"<eos>": [0], "A": [1], "B": [2]}[token]

    graph, item = GenerationEncoder(["<eos>", "A", "B"], eos_token="<eos>",
                                    tokenizer=Tokenizer()).build_graph()
    with graph:
        answer = queryL(item.generated_token, iotaL(andL(
            item.context.token_value("A", "x"),
            eqL(item.generated_token, "instanceID", {"chosen"}),
        )))
    SEMANTIC_PROFILES["contract-test"] = compile_generation_constraint_plan(graph, item)
    spec = dict(scope="local", identity=dict(mission="m", revision=1,
        domainFingerprint="d", planDigest="p", actor="drone", awardKey="a", epoch=0),
        nodes=[dict(id="survey", actionClass="survey", actor="drone", dependsOn=[])])
    facts = dict(profile="contract-test", actions=["B", "A"],
                 dataNodes=[dict(instanceID="other"), dict(instanceID="chosen")],
                 expected={answer.lcName: "A"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/compile"
        def post(request_facts):
            request = urllib.request.Request(url,
                data=json.dumps(dict(spec=spec, requestFacts=request_facts)).encode(),
                headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(request)
        with post(facts) as response:
            assert json.load(response)["scope"] == "local"
        with pytest.raises(urllib.error.HTTPError) as refused:
            post(dict(facts, expected={}))
        assert refused.value.code == 422
    finally:
        SEMANTIC_PROFILES.pop("contract-test", None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
