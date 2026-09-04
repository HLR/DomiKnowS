from __future__ import annotations

import json
from pathlib import Path

from domiknows.generation import constraints_to_dfa_from_graph, discover_generation_constraints

from Tasks.simple_hmm_dfa_viz.flow import (
    CANDIDATES,
    TWO_CONSTRAINT_CANDIDATES,
    build_bundle,
    build_flow,
    build_tiny_hmm,
    build_two_constraint_hmm,
    trace_tiny_hmm,
)
from Tasks.simple_hmm_dfa_viz.graph import ENUM_VALUES, build_graph, build_two_constraint_graph
from Tasks.simple_hmm_dfa_viz.run_demo import main as run_demo_main, terminal_file_link


REPO_ROOT = Path(__file__).resolve().parents[2]


def _labels(bundle, symbols):
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def test_simple_viz_graph_has_one_readable_constraint():
    graph, parts = build_graph()
    generated_symbol = parts[4]

    logical_constraints = getattr(graph, "logicalConstrains", getattr(graph, "_logicalConstrains", {}))
    assert graph.findConcept("string") is not None
    assert graph.findConcept("position") is not None
    assert graph.findConcept("symbol") is not None
    assert graph.findConcept("generated_symbol") is generated_symbol
    assert tuple(generated_symbol.enum) == ENUM_VALUES
    assert len(logical_constraints) == 1


def test_simple_viz_dfa_accepts_at_most_one_b_and_rejects_two_b():
    graph, bundle = build_bundle()
    constraints = discover_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")

    assert len(constraints) == 1
    assert "at most 1" in constraints[0].name
    assert dfa.accepts(_labels(bundle, CANDIDATES["valid"]))
    assert not dfa.accepts(_labels(bundle, CANDIDATES["invalid"]))


def test_simple_viz_two_constraint_graph_and_dfa():
    graph, parts = build_two_constraint_graph()
    generated_symbol = parts[4]
    logical_constraints = getattr(graph, "logicalConstrains", getattr(graph, "_logicalConstrains", {}))
    graph, bundle = build_bundle("two")
    constraints = discover_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")

    assert tuple(generated_symbol.enum) == ENUM_VALUES
    assert len(logical_constraints) == 2
    assert len(constraints) == 2
    assert dfa.accepts(_labels(bundle, TWO_CONSTRAINT_CANDIDATES["valid"]))
    assert not dfa.accepts(_labels(bundle, TWO_CONSTRAINT_CANDIDATES["two_b"]))
    assert not dfa.accepts(_labels(bundle, TWO_CONSTRAINT_CANDIDATES["missing_c"]))


def test_simple_viz_flow_json_contains_dfa_and_hmm_steps():
    flow = build_flow("invalid")

    assert set(flow) >= {"vocabulary", "constraint", "dfa", "hmm", "steps"}
    assert flow["constraint"]["text"] == "Token B may appear at most once."
    assert flow["dfa"]["accepted"] is False
    assert flow["dfa"]["rejection_reason"]
    assert flow["hmm"]["states"] == ["before_B", "emit_B", "after_B"]
    assert len(flow["steps"]) == len(CANDIDATES["invalid"])

    for step in flow["steps"]:
        assert set(step) >= {"index", "symbol", "dfa", "hmm", "explanation"}
        assert set(step["hmm"]) >= {"prior", "emission_likelihood", "belief", "normalizer"}


def test_simple_viz_hmm_masks_have_expected_shape_and_finite_valid_trace():
    hmm = build_tiny_hmm()
    trace = trace_tiny_hmm(hmm, CANDIDATES["valid"])

    assert len(hmm.transition_mask) == len(hmm.states)
    assert all(len(row) == len(hmm.states) for row in hmm.transition_mask)
    assert len(hmm.emission_mask) == len(hmm.states)
    assert all(len(row) == len(hmm.symbols) for row in hmm.emission_mask)
    assert trace["accepted_by_hmm_support"] is True
    assert trace["log_likelihood"] < 0
    assert trace["viterbi_path"]


def test_simple_viz_invalid_flow_hmm_and_dfa_block_final_b():
    flow = build_flow("invalid")
    final_step = flow["steps"][-1]

    assert final_step["symbol"] == "B"
    assert final_step["dfa"]["blocked"] is True
    assert final_step["hmm"]["support_blocked"] is True
    assert flow["hmm"]["trace"]["log_likelihood"] == "-inf"


def test_simple_viz_two_constraint_flow_shows_both_rules():
    valid = build_flow("valid", demo="two")
    two_b = build_flow("two_b", demo="two")
    missing_c = build_flow("missing_c", demo="two")
    hmm = build_two_constraint_hmm()

    assert "C must appear at least once" in valid["constraint"]["text"]
    assert valid["dfa"]["accepted"] is True
    assert valid["hmm"]["trace"]["accepted_by_hmm_support"] is True
    assert valid["hmm"]["trace"]["log_likelihood"] < 0
    assert len(hmm.states) == 8
    assert two_b["dfa"]["accepted"] is False
    assert two_b["steps"][3]["symbol"] == "B"
    assert two_b["steps"][3]["hmm"]["support_blocked"] is True
    assert missing_c["dfa"]["accepted"] is False
    assert missing_c["steps"][-1]["symbol"] == "END"
    assert missing_c["steps"][-1]["hmm"]["support_blocked"] is True


def test_simple_viz_viewer_contains_interactive_hooks():
    html = (REPO_ROOT / "Tasks" / "simple_hmm_dfa_viz" / "viewer.html").read_text(encoding="utf-8")

    assert 'id="flow-data"' in html
    assert 'id="step-list"' in html
    assert "addEventListener(\"click\"" in html
    assert "flow.json" in html
    assert "Transition Mask" in html
    assert "Emission Mask" in html
    assert "Belief is the HMM" in html
    assert "data-help=" in html


def test_simple_viz_run_demo_writes_json_and_html(tmp_path):
    output_dir = tmp_path / "demo"

    assert run_demo_main(["--candidate", "invalid", "--output-dir", str(output_dir)]) == 0

    flow_path = output_dir / "flow.json"
    html_path = output_dir / "index.html"
    assert flow_path.exists()
    assert html_path.exists()
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    assert flow["dfa"]["accepted"] is False
    assert "Infinity" not in flow_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert '<script id="flow-data" type="application/json">{' in html
    assert 'embedded !== "__FLOW_JSON__"' in html


def test_simple_viz_run_demo_two_constraint_variant(tmp_path):
    output_dir = tmp_path / "demo"

    assert run_demo_main(["--demo", "two", "--candidate", "two_b", "--output-dir", str(output_dir)]) == 0

    flow = json.loads((output_dir / "flow.json").read_text(encoding="utf-8"))
    assert flow["generator"]["demo"] == "two"
    assert flow["dfa"]["accepted"] is False
    assert flow["steps"][3]["symbol"] == "B"
    assert flow["steps"][3]["hmm"]["support_blocked"] is True


def test_simple_viz_terminal_link_is_clickable_file_uri(tmp_path):
    html_path = tmp_path / "index.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    link = terminal_file_link(html_path)

    assert "file:///" in link
    assert "index.html" in link
    assert "\033]8;;" in link
