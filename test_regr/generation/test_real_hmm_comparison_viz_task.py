from __future__ import annotations

import json
import math

from domiknows.generation import constraints_to_dfa_from_graph, discover_generation_constraints, trace_discrete_hmm

from Tasks.real_hmm_comparison_viz.flow import (
    CANDIDATES,
    TWO_CONSTRAINT_CANDIDATES,
    build_bundle,
    build_flow,
    build_domiknows_hmm,
    build_plain_hmm,
    build_two_constraint_domiknows_hmm,
)
from Tasks.real_hmm_comparison_viz.graph import ENUM_VALUES, build_graph, build_two_constraint_graph
from Tasks.real_hmm_comparison_viz.run_demo import main as run_demo_main


def _labels(bundle, symbols):
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def test_real_hmm_comparison_graph_has_one_constraint():
    graph, parts = build_graph()
    generated_symbol = parts[4]
    logical_constraints = getattr(graph, "logicalConstrains", getattr(graph, "_logicalConstrains", {}))

    assert graph.findConcept("string") is not None
    assert graph.findConcept("generated_symbol") is generated_symbol
    assert tuple(generated_symbol.enum) == ENUM_VALUES
    assert len(logical_constraints) == 1


def test_real_hmm_comparison_dfa_accepts_valid_and_rejects_invalid():
    graph, bundle = build_bundle()
    constraints = discover_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")

    assert len(constraints) == 1
    assert "at most 1" in constraints[0].name
    assert dfa.accepts(_labels(bundle, CANDIDATES["valid"]))
    assert not dfa.accepts(_labels(bundle, CANDIDATES["invalid"]))


def test_real_hmm_comparison_two_constraint_graph_and_dfa():
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


def test_plain_discrete_hmm_scores_invalid_candidate_finitely():
    hmm = build_plain_hmm()
    trace = trace_discrete_hmm(hmm, CANDIDATES["invalid"])

    assert math.isfinite(trace["log_likelihood"])
    assert trace["support_blocked"] is False
    assert trace["states"] == ["S0", "S1", "S2"]
    assert len(trace["steps"]) == len(CANDIDATES["invalid"])
    assert set(trace["steps"][0]) >= {"alpha", "beta", "gamma", "viterbi_state"}


def test_domiknows_hmm_blocks_invalid_second_b():
    model = build_domiknows_hmm()

    assert model.score(CANDIDATES["valid"]) > float("-inf")
    assert model.score(CANDIDATES["invalid"]) == float("-inf")
    assert model.emission_mask_[2, model.symbol_to_id["B"]].item() == 0.0


def test_two_constraint_domiknows_hmm_blocks_second_b_and_missing_c():
    model = build_two_constraint_domiknows_hmm()

    assert model.score(TWO_CONSTRAINT_CANDIDATES["valid"]) > float("-inf")
    assert model.score(TWO_CONSTRAINT_CANDIDATES["two_b"]) == float("-inf")
    assert model.score(TWO_CONSTRAINT_CANDIDATES["missing_c"]) == float("-inf")
    assert all("seen_C_seen_B__emit_B" not in state for state in model.state_names)
    assert all("need_C_seen_B__emit_END" not in state for state in model.state_names)


def test_real_hmm_comparison_flow_contains_all_three_layers():
    flow = build_flow("invalid")

    assert set(flow) >= {"vocabulary", "constraint", "candidate", "dfa", "discrete_hmm", "domiknows_hmm", "steps"}
    assert flow["dfa"]["accepted"] is False
    assert math.isfinite(flow["discrete_hmm"]["log_likelihood"])
    assert flow["domiknows_hmm"]["log_likelihood"] == "-inf"
    assert flow["discrete_hmm"]["states"] == ["S0", "S1", "S2"]
    assert all(row["B"] == 1.0 for row in flow["discrete_hmm"]["emission_mask"].values())
    assert flow["discrete_hmm"]["transition_mask"]["S2"]["S0"] == 1.0
    assert all("seen_B__emit_B" not in state for state in flow["domiknows_hmm"]["states"])
    assert any(state.startswith("seen_B__emit_A") for state in flow["domiknows_hmm"]["states"])
    assert flow["steps"][-1]["dfa"]["blocked"] is True
    assert flow["steps"][-1]["discrete_hmm"]["support_blocked"] is False
    assert flow["steps"][-1]["domiknows_hmm"]["support_blocked"] is True
    assert set(flow["steps"][0]["domiknows_hmm"]) >= {"alpha", "beta", "gamma"}
    assert all(value == 0.0 for value in flow["steps"][-1]["domiknows_hmm"]["gamma"].values())


def test_real_hmm_comparison_two_constraint_flow_contains_all_three_layers():
    valid = build_flow("valid", demo="two")
    two_b = build_flow("two_b", demo="two")
    missing_c = build_flow("missing_c", demo="two")

    assert "C must appear at least once" in valid["constraint"]["text"]
    assert valid["dfa"]["accepted"] is True
    assert math.isfinite(valid["discrete_hmm"]["log_likelihood"])
    assert valid["domiknows_hmm"]["log_likelihood"] < 0
    assert valid["domiknows_hmm"]["states"][0].startswith("need_C_no_B")
    assert two_b["dfa"]["accepted"] is False
    assert math.isfinite(two_b["discrete_hmm"]["log_likelihood"])
    assert two_b["domiknows_hmm"]["log_likelihood"] == "-inf"
    assert two_b["steps"][3]["symbol"] == "B"
    assert two_b["steps"][3]["domiknows_hmm"]["support_blocked"] is True
    assert missing_c["dfa"]["accepted"] is False
    assert math.isfinite(missing_c["discrete_hmm"]["log_likelihood"])
    assert missing_c["steps"][-1]["symbol"] == "END"
    assert missing_c["steps"][-1]["domiknows_hmm"]["support_blocked"] is True


def test_real_hmm_comparison_viewer_has_comparison_panels_and_help():
    html = open("Tasks/real_hmm_comparison_viz/viewer.html", encoding="utf-8").read()

    assert "DFA Hard Rule" in html
    assert "Plain DiscreteHMM" in html
    assert "DomiKnowS-Aware HMM" in html
    assert "Plain HMM Static Support" in html
    assert "all tokens from every state" in html
    assert "Generic latent clusters" in html
    assert "S0 is a generic hidden pattern" in html
    assert "Plain DiscreteHMM Factors" in html
    assert "DomiKnowS-Aware HMM Factors" in html
    assert "alpha" in html
    assert "gamma" in html
    assert "data-help=" in html
    assert 'id="flow-data"' in html


def test_real_hmm_comparison_run_demo_writes_strict_json_and_html(tmp_path):
    output_dir = tmp_path / "demo"

    assert run_demo_main(["--candidate", "invalid", "--output-dir", str(output_dir)]) == 0

    flow_path = output_dir / "flow.json"
    html_path = output_dir / "index.html"
    assert flow_path.exists()
    assert html_path.exists()
    raw = flow_path.read_text(encoding="utf-8")
    assert "Infinity" not in raw
    assert "NaN" not in raw
    flow = json.loads(raw)
    assert flow["domiknows_hmm"]["log_likelihood"] == "-inf"
    html = html_path.read_text(encoding="utf-8")
    assert '<script id="flow-data" type="application/json">{' in html
    assert 'embedded !== "__FLOW_JSON__"' in html


def test_real_hmm_comparison_run_demo_two_constraint_variant(tmp_path):
    output_dir = tmp_path / "demo"

    assert run_demo_main(["--demo", "two", "--candidate", "two_b", "--output-dir", str(output_dir)]) == 0

    flow = json.loads((output_dir / "flow.json").read_text(encoding="utf-8"))
    assert flow["candidate"]["demo"] == "two"
    assert flow["dfa"]["accepted"] is False
    assert flow["domiknows_hmm"]["log_likelihood"] == "-inf"
    assert flow["steps"][3]["symbol"] == "B"
