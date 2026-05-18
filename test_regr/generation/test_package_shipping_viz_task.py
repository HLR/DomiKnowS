from __future__ import annotations

import json

from Tasks.package_shipping_viz.flow import build_flow
from Tasks.package_shipping_viz.run_demo import write_demo


def test_package_shipping_viz_invalid_flow_explains_dfa_and_hmm_block():
    flow = build_flow(
        task="ship_fragile_vase",
        candidate_source="invalid_drop_before_seal",
        hmm_iterations=2,
    )

    assert flow["task"]["selected"] == "ship_fragile_vase"
    assert flow["candidate"]["source"] == "invalid_drop_before_seal"
    assert flow["dfa"]["accepted"] is False
    assert "drop_off" in flow["dfa"]["rejection_reason"]
    assert flow["graph_hmm"]["log_likelihood"] == "-inf"
    assert flow["graph_hmm"]["support_blocked"] is True
    assert len(flow["steps"]) == len(flow["candidate"]["actions"])

    blocked_step = flow["steps"][5]
    assert blocked_step["action"] == "drop_off"
    assert blocked_step["dfa"]["blocked"] is True
    assert blocked_step["graph_hmm"]["support_blocked"] is True
    assert "seal_box" in blocked_step["diagnostics"]["remaining_required_actions"]


def test_package_shipping_viz_valid_flow_has_finite_graph_hmm_path():
    flow = build_flow(
        task="ship_book",
        candidate_source="graph_reference_plan",
        hmm_iterations=2,
    )

    assert flow["dfa"]["accepted"] is True
    assert isinstance(flow["graph_hmm"]["log_likelihood"], float)
    assert flow["graph_hmm"]["viterbi_path"]
    assert flow["graph_hmm"]["viterbi_path"][0] == "start"
    assert len(flow["steps"]) == len(flow["candidate"]["actions"])


def test_package_shipping_viz_masks_and_candidates_are_serializable():
    flow = build_flow(
        task="ship_fragile_vase",
        candidate_source="invalid_drop_before_seal",
        hmm_iterations=1,
    )

    assert flow["graph_hmm"]["emission_mask"]["start"]["choose_box"] == 1.0
    assert flow["graph_hmm"]["emission_mask"]["labeled"]["seal_box"] == 1.0
    assert flow["graph_hmm"]["transition_mask"]["labeled"]["sealed"] == 1.0
    assert any(item["accepted"] for item in flow["candidate_options"])
    assert any(not item["accepted"] for item in flow["candidate_options"])
    json.dumps(flow, allow_nan=False)


def test_package_shipping_viz_write_demo_outputs_populated_html(tmp_path):
    flow = write_demo(
        "ship_fragile_vase",
        "invalid_drop_before_seal",
        output_dir=tmp_path,
        hmm_iterations=1,
    )

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    loaded = json.loads((tmp_path / "flow.json").read_text(encoding="utf-8"))
    assert loaded["candidate"]["source"] == flow["candidate"]["source"]
    assert "__FLOW_JSON__" not in html
    assert "Click A Step" in html
    assert "Graph-HMM Phase Belief" in html
    assert "invalid_drop_before_seal" in html
