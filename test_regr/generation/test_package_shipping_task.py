from __future__ import annotations

import torch

from domiknows.generation.planning import (
    planning_bundle_from_graph,
    planning_dfa_from_graph,
    planning_hmm_masks_from_graph,
)

from Tasks.package_shipping.graph import build_graph
from Tasks.package_shipping.learning_program import (
    build_graph_hmm_head,
    fit_graph_hmm,
    run_one_head_step,
    supervised_head_loss,
)
from Tasks.package_shipping.planner_agent import MockPackageShippingPlannerAgent
from Tasks.package_shipping.run_demo import PLANNING_KWARGS, main as run_demo_main


def _bundle(task="ship_book"):
    graph, _ = build_graph(task)
    return planning_bundle_from_graph(graph, selected_task=task, **PLANNING_KWARGS)


def test_declarative_shipping_graph_builds_for_each_task():
    expected_actions = (
        "done",
        "choose_box",
        "wrap_item",
        "add_padding",
        "insert_item",
        "print_label",
        "print_return_label",
        "seal_box",
        "drop_off",
        "request_pickup",
    )
    for task in ("ship_book", "ship_fragile_vase", "return_item"):
        graph, parts = build_graph(task)
        planned_action = parts[3]
        planned_shipping_task = parts[7]
        assert graph.findConcept("shipping_task") is not None
        assert graph.findConcept("planned_shipping_task") is not None
        assert graph.findConcept("task_requires_action") is not None
        assert graph.findConcept("reference_plan_step") is not None
        assert graph.findConcept("phase_transition") is not None
        assert tuple(planned_action.enum) == expected_actions
        assert tuple(planned_shipping_task.enum) == ("ship_book", "ship_fragile_vase", "return_item")
        assert len(graph._logicalConstrains) >= 9


def test_shipping_bundle_uses_custom_schema_names():
    bundle = _bundle("ship_fragile_vase")
    assert bundle.selected_task == "ship_fragile_vase"
    assert bundle.planned_task.name == "planned_shipping_task"
    assert bundle.selected_required_actions == (
        "choose_box",
        "wrap_item",
        "add_padding",
        "insert_item",
        "print_label",
        "seal_box",
        "drop_off",
    )
    assert bundle.selected_reference_plan == (
        "choose_box",
        "wrap_item",
        "add_padding",
        "insert_item",
        "print_label",
        "seal_box",
        "drop_off",
        "done",
    )
    assert bundle.action_count_limits["seal_box"] == 1
    assert bundle.non_terminal_limit == 7
    assert bundle.phase_transitions[("labeled", "seal_box")] == "sealed"


def test_shipping_dfa_accepts_valid_plans_and_rejects_invalid_ones():
    for task in ("ship_book", "ship_fragile_vase", "return_item"):
        bundle = _bundle(task)
        dfa = planning_dfa_from_graph(bundle)
        assert dfa.accepts(bundle.selected_reference_plan)

    fragile = _bundle("ship_fragile_vase")
    fragile_dfa = planning_dfa_from_graph(fragile)
    assert not fragile_dfa.accepts(("choose_box", "wrap_item", "insert_item", "print_label", "seal_box", "drop_off", "done"))

    returned = _bundle("return_item")
    return_dfa = planning_dfa_from_graph(returned)
    assert not return_dfa.accepts(("choose_box", "insert_item", "print_label", "seal_box", "request_pickup", "done"))

    book = _bundle("ship_book")
    book_dfa = planning_dfa_from_graph(book)
    assert not book_dfa.accepts(("choose_box", "insert_item", "print_label", "drop_off", "seal_box", "done"))
    assert not book_dfa.accepts(("insert_item", "choose_box", "print_label", "seal_box", "drop_off", "done"))
    assert not book_dfa.accepts(("choose_box", "insert_item", "print_label", "seal_box", "seal_box", "drop_off", "done"))
    assert not book_dfa.accepts(("choose_box", "insert_item", "print_label", "seal_box", "request_pickup", "done"))


def test_mock_shipping_planner_proposes_valid_and_rejected_candidates():
    bundle = _bundle("ship_fragile_vase")
    dfa = planning_dfa_from_graph(bundle)
    candidates = MockPackageShippingPlannerAgent(seed=3).propose(bundle, count=7)
    assert any(dfa.accepts(candidate.actions) for candidate in candidates)
    assert any(not dfa.accepts(candidate.actions) for candidate in candidates)


def test_shipping_graph_hmm_fits_scores_and_decodes_phase_paths():
    bundle = _bundle("return_item")
    artifacts = fit_graph_hmm(bundle, max_iter=3, random_seed=0)
    valid_score = artifacts.hmm.score(bundle.selected_reference_plan)
    impossible_score = artifacts.hmm.score(("insert_item", "choose_box", "done"))
    assert valid_score > float("-inf")
    assert impossible_score == float("-inf")
    viterbi = artifacts.hmm.viterbi(bundle.selected_reference_plan)
    assert viterbi.states[0] == "start"
    assert all(state in bundle.phase_names for state in viterbi.states)


def test_shipping_hmm_masks_and_graph_head_are_finite_and_trainable():
    bundle = _bundle("ship_book")
    transition_mask, emission_mask = planning_hmm_masks_from_graph(bundle)
    assert transition_mask.shape == (len(bundle.phase_names), len(bundle.phase_names))
    assert emission_mask.shape == (len(bundle.phase_names), len(bundle.action_names))

    head = build_graph_hmm_head(bundle, random_seed=4)
    before = supervised_head_loss(head, bundle)
    assert torch.isfinite(before)
    losses = run_one_head_step(head, bundle, lr=0.01)
    assert losses["model_loss"] > 0
    after = supervised_head_loss(head, bundle)
    assert torch.isfinite(after)


def test_shipping_run_demo_main_runs_offline():
    assert (
        run_demo_main(
            [
                "--task",
                "ship_fragile_vase",
                "--candidates",
                "2",
                "--hmm-iterations",
                "1",
                "--head-steps",
                "0",
            ]
        )
        == 0
    )
