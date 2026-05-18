from __future__ import annotations

import torch

from domiknows.generation.planning import (
    planning_bundle_from_graph,
    planning_dfa_from_graph,
    planning_hmm_masks_from_graph,
)

from Tasks.cooking_planner.graph import build_graph
from Tasks.cooking_planner.learning_program import (
    build_graph_hmm_head,
    fit_graph_hmm,
    run_one_head_step,
    supervised_head_loss,
)
from Tasks.cooking_planner.planner_agent import MockCookingPlannerAgent
from Tasks.cooking_planner.run_demo import main as run_demo_main


def _bundle(dish="cookie"):
    graph, _ = build_graph(dish)
    return planning_bundle_from_graph(graph, selected_task=dish)


def test_declarative_graph_builds_for_each_dish():
    expected_actions = (
        "done",
        "open_fridge",
        "take_eggs",
        "take_butter",
        "take_lettuce",
        "take_cheese",
        "close_fridge",
        "put_on_table",
        "mix_dough",
        "bake_cookies",
        "cook_omelette",
        "chop_lettuce",
        "serve",
    )
    for dish in ("cookie", "omelette", "salad"):
        graph, parts = build_graph(dish)
        planned_action = parts[3]
        planned_dish = parts[7]
        assert graph.findConcept("plan") is not None
        assert graph.findConcept("step") is not None
        assert graph.findConcept("dish_requires_action") is not None
        assert graph.findConcept("reference_plan_step") is not None
        assert graph.findConcept("phase_transition") is not None
        assert tuple(planned_action.enum) == expected_actions
        assert tuple(planned_dish.enum) == ("cookie", "omelette", "salad")
        assert len(graph._logicalConstrains) >= 10


def test_planning_bundle_extracts_domain_facts_from_graph():
    bundle = _bundle("cookie")
    assert bundle.selected_task == "cookie"
    assert bundle.terminal_action == "done"
    assert bundle.selected_required_actions == (
        "take_eggs",
        "take_butter",
        "mix_dough",
        "bake_cookies",
        "serve",
    )
    assert bundle.selected_reference_plan == (
        "open_fridge",
        "take_eggs",
        "take_butter",
        "close_fridge",
        "put_on_table",
        "mix_dough",
        "bake_cookies",
        "serve",
        "done",
    )
    assert bundle.action_count_limits["open_fridge"] == 2
    assert bundle.non_terminal_limit == 8
    assert bundle.phase_transitions[("fridge_open", "close_fridge")] == "after_fridge"


def test_planning_dfa_accepts_valid_plans_and_rejects_invalid_ones():
    for dish in ("cookie", "omelette", "salad"):
        bundle = _bundle(dish)
        dfa = planning_dfa_from_graph(bundle)
        assert dfa.accepts(bundle.selected_reference_plan)

    bundle = _bundle("cookie")
    dfa = planning_dfa_from_graph(bundle)
    assert not dfa.accepts(("open_fridge", "take_eggs", "take_butter", "put_on_table", "done"))
    assert not dfa.accepts(("take_eggs", "open_fridge", "close_fridge", "done"))
    assert not dfa.accepts(
        (
            "open_fridge",
            "close_fridge",
            "open_fridge",
            "close_fridge",
            "open_fridge",
            "close_fridge",
            "done",
        )
    )
    assert not dfa.accepts(("open_fridge", "take_eggs", "close_fridge", "mix_dough", "done"))


def test_mock_planner_proposes_valid_and_rejected_candidates():
    bundle = _bundle("omelette")
    dfa = planning_dfa_from_graph(bundle)
    candidates = MockCookingPlannerAgent(seed=1).propose(bundle, count=6)
    accepted = [candidate for candidate in candidates if dfa.accepts(candidate.actions)]
    rejected = [candidate for candidate in candidates if not dfa.accepts(candidate.actions)]
    assert accepted
    assert rejected


def test_graph_hmm_fits_scores_and_decodes_phase_paths():
    bundle = _bundle("cookie")
    artifacts = fit_graph_hmm(bundle, max_iter=3, random_seed=0)
    valid_score = artifacts.hmm.score(bundle.selected_reference_plan)
    impossible_score = artifacts.hmm.score(("take_eggs", "open_fridge", "done"))
    assert valid_score > float("-inf")
    assert impossible_score == float("-inf")
    viterbi = artifacts.hmm.viterbi(bundle.selected_reference_plan)
    assert viterbi.states[0] == "start"
    assert all(state in bundle.phase_names for state in viterbi.states)


def test_hmm_masks_and_graph_hmm_head_are_finite_and_trainable():
    bundle = _bundle("salad")
    transition_mask, emission_mask = planning_hmm_masks_from_graph(bundle)
    assert transition_mask.shape == (len(bundle.phase_names), len(bundle.phase_names))
    assert emission_mask.shape == (len(bundle.phase_names), len(bundle.action_names))

    head = build_graph_hmm_head(bundle, random_seed=2)
    before = supervised_head_loss(head, bundle)
    assert torch.isfinite(before)
    losses = run_one_head_step(head, bundle, lr=0.01)
    assert losses["model_loss"] > 0
    after = supervised_head_loss(head, bundle)
    assert torch.isfinite(after)


def test_run_demo_main_runs_offline():
    assert run_demo_main(["--dish", "cookie", "--candidates", "2", "--hmm-iterations", "1", "--head-steps", "0"]) == 0
