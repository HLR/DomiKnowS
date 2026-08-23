import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import EOS_TOKEN, load_eai_dataset
from reward import TokenVocabulary, evaluate_goal_satisfaction, make_eai_reward_function, prepare_eai_goal
from world_graph import (
    ABSENT_ENTITY,
    ACTION_GOAL_NAMES,
    ACTION_SPECS,
    NEGATIVE_TO_POSITIVE,
    PREDICATE_ALIASES,
    STATE_SPECS,
    build_eai_world_graph,
    evaluate_default_world_constraints,
    materialize_world_trajectory,
    verify_world_constraints,
)


def _prepared(task_id="synthetic", entities=("apple", "bowl"), pairs=(("apple", "bowl"),)):
    return SimpleNamespace(
        task_id=task_id,
        entity_universe=tuple(entities),
        tracked_binary_pairs=frozenset(pairs),
    )


def _all_descendants(node):
    result = []
    pending = list(node.getChildDataNodes() or [])
    while pending:
        current = pending.pop()
        result.append(current)
        pending.extend(current.getChildDataNodes() or [])
    return result


def test_registry_and_namespaces():
    bundle = build_eai_world_graph("test_eai_world_registry")
    assert bundle.state.name == "world_state"
    assert bundle.action.name == "world_action"
    assert set(bundle.actions) == set(ACTION_SPECS)
    assert set(bundle.states) == set(STATE_SPECS)
    assert len({concept.name for concept in bundle.actions.values()}) == len(ACTION_SPECS)
    assert len({concept.name for concept in bundle.states.values()}) == len(STATE_SPECS)
    assert all(concept.name == f"action__{name}" for name, concept in bundle.actions.items())
    assert all(concept.name == f"state__{name}" for name, concept in bundle.states.items())
    assert all(concept.is_a()[0].dst is bundle.state for concept in bundle.states.values())
    assert all(concept.is_a()[0].dst is bundle.action for concept in bundle.actions.values())
    assert bundle.actions["open"] is not bundle.states["open"]
    assert bundle.goal_actions == frozenset(bundle.actions[name] for name in ACTION_GOAL_NAMES)
    assert all(bundle.aliases[alias] is bundle.states[canonical] for alias, canonical in PREDICATE_ALIASES.items())
    assert bundle.canonical_state_name("next_to") == "nextto"
    assert bundle.is_state_predicate("next_to")
    assert bundle.is_action("open") and bundle.is_goal_action("touch")
    assert bundle.positive_state_name("not_open") == "open"
    assert all(
        bundle.negative_to_positive[bundle.states[negative]] is bundle.states[positive]
        for negative, positive in NEGATIVE_TO_POSITIVE.items()
    )
    assert all(STATE_SPECS[name].arity == 1 for name in ("open", "closed"))
    assert all(STATE_SPECS[name].arity == 2 for name in ("inside", "ontop"))


def test_materializes_actions_and_sparse_state_groundings():
    bundle = build_eai_world_graph("test_eai_world_materialize")
    events = [
        SimpleNamespace(name="sleep", args=()),
        SimpleNamespace(name="grab", args=("apple",)),
        SimpleNamespace(name="pour", args=("apple", "bowl")),
    ]
    states = [
        {("open", "bowl")},
        {("open", "bowl")},
        {("holds_rh", "character", "apple")},
        {("inside", "apple", "bowl")},
    ]
    root = materialize_world_trajectory(_prepared(), states, events, bundle)
    descendants = _all_descendants(root)
    event_nodes = [node for node in descendants if node.ontologyNode is bundle.action]
    next_nodes = [node for node in descendants if node.ontologyNode is bundle.next_step]
    unary_nodes = [
        node for node in descendants
        if node.ontologyNode is bundle.state and node.attributes.get("grounding_arity") == 1
    ]
    binary_nodes = [
        node for node in descendants
        if node.ontologyNode is bundle.state and node.attributes.get("grounding_arity") == 2
    ]
    step_nodes = [node for node in descendants if node.ontologyNode is bundle.step]
    entity_count = len(("apple", "bowl", "character", ABSENT_ENTITY))
    assert len(event_nodes) == 3
    assert len(next_nodes) == 3
    assert len(step_nodes) == 4
    assert len(unary_nodes) == len(states) * entity_count
    # Only apple->bowl and the runtime character->apple pair are tracked.
    assert len(binary_nodes) == len(states) * 2

    by_index = {node.instanceID: node for node in event_nodes}
    absent = ABSENT_ENTITY
    assert by_index[0].relationLinks[bundle.action_roles["arg1"].name][0].instanceValue == absent
    assert by_index[1].relationLinks[bundle.action_roles["arg1"].name][0].instanceValue == "apple"
    assert by_index[1].relationLinks[bundle.action_roles["arg2"].name][0].instanceValue == absent
    assert by_index[2].relationLinks[bundle.action_roles["arg2"].name][0].instanceValue == "bowl"
    assert by_index[2].relationLinks[bundle.action_roles["source_step"].name][0].instanceValue == "2"
    assert by_index[2].relationLinks[bundle.action_roles["result_step"].name][0].instanceValue == "3"
    assert by_index[2].relationLinks[bundle.action_roles["actor"].name][0].instanceValue == "character"
    first_next = {node.instanceID: node for node in next_nodes}[0]
    assert first_next.relationLinks[bundle.step_roles["current"].name][0].instanceValue == "0"
    assert first_next.relationLinks[bundle.step_roles["following"].name][0].instanceValue == "1"
    assert torch.argmax(by_index[2].attributes["<action__pour>"]).item() == 1
    assert torch.argmax(by_index[2].attributes["<action__open>"]).item() == 0
    open_bowl = next(
        node for node in unary_nodes
        if node.relationLinks[bundle.state_roles["step"].name][0].instanceValue == "0"
        and node.relationLinks[bundle.state_roles["subject"].name][0].instanceValue == "bowl"
    )
    assert torch.argmax(open_bowl.attributes["<state__open>"]).item() == 1
    assert torch.argmax(open_bowl.attributes["<state__closed>"]).item() == 0
    inside_apple_bowl = next(
        node for node in binary_nodes
        if node.relationLinks[bundle.state_roles["step"].name][0].instanceValue == "3"
        and node.relationLinks[bundle.state_roles["subject"].name][0].instanceValue == "apple"
        and node.relationLinks[bundle.state_roles["object"].name][0].instanceValue == "bowl"
    )
    assert torch.argmax(inside_apple_bowl.attributes["<state__inside>"]).item() == 1
    assert torch.argmax(inside_apple_bowl.attributes["<state__ontop>"]).item() == 0


def test_constraint_verification_and_aggregation():
    from domiknows.graph.logicalConstrain import nandL

    def constraints(bundle):
        nandL(bundle.states["open"], bundle.states["closed"])
        nandL(bundle.states["on"], bundle.states["off"])

    bundle = build_eai_world_graph("test_eai_world_constraints", (constraints,))
    prepared = _prepared(entities=("door",), pairs=())
    satisfying = materialize_world_trajectory(prepared, [{("open", "door")}], [], bundle)
    assert verify_world_constraints(satisfying, bundle, "mean").score == 1.0

    # Make every compatible unary grounding violate the first constraint while
    # the second remains satisfied: mean=.5, min=0, prod=0.
    entities = ("door", "character", ABSENT_ENTITY)
    violating_facts = {
        fact for entity in entities for fact in (("open", entity), ("closed", entity))
    }
    violating_facts.update(("on", entity) for entity in entities)
    partial = materialize_world_trajectory(prepared, [violating_facts], [], bundle)
    assert verify_world_constraints(partial, bundle, "mean").score == 0.5
    assert verify_world_constraints(partial, bundle, "min").score == 0.0
    assert verify_world_constraints(partial, bundle, "prod").score == 0.0


def test_default_precondition_registry_and_verification():
    bundle = build_eai_world_graph(
        "test_eai_world_default_preconditions",
        include_default_constraints=True,
    )
    assert bundle.default_preconditions
    assert len(bundle.default_constraint_names) == len(bundle.default_preconditions)
    assert not any(name.startswith("action_effect__") for name in bundle.default_constraint_names)
    assert not any(name.startswith("state_mutex__") for name in bundle.default_constraint_names)
    assert {spec.kind for spec in bundle.default_preconditions} == {
        "placement_source_ready",
        "release_source_ready",
        "pour_source_ready",
        "destination_open_if_known",
        "argument_available_in_task",
    }
    assert set(bundle.precondition_concepts) == {
        "placement_source_ready",
        "release_source_ready",
        "pour_source_ready",
        "destination_open_if_known",
        "argument_available_in_task",
    }

    prepared = _prepared(
        entities=("apple", "bowl"),
        pairs=(("character", "apple"), ("apple", "bowl")),
    )
    event = [SimpleNamespace(name="right_place_inside", args=("bowl",))]
    ready_states = [
        {
            ("holds_rh", "character", "apple"),
            ("open", "bowl"),
        },
        {("inside", "apple", "bowl")},
    ]
    ready_root = materialize_world_trajectory(prepared, ready_states, event, bundle)
    graph_evaluation = verify_world_constraints(ready_root, bundle)
    fast_evaluation = evaluate_default_world_constraints(
        ready_states, event, bundle
    )
    assert graph_evaluation.score == 1.0
    assert fast_evaluation.score == 1.0
    assert fast_evaluation.constraint_count == 2

    blocked_states = [
        {("closed", "bowl")},
        set(),
    ]
    blocked_root = materialize_world_trajectory(
        prepared, blocked_states, event, bundle
    )
    blocked_graph = verify_world_constraints(blocked_root, bundle)
    blocked_fast = evaluate_default_world_constraints(
        blocked_states, event, bundle
    )
    holding_name = "action_precondition__right_place_inside__source_holding"
    open_name = "action_precondition__right_place_inside__destination_open_if_known"
    assert blocked_graph.results[holding_name]["satisfied"] == 0.0
    assert blocked_graph.results[open_name]["satisfied"] == 0.0
    assert blocked_fast.score == 0.0

    unknown_open_states = [
        {("holds_rh", "character", "apple")},
        {("inside", "apple", "bowl")},
    ]
    unknown_evaluation = evaluate_default_world_constraints(
        unknown_open_states, event, bundle
    )
    assert unknown_evaluation.score == 1.0
    assert unknown_evaluation.constraint_count == 1

    no_hold_pour = evaluate_default_world_constraints(
        [set(), set()],
        [SimpleNamespace(name="pour", args=("bowl",))],
        bundle,
    )
    assert no_hold_pour.score == 0.0

    unavailable_argument = evaluate_default_world_constraints(
        [set(), set()],
        [SimpleNamespace(name="clean", args=("bathtub_35",))],
        bundle,
        task_entity_types=("novel", "filing_cabinet"),
    )
    assert unavailable_argument.score == 0.0
    available_argument = evaluate_default_world_constraints(
        [set(), set()],
        [SimpleNamespace(name="clean", args=("bathtub_35",))],
        bundle,
        task_entity_types=("bathtub",),
    )
    assert available_argument.score == 1.0


def test_reward_bypass_and_blending():
    from domiknows.graph.logicalConstrain import nandL

    sample = {
        "task_id": "blend",
        "tl_goal": "closed(fridge.1)",
        "target_action_tokens": ["close", "fridge_1", EOS_TOKEN],
        "object_tokens": ("fridge_1",),
    }
    unconstrained = build_eai_world_graph("test_eai_world_no_constraints")
    baseline = evaluate_goal_satisfaction(sample["target_action_tokens"], sample)
    bypass = evaluate_goal_satisfaction(
        sample["target_action_tokens"], sample, world_bundle=unconstrained,
    )
    assert bypass["world_constraint_score"] is None
    assert bypass["rl_reward_score"] == baseline["is_success"]
    cached_reward = make_eai_reward_function(sample, world_bundle=unconstrained)
    assert cached_reward(sample["target_action_tokens"]).item() == 1.0
    assert cached_reward(sample["target_action_tokens"]).item() == 1.0
    assert cached_reward.cache_hits == 1

    constrained = build_eai_world_graph(
        "test_eai_world_reward_blend",
        (lambda bundle: nandL(bundle.states["open"], bundle.states["closed"]),),
    )
    result = evaluate_goal_satisfaction(
        sample["target_action_tokens"],
        sample,
        world_bundle=constrained,
        constraint_weight=0.25,
    )
    expected = result["is_success"] * (
        0.75 + 0.25 * result["world_constraint_score"]
    )
    assert result["rl_reward_score"] == expected
    failed_but_compliant = evaluate_goal_satisfaction(
        ["open", "fridge_1", EOS_TOKEN],
        sample,
        world_bundle=constrained,
        constraint_weight=0.25,
    )
    assert failed_but_compliant["is_success"] == 0.0
    assert failed_but_compliant["world_constraint_score"] == 1.0
    assert failed_but_compliant["rl_reward_score"] == 0.0
    dense = evaluate_goal_satisfaction(
        [EOS_TOKEN], sample, world_bundle=constrained, reward_mode="dense", constraint_weight=0.25,
    )
    assert dense["world_constraint_score"] is None
    assert dense["rl_reward_score"] == dense["recall"]


def test_complete_benchmark_bypass_and_scale():
    examples = load_eai_dataset("all", limit=None, max_steps=135, device="cpu")
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    world = build_eai_world_graph("test_eai_world_benchmark_bypass")
    default_world = build_eai_world_graph(
        "test_eai_world_benchmark_defaults",
        include_default_constraints=True,
    )
    largest = None
    precondition_trajectories = 0
    for sample in examples:
        prepared = prepare_eai_goal(sample, vocabulary)
        precondition_evaluation = evaluate_default_world_constraints(
            prepared.reference_states,
            prepared.reference_events,
            default_world,
        )
        if precondition_evaluation is not None:
            precondition_trajectories += 1
            assert precondition_evaluation.score == 1.0, (
                sample["task_id"],
                {
                    name: result
                    for name, result in precondition_evaluation.results.items()
                    if result["applicable"] and result["satisfied"] < 100.0
                },
            )
        if largest is None or len(prepared.entity_universe) > len(largest[0].entity_universe):
            largest = (prepared, sample)
        ordinary = evaluate_goal_satisfaction(sample["target_action_tokens"], sample, vocabulary)
        with_world = evaluate_goal_satisfaction(
            sample["target_action_tokens"], sample, vocabulary, world_bundle=world,
        )
        assert with_world["world_constraint_score"] is None
        assert with_world["rl_reward_score"] == ordinary["is_success"]
        ordinary_empty = evaluate_goal_satisfaction([vocabulary.eos_label], sample, vocabulary)
        with_world_empty = evaluate_goal_satisfaction(
            [vocabulary.eos_label], sample, vocabulary, world_bundle=world,
        )
        assert with_world_empty["world_constraint_score"] is None
        assert with_world_empty["rl_reward_score"] == ordinary_empty["is_success"]

    assert precondition_trajectories > 0

    prepared, sample = largest
    from reward import _simulate_events
    states, events = _simulate_events(
        list(prepared.reference_events),
        initial_state=set(prepared.initial_state),
        goal_facts=set(prepared.gold_state),
    )
    root = materialize_world_trajectory(prepared, states, events, world)
    binary_count = sum(
        node.ontologyNode is world.state and node.attributes.get("grounding_arity") == 2
        for node in _all_descendants(root)
    )
    dense_count = len(states) * len(prepared.entity_universe) ** 2
    assert binary_count <= len(states) * len(prepared.tracked_binary_pairs)
    assert binary_count < dense_count


def run_tests():
    tests = [
        test_registry_and_namespaces,
        test_materializes_actions_and_sparse_state_groundings,
        test_constraint_verification_and_aggregation,
        test_default_precondition_registry_and_verification,
        test_reward_bypass_and_blending,
        test_complete_benchmark_bypass_and_scale,
    ]
    for test in tests:
        test()
    print(f"WORLD_GRAPH_REGRESSION_DONE {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    run_tests()
