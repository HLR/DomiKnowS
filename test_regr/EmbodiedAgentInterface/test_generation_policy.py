"""Regression checks for graph-defined EAI generation policy compilation."""

import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import load_eai_dataset, transition_model_entity_types
from graph import create_generation_graph
from main import (
    action_object_constraint_tokens_from_examples,
    action_tokens_from_examples,
    action_tokens_requiring_object_from_examples,
    generation_vocab_from_examples,
    object_tokens_from_examples,
)
from domiknows.generation import (
    analyze_generation_constraints,
    bind_contextual_dfa,
    constraints_to_dfa_from_graph,
)


def _synthetic_policy():
    graph, bundle = create_generation_graph(
        max_steps=8,
        vocab=["<eos>", "open", "close", "clean", "find", "sleep", "standup", "cup", "door"],
        object_tokens=["cup", "door"],
        action_tokens=["open", "close"],
        action_sequence_tokens=["open", "close", "clean", "find", "sleep", "standup"],
        action_object_constraint_tokens={
            "open": ["door"],
            "close": ["door"],
            "clean": ["cup", "door"],
        },
    )
    dfa = constraints_to_dfa_from_graph(
        graph, bundle, on_unsupported="raise", minimize=False
    )
    label = bundle.vocabulary.label_for_token
    return graph, bundle, dfa, label


def test_every_eai_policy_is_a_supported_graph_constraint():
    graph, bundle, dfa, _label = _synthetic_policy()
    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="raise")
    relevant = [analysis for analysis in analyses if analysis.relevant]
    assert relevant
    assert all(analysis.supported for analysis in relevant)
    assert bundle.sequence_start is not None
    assert not hasattr(dfa, "overlays")
    contextual = {
        constraint.name: constraint
        for constraint in graph.logicalConstrains.values()
        if getattr(constraint, "_generation_contextual_dfa_specs", ())
    }
    assert set(contextual) == {
        "generated_object_exists_in_task_world",
        "generated_semantic_action_is_task_relevant",
    }
    assert all(constraint.__class__.__name__ == "ifL" for constraint in contextual.values())
    assert all(not constraint.active for constraint in contextual.values())
    assert graph.findConcept("eai_task_entity_available") is not None
    assert graph.findConcept("eai_task_action_permitted") is not None


def test_eai_policy_language_semantics():
    _graph, _bundle, dfa, label = _synthetic_policy()
    eos = label("<eos>")
    assert dfa.accepts([label("open"), label("door"), eos])
    assert dfa.accepts([label("sleep"), label("standup"), eos])
    assert not dfa.accepts([eos])
    assert not dfa.accepts([label("door"), eos])
    assert not dfa.accepts([label("open"), eos])
    assert not dfa.accepts([label("open"), label("cup"), eos])
    assert not dfa.accepts([label("sleep"), label("door"), eos])
    assert not dfa.accepts([label("open"), label("door"), label("cup"), eos])


def test_task_world_context_rejects_objects_absent_from_transition_model():
    graph, bundle, dfa, label = _synthetic_policy()
    conditioned = bind_contextual_dfa(
        dfa,
        graph,
        {
            "generation_entity_types": ("door",),
            "semantic_action_permissions": ("clean",),
        },
    )
    eos = label("<eos>")
    assert conditioned.accepts([label("clean"), label("door"), eos])
    assert not conditioned.accepts([label("clean"), label("cup"), eos])
    # The same token remains legal in an unguarded scene-navigation role.
    assert conditioned.accepts([label("find"), label("cup"), eos])

    book_task = bind_contextual_dfa(
        dfa,
        graph,
        {
            "generation_entity_types": ("door", "cup"),
            "semantic_action_permissions": (),
        },
    )
    assert not book_task.accepts([label("clean"), label("door"), eos])


def test_transition_model_entities_are_parsed_without_reference_actions():
    transition_model = """
    (define (problem sample)
      (:objects
        character - character
        filing_cabinet bathroom novel home_office - object
      )
      (:init (closed filing_cabinet)))
    """
    assert transition_model_entity_types(transition_model) == (
        "bathroom",
        "filing_cabinet",
        "home_office",
        "novel",
    )


def test_all_reference_trajectories_are_accepted_with_bounded_compilation():
    examples = load_eai_dataset("all", limit=None, max_steps=30, device="cpu")
    graph, bundle = create_generation_graph(
        max_steps=30,
        vocab=generation_vocab_from_examples(examples),
        object_tokens=object_tokens_from_examples(examples),
        action_tokens=action_tokens_requiring_object_from_examples(examples),
        action_sequence_tokens=action_tokens_from_examples(examples),
        action_object_constraint_tokens=action_object_constraint_tokens_from_examples(examples),
    )
    started = time.perf_counter()
    dfa = constraints_to_dfa_from_graph(
        graph, bundle, on_unsupported="raise", minimize=False
    )
    elapsed = time.perf_counter() - started
    assert len(examples) == 438
    rejected = [
        (
            example["task_id"],
            example["target_action_tokens"],
            example["task_entity_types"],
        )
        for example in examples
        if not bind_contextual_dfa(dfa, graph, example).accepts(
            [int(label) for label in example["target_action_labels"]]
        )
    ]
    assert not rejected, rejected[:10]
    assert len(dfa.states) < 5000
    assert elapsed < 30.0


def run_tests():
    tests = [
        test_every_eai_policy_is_a_supported_graph_constraint,
        test_eai_policy_language_semantics,
        test_task_world_context_rejects_objects_absent_from_transition_model,
        test_transition_model_entities_are_parsed_without_reference_actions,
        test_all_reference_trajectories_are_accepted_with_bounded_compilation,
    ]
    for test in tests:
        test()
    print(f"EAI_GENERATION_POLICY_DONE {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    run_tests()
