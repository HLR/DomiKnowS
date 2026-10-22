"""Independent checks of declared, prefix-aware operation progress."""
from itertools import product

import pytest

from ..constraint_compiler import VLABenchPredicateAdapter, compile_world_constraints
from ..pattern_runtime import pattern_progress_for, trace_progress, _cached_progress
from ..task_policy import TaskRequirements, task_requirements
from ..training import build_constraint_runtime
from ..world_graph import build_vlabench_world_graph, materialize_plan, verify_plan_constraints


def operation(skill, target=0):
    role = {"pick": "target_entity_name", "press": "target_entity_name", "open_door": "target_entity_name",
            "place": "target_container_name", "pour": "target_container_name", "insert": "target_container_name"}.get(skill)
    return {"name": skill, "params": {} if role is None else {role: target}}


def nodes(root, concept):
    pending = list(root.getChildDataNodes() or ())
    while pending:
        node = pending.pop()
        if node.ontologyNode is concept:
            yield node
        pending.extend(node.getChildDataNodes() or ())


def truth(node, concept):
    return bool(node.attributes[f"<{concept.name}>"].argmax().item())


def reference(sequence, patterns, *, prefix=False):
    """Recursive segmentation, independent of automata and graph compilation."""
    if not sequence:
        return True
    for pattern in patterns:
        if prefix and len(sequence) < len(pattern) and tuple(sequence) == tuple(pattern[:len(sequence)]):
            return True
        if tuple(sequence[:len(pattern)]) == tuple(pattern) and reference(sequence[len(pattern):], patterns, prefix=prefix):
            return True
    return False


def test_projected_progress_matches_independent_bounded_segmentation():
    world = build_vlabench_world_graph("progress_reference")
    progress = pattern_progress_for(world, TaskRequirements())
    for length in range(5):
        for sequence in product(("pick", "lift", "place", "pour", "press"), repeat=length):
            viable, complete = trace_progress(progress, sequence)
            assert viable == tuple(reference(sequence[:i], world.subtask_patterns, prefix=True)
                                   for i in range(1, length + 1))
            assert complete == (bool(sequence) and reference(sequence, world.subtask_patterns))
    assert not trace_progress(progress, ("unknown",))[1]


def test_transitions_report_prefix_failure_without_poisoning_prior_pairs():
    world = build_vlabench_world_graph("progress_local")
    plan = [operation("pick", 0), operation("place", 1), operation("pick", 2), operation("press", 3)]
    root = materialize_plan(plan, ("apple", "bowl", "pear", "button0"), world)
    transitions = sorted(nodes(root, world.transition), key=lambda n: n.instanceID)
    assert [truth(node, world.allowed_transition) for node in transitions] == [True, True, False]
    assert not truth(root, world.pattern_complete)


def test_argument_errors_do_not_change_pattern_observations():
    world = build_vlabench_world_graph("progress_bad_argument")
    plan = [operation("pick", 99), operation("place", 1)]
    root = materialize_plan(plan, ("apple", "bowl"), world)
    assert all(truth(node, world.allowed_transition) for node in nodes(root, world.transition))
    assert truth(root, world.pattern_complete)
    result = verify_plan_constraints(root, world)
    assert result.score < 1
    assert result.results["transition_matches_subtask_pattern"]["satisfied"] == 100


@pytest.mark.parametrize("skills,starts,complete", [
    (("pick",), True, False), (("lift",), False, False),
    (("press",), True, True), (("lift", "place"), False, False),
    (("pick", "pour"), True, True), (("pick", "pour", "place"), True, True),
    (("pick", "place", "pick", "lift"), True, True), ((), None, False),
])
def test_boundaries_and_ambiguous_subtask_prefixes(skills, starts, complete):
    world = build_vlabench_world_graph("progress_boundary")
    root = materialize_plan([operation(skill, 0 if skill == "pick" else 1) for skill in skills],
                            ("apple", "button0"), world)
    observed = sorted(nodes(root, world.operation), key=lambda n: n.instanceID)
    if observed:
        assert truth(observed[0], world.pattern_start_allowed) == starts
    assert truth(root, world.pattern_complete) == complete
    if not skills or not starts or not complete:
        assert verify_plan_constraints(root, world).score < 1


def test_decoding_events_match_materialized_progress():
    world = build_vlabench_world_graph("progress_events")
    from ..plan_codec import PlanVocabulary
    vocabulary = PlanVocabulary.from_world(world, max_entities=2)
    adapter = VLABenchPredicateAdapter(vocabulary, world, ("apple", "bowl"))
    state = adapter.start_state
    events = []
    # Bypass the grammar layer intentionally: the predicate adapter must observe
    # the illegal third operation rather than asserting allowed_transition=True.
    tokens = ("skill:pick", "arg:target_entity_name", "obj:0", "skill:place",
              "arg:target_container_name", "obj:1", "skill:lift", "<eos>")
    for token in tokens:
        step = adapter.step(state, vocabulary.label_for_token(token))
        assert step is not None
        state = step.state
        events.extend(step.events)
    assert [event.values["allowed_transition"] for event in events if event.domain == "transition"] == [True, False]
    assert not compile_world_constraints(world).check_events(events)
    root = materialize_plan([operation("pick"), operation("place", 1), operation("lift")], ("apple", "bowl"), world)
    assert [truth(n, world.allowed_transition) for n in sorted(nodes(root, world.transition), key=lambda n: n.instanceID)] == [True, False]


def test_held_continuation_uses_the_same_progress_in_both_paths():
    world = build_vlabench_world_graph("progress_continuation")
    context = {"entities": ("apple", "bowl"), "task_skill_sequence": ("lift", "place"),
               "planning_facts": {"initial_state": {"held": {"right": "apple"}}}}
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="progress_continuation")
    policy = runtime.dfa_for("vlabench", context)
    plan = [operation("lift"), operation("place", 1)]
    from ..plan_codec import plan_to_tokens
    assert policy.accepts([runtime.vocabulary.label_for_token(token) for token in plan_to_tokens(plan, world=world)])
    root = materialize_plan(plan, context["entities"], world, task_context=context, policy=policy)
    assert truth(root, world.pattern_complete)
    assert verify_plan_constraints(root, world).score == 1


def test_actual_grammar_edit_changes_projected_progress(monkeypatch):
    from .. import graph
    world = build_vlabench_world_graph("progress_edited_declaration")
    assert not trace_progress(pattern_progress_for(world, TaskRequirements()), ("lift", "place"))[1]
    original = graph._declare_plan_grammar

    def without_start_guard(*args, **kwargs):
        formula = original(*args, **kwargs)
        # Remove the actual declared start condition, retaining exactly-one,
        # state/label and adjacent-state formulas. No names drive compilation.
        formula.e = tuple(formula.e[:-2]) + tuple(formula.e[-1:])
        return formula

    monkeypatch.setattr(graph, "_declare_plan_grammar", without_start_guard)
    assert trace_progress(pattern_progress_for(world, TaskRequirements()), ("lift", "place"))[1]


def test_progress_compilation_reused_across_scenes_and_plan_lengths():
    world = build_vlabench_world_graph("progress_cache")
    _cached_progress.cache_clear()
    first = pattern_progress_for(world, task_requirements({"entities": ("apple",)}, world, ("apple",)))
    second = pattern_progress_for(world, task_requirements({"entities": ("pear", "bowl")}, world, ("pear", "bowl")))
    assert first is second
    assert _cached_progress.cache_info().misses == 1


def test_policy_budget_is_preserved_in_world_progress():
    world = build_vlabench_world_graph("progress_budget")
    runtime = build_constraint_runtime(world, max_entities=1, max_operations=1, name_prefix="progress_budget")
    policy = runtime.dfa_for("vlabench", {"entities": ("button0",)})
    root = materialize_plan([operation("press"), operation("press")], ("button0",), world, policy=policy)
    assert not truth(root, world.pattern_complete)
    assert verify_plan_constraints(root, world).score < 1


def test_completed_prefix_and_permitted_empty_plan():
    world = build_vlabench_world_graph("progress_completed_prefix")
    context = {"entities": ("apple", "bowl"),
               "planning_facts": {"completed_skill_prefix": ["pick"],
                                  "initial_state": {"held": {"right": "apple"}}}}
    requirements = task_requirements(context, world, context["entities"])
    assert trace_progress(pattern_progress_for(world, requirements), ("lift", "place"))[1]
    from ..plan_codec import PlanVocabulary
    vocabulary = PlanVocabulary.from_world(world, max_entities=2)
    adapter = VLABenchPredicateAdapter(vocabulary, world, context["entities"], requirements=requirements)
    first = adapter.step(adapter.start_state, vocabulary.label_for_token("skill:lift"))
    operation_events = [event for event in first.events if event.domain == "operation" and event.phase == "immediate"]
    assert len(operation_events) == 1 and operation_events[0].values["pattern_first_operation"]
    assert not any(event.domain == "transition" for event in first.events)
    context["planning_facts"] = {"initial_state": {"positive": [["at", "apple", "bowl"]]},
                                 "goals": {"positive": [["at", "apple", "bowl"]]}}
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="progress_empty")
    policy = runtime.dfa_for("vlabench", context)
    assert policy.accepts((runtime.vocabulary.eos_label,))
    root = materialize_plan([], context["entities"], world, policy=policy)
    assert truth(root, world.pattern_complete)
    assert verify_plan_constraints(root, world).score == 1


def test_nonaccepting_eos_sink_does_not_allow_a_progress_state():
    from domiknows.generation.dfa.core import DFA
    from ..pattern_runtime import project_skill_progress
    from ..plan_codec import PlanVocabulary
    world = build_vlabench_world_graph("progress_sink")
    vocabulary = PlanVocabulary.from_world(world, max_entities=1)
    pick = vocabulary.label_for_token("skill:pick")
    eos = vocabulary.eos_label
    raw = DFA(frozenset((0, 1, 2)), frozenset((pick, eos)),
              {(0, pick): 1, (0, eos): 2, (1, eos): 2, (2, eos): 2}, 0, frozenset())
    assert not trace_progress(project_skill_progress(raw, vocabulary), ("pick",))[1]
