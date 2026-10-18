"""Task restrictions are compiled world LCs, shared with final verification."""
import pytest

from domiknows.generation import find_predicate_monitor
from .constraint_compiler import WorldConstraintDFA
from .graph import plan_to_tokens
from .training import build_constraint_runtime
from .world_graph import build_vlabench_world_graph, materialize_plan, verify_plan_constraints


def encode(runtime, plan, entities):
    return [runtime.vocabulary.label_for_token(token) for token in
            plan_to_tokens(plan, entities, world=runtime.world_bundle)]


def op(skill, target):
    key = "target_container_name" if skill == "place" else "target_entity_name"
    return {"name": skill, "params": {key: target}}


def verified(runtime, plan, entities, *, context=None, policy=None):
    result = verify_plan_constraints(materialize_plan(
        plan, entities, runtime.world_bundle, task_context=context, policy=policy,
    ), runtime.world_bundle)
    return result is None or result.score == 1.0


def test_button_presence_does_not_invent_a_press_goal():
    runtime = build_constraint_runtime(max_entities=3, max_operations=2, name_prefix="no_implicit_qa")
    entities = ("apple", "bowl", "button0")
    policy = runtime.dfa_for("vlabench", {"entities": entities})
    assert isinstance(policy, WorldConstraintDFA)
    plan = [op("pick", 0), op("place", 1)]
    assert policy.accepts(encode(runtime, plan, entities))
    assert verified(runtime, plan, entities, policy=policy)
    # Press applicability is also a logical declaration, not a contextual DFA.
    wrong = [op("press", 1)]
    assert not policy.accepts(encode(runtime, wrong, entities))
    assert not verified(runtime, wrong, entities, policy=policy)


def test_qa_keywords_do_not_guess_an_answer_or_override_explicit_target():
    runtime = build_constraint_runtime(max_entities=3, max_operations=1, name_prefix="explicit_qa_target")
    entities = ("button0", "button1", "button2")
    context = {"entities": entities, "instruction": "Which object has the largest density?"}
    policy = runtime.dfa_for("vlabench", context)
    assert all(policy.accepts(encode(runtime, [op("press", i)], entities)) for i in range(3))
    context.update(task="select_painting", task_target_entity="button0")
    policy = runtime.dfa_for("vlabench", context)
    for i in range(3):
        plan = [op("press", i)]
        assert policy.accepts(encode(runtime, plan, entities)) == (i == 0)
        assert verified(runtime, plan, entities, context=context) == (i == 0)
        assert verified(runtime, plan, entities, policy=policy) == (i == 0)


def test_task_sequences_targets_and_completion_use_actual_logical_constraints():
    world = build_vlabench_world_graph("editable_task_rules")
    entities = ("apple", "bowl", "button0")
    runtime = build_constraint_runtime(world, max_entities=3, max_operations=2, name_prefix="task_rule_edit")
    context = {"entities": entities, "task_skill_sequence": ("pick", "place"), "task_target_entity": "apple"}
    policy = runtime.dfa_for("vlabench", context)
    assert find_predicate_monitor(policy).program.coverage.compiled
    assert policy.allowed_tokens(policy.start_state) == {runtime.vocabulary.label_for_token("skill:pick")}
    good, wrong = [op("pick", 0), op("place", 1)], [op("pick", 1), op("place", 0)]
    assert policy.accepts(encode(runtime, good, entities))
    assert not policy.accepts(encode(runtime, wrong, entities))
    rule = next(lc for lc in world.graph.logicalConstrains.values() if lc.name == "task_initial_target")
    rule.active = False
    relaxed_runtime = build_constraint_runtime(world, max_entities=3, max_operations=2, name_prefix="relaxed_task_rule")
    relaxed = relaxed_runtime.dfa_for("vlabench", context)
    assert relaxed.accepts(encode(relaxed_runtime, wrong, entities))
    assert verified(relaxed_runtime, wrong, entities, policy=relaxed)
    assert not policy.accepts(encode(runtime, wrong, entities))


def test_short_task_prefix_and_wrong_task_sequence_fail_both_paths():
    runtime = build_constraint_runtime(max_entities=2, max_operations=2, name_prefix="task_complete")
    entities = ("apple", "bowl")
    context = {"entities": entities, "task_skill_sequence": ("pick", "place")}
    policy = runtime.dfa_for("vlabench", context)
    wrong = [op("pick", 0), {"name": "lift", "params": {}}]
    assert not policy.accepts(encode(runtime, wrong, entities))
    assert not verified(runtime, wrong, entities, policy=policy)
    prefix = [op("pick", 0)]
    assert not policy.accepts(encode(runtime, prefix, entities))
    assert not verified(runtime, prefix, entities, policy=policy)


@pytest.mark.parametrize("context", [
    {"task_skill_sequence": ()}, {"task_skill_sequence": ("teleport",)},
    {"task_target_entity": "missing"},
])
def test_invalid_task_facts_fail_closed(context):
    runtime = build_constraint_runtime(max_entities=2, max_operations=1, name_prefix="invalid_task_fact")
    with pytest.raises(ValueError):
        runtime.dfa_for("vlabench", {"entities": ("apple", "button0"), **context})



def test_disabling_task_rules_removes_restrictions_without_changing_decoder():
    world = build_vlabench_world_graph("disabled_task_rules")
    for rule in world.graph.logicalConstrains.values():
        if rule.name in {"task_skill_sequence", "task_initial_target", "task_sequence_completion"}:
            rule.active = False
    runtime = build_constraint_runtime(world, max_entities=3, max_operations=2, name_prefix="no_task_masks")
    entities = ("apple", "bowl", "button0")
    context = {"entities": entities, "task_skill_sequence": ("press",), "task_target_entity": "button0"}
    policy = runtime.dfa_for("vlabench", context)
    plan = [op("pick", 0), op("place", 1)]
    assert policy.accepts(encode(runtime, plan, entities))
    assert verified(runtime, plan, entities, policy=policy)


def test_no_default_constraints_has_no_hidden_button_or_pointer_filter():
    world = build_vlabench_world_graph("no_hidden_rule_wrappers", include_default_constraints=False)
    runtime = build_constraint_runtime(world, max_entities=3, max_operations=1, name_prefix="no_hidden_masks")
    entities = ("apple", "button0")
    policy = runtime.dfa_for("vlabench", {"entities": entities})
    for target in (0, 2):
        plan = [op("press", target)]
        assert policy.accepts(encode(runtime, plan, entities))
        assert verified(runtime, plan, entities, policy=policy)



def test_bounded_task_language_and_masks_match_completed_world_verification():
    from test_regr.generation.constraint_equivalence import assert_bounded_equivalence
    from .graph import tokens_to_plan
    runtime = build_constraint_runtime(max_entities=3, max_operations=2, name_prefix="task_language_audit")
    entities = ("apple", "bowl", "button0")
    context = {"entities": entities, "task_skill_sequence": ("pick", "place"), "task_target_entity": "apple"}
    policy = runtime.dfa_for("vlabench", context)
    def oracle(sequence):
        plan = tokens_to_plan([runtime.vocabulary.token_for_label(label) for label in sequence], world=runtime.world_bundle)
        return verified(runtime, plan, entities, context=context)
    audit = assert_bounded_equivalence(
        runtime.dfa, policy, oracle, max_steps=7, terminal_symbols={runtime.vocabulary.eos_label},
    )
    assert audit.accepted == 1 and audit.rejected > 0
    assert audit.checked_masks > audit.checked_prefixes



def test_episode_metadata_supplies_task_facts_without_instruction_heuristics():
    runtime = build_constraint_runtime(max_entities=2, max_operations=2, name_prefix="metadata_task_facts")
    entities = ("apple", "bowl")
    context = {"entities": entities, "episode_id": "validation/select_fruit/example0"}
    policy = runtime.dfa_for("vlabench", context)
    wrong = [op("pick", 0), {"name": "lift", "params": {}}]
    assert not policy.accepts(encode(runtime, wrong, entities))
    assert not verified(runtime, wrong, entities, context=context)
