"""Independent syntax and initial-observation support counterexamples."""
import json
from dataclasses import replace

import pytest

from ..constraint_compiler import WorldConstraintDFA
from ..native_planning import NativePlanDFA, validate_native_plan
from ..official.decoding_graph import build_decoding_graph
from ..official.knowledge_dfa import KnowledgePlanDFA
from ..plan_codec import plan_to_tokens
from ..semantics import operation_schema_errors
from ..training import build_constraint_runtime
from ..visibility import visible_entity_ids, visibility_from_segmentation
from ..world_graph import canonicalize_plan, materialize_plan, validate_plan, verify_plan_constraints

ENTITIES = ("apple", "bowl", "cup", "button")
EOS = 200
TOKENS = {b: bytes([b]) for b in (9, 10, 13, *range(32, 127))}
TOKENS[201] = b'[{"name":"pick","params":{"target_entity_name":'
TOKENS[202] = b'}},{"name":"place","params":{"target_container_name":'
TOKENS[203] = b'}}]'


def op(skill, **params):
    return {"name": skill, "params": params}


def plan(subject=0, destination=1):
    return [op("pick", target_entity_name=subject), op("place", target_container_name=destination)]


def labels(value):
    return list(json.dumps(value, separators=(",", ":")).encode()) + [EOS]


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=4, max_operations=4, name_prefix="syntax_visibility")


def bound(runtime, visible=None):
    return runtime.dfa_for("vlabench", {"entities": ENTITIES, "visible_entities": visible})


def accepted(runtime, policy, value):
    return policy.accepts([runtime.vocabulary.label_for_token(t)
                          for t in plan_to_tokens(value, ENTITIES, world=runtime.world_bundle)])


@pytest.mark.parametrize("value", [
    plan(), [op("pick", target_entity_name=0), op("insert", target_container_name=1)],
    [op("pick", target_entity_name=0), op("pour", target_container_name=1)],
    [op("pick", target_entity_name=0), op("push", target_container_name=1)],
    [op("pick", target_entity_name=0), op("open_door", target_entity_name=1)],
    [op("press", target_entity_name=3)],
])
def test_all_explicit_reference_roles_require_visual_support(runtime, value):
    # These ids exist in metadata, but the initial observation supports only apple.
    policy = bound(runtime, [0])
    assert not accepted(runtime, policy, value)
    assert not validate_plan(value, entity_table=ENTITIES, policy=policy).valid
    root = materialize_plan(value, ENTITIES, runtime.world_bundle, policy=policy)
    result = verify_plan_constraints(root, runtime.world_bundle)
    assert result.results["observation_supported_references"]["satisfied"] < 100


def test_missing_and_empty_support_are_distinct(runtime):
    missing, empty, supported = bound(runtime), bound(runtime, []), bound(runtime, ["apple", "bowl"])
    assert accepted(runtime, missing, plan())
    assert not accepted(runtime, empty, plan())
    assert accepted(runtime, supported, plan())
    for policy, status, ids in ((missing, "unavailable", None), (empty, "supplied", []), (supported, "supplied", [0, 1])):
        root = materialize_plan(plan(), ENTITIES, runtime.world_bundle, policy=policy)
        assert root.attributes["visibility_report"] == {"status": status, "visible_ids": ids}


@pytest.mark.parametrize("bad", [True, "apple", {"apple": True}, [True], [1.0], [-1], [4], ["ghost"]])
def test_invalid_visual_support_is_rejected(bad):
    with pytest.raises(ValueError):
        visible_entity_ids(bad, ENTITIES)


def test_sparse_native_visibility_is_not_an_id_bound():
    dfa = KnowledgePlanDFA(TOKENS, EOS, entity_count=12, max_operations=2, causal=False,
                           scene_bound=12, visible_entities=[1, 10])
    assert dfa.accepts(labels(plan(1, 10)))
    assert not dfa.accepts(labels(plan(1, 11)))
    prefix = dfa.start_state
    for token in [201, ord("1")]:
        prefix = dfa.step(prefix, token)
    assert ord("0") in dfa.allowed_tokens(prefix)  # 10 is seen.
    assert ord("1") not in dfa.allowed_tokens(prefix)  # 11 is hidden.
    assert dfa.accepts([201, ord("1"), 202, ord("1"), ord("0"), 203, EOS])
    assert not dfa.accepts([201, ord("1"), 202, ord("1"), ord("1"), 203, EOS])
    assert not validate_native_plan(plan(1, 11), [str(i) for i in range(12)], visible_entities=[1, 10])[0]


def test_visibility_is_enforced_by_editable_logical_constraints(runtime):
    policy = bound(runtime, [0])
    relaxed_program = replace(policy.program, rules=tuple(r for r in policy.program.rules
                                                         if r.name != "observation_supported_references"))
    relaxed = WorldConstraintDFA(policy.base_dfa, runtime.vocabulary, runtime.world_bundle,
                                 relaxed_program, ENTITIES, requirements=policy.adapter.requirements)
    assert not accepted(runtime, policy, plan())
    assert accepted(runtime, relaxed, plan())
    assert validate_plan(plan(), entity_table=ENTITIES, policy=relaxed).valid
    graph = build_decoding_graph("visibility_editable")
    for rule in graph.graph.logicalConstrains.values():
        if rule.name == "observation_supported_references":
            rule.active = False
    dfa = KnowledgePlanDFA(TOKENS, EOS, entity_count=4, max_operations=2, causal=False,
                           visible_entities=[0], decoding_graph=graph)
    assert dfa.accepts(labels(plan()))


def test_visibility_and_reachability_compose(runtime):
    ctx = {"entities": ENTITIES, "task_skill_sequence": ["pick", "place"],
           "visible_entities": [0], "planning_facts": {"goals": {"positive": [["at", "apple", "bowl"]]}}}
    hidden = runtime.dfa_for("vlabench", ctx)
    assert hidden.action_reachability(strict=False).can_reach(remaining_actions=2) is False
    seen = runtime.dfa_for("vlabench", {**ctx, "visible_entities": [0, 1]})
    assert seen.action_reachability(strict=False).can_reach(remaining_actions=2) is True


@pytest.mark.parametrize("params", [
    {"target_entity_name": 0, "unexpected": 1}, {"target_entity_name": True},
    {"target_entity_name": 0.0}, {"target_entity_name": []}, {},
    {"target_entity_name": 0, "target_container_name": 1},
])
def test_symbolic_syntax_cannot_be_lost_during_encoding(runtime, params):
    value = [op("pick", **params), op("lift")]
    assert not validate_plan(value, entity_table=ENTITIES).valid
    with pytest.raises(ValueError):
        plan_to_tokens(value, ENTITIES, world=runtime.world_bundle)


def test_undeclared_argument_fails_world_schema_lc(runtime):
    value = [op("pick", target_entity_name=0, bogus=1), op("lift")]
    root = materialize_plan(value, ENTITIES, runtime.world_bundle)
    result = verify_plan_constraints(root, runtime.world_bundle)
    assert result.results["operation_schema_is_valid"]["satisfied"] < 100


@pytest.mark.parametrize("raw", [
    '[{"name":"pick","name":"press","params":{"target_entity_name":3}}]',
    '[{"name":"pick","params":{"target_entity_name":0,"target_entity_name":1}}]',
    '[{"name":12,"params":{}}]',
])
def test_ambiguous_or_nonstrings_are_not_normalized_into_valid_actions(raw):
    with pytest.raises(ValueError):
        canonicalize_plan(raw)
    assert not validate_plan(raw, entity_table=ENTITIES).valid
    assert not validate_native_plan(raw, ENTITIES)[0]


@pytest.mark.parametrize("raw", [
    '[{"name":"pick","params":{"target_entity_name":true}}]',
    '[{"name":"pick","params":{"target_entity_name":0.0}}]',
    '[{"name":"pick","params":{"target_entity_name":0,"bogus":1}}]',
    '[{"name":"pick","params":{}}]',
    '[{"name":"pick","params":{"target_entity_name":0}},]',
])
def test_native_grammar_rejects_malformed_serializations(raw):
    dfa = NativePlanDFA(TOKENS, EOS, entity_count=4, max_operations=2)
    assert not dfa.accepts(list(raw.encode()) + [EOS])


def test_initial_multiview_segmentation_support_requires_explicit_bindings():
    import numpy as np
    bindings = {12: "apple", 77: "bowl", 99: "cup"}
    masks = [np.array([[0, 12], [12, 0]]), np.array([[77, 0], [0, 88]])]
    assert visibility_from_segmentation(masks, bindings, ENTITIES) == frozenset({0, 1})
    # Neither an absent bound cup nor an unbound visible label invents support.
    assert visibility_from_segmentation([np.zeros((2, 2), dtype=int)], bindings, ENTITIES) == frozenset()
    with pytest.raises(ValueError):
        visibility_from_segmentation([np.zeros((2, 2), dtype=float)], bindings, ENTITIES)
    with pytest.raises(ValueError):
        visibility_from_segmentation(masks, {12: "ghost"}, ENTITIES)


def test_native_initial_support_is_frozen_across_operations():
    dfa = KnowledgePlanDFA(TOKENS, EOS, entity_count=4, max_operations=4, causal=False,
                           visible_entities=[0, 1])
    assert dfa.accepts(labels(plan() + plan()))
    assert not dfa.accepts(labels(plan() + plan(subject=2)))


def test_planning_example_preserves_visibility_in_context():
    from ..dataset import PlanningExample
    sample = PlanningExample("sample", "put apple in bowl", tuple(plan()), (), (), None,
                              entities=ENTITIES, visible_entities=(0, 1))
    assert sample.as_reward_item()["visible_entities"] == (0, 1)


@pytest.mark.parametrize("visible", [(0, 1), (0,), ()])
def test_generator_enforces_example_visibility_without_reachability(runtime, visible):
    from types import SimpleNamespace
    import torch
    from ..native_planning import FrozenNativePlanner, NativePlanFailure

    class Batch(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Batch(input_ids=torch.tensor([[0]]))

    planner = FrozenNativePlanner.__new__(FrozenNativePlanner)
    planner.processor = Processor()
    planner.tokenizer = SimpleNamespace(eos_token_id=EOS,
        decode=lambda ids, **kw: bytes(i for i in ids if i < 128).decode())
    planner.model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
        generate=lambda **kw: torch.tensor([[0, *labels(plan())]]))
    planner.token_bytes = TOKENS
    planner.dfas, planner.tables, planner.dp_cache_directory = {}, {}, None
    example = SimpleNamespace(segmented_image_paths=(), image_paths=(), entities=ENTITIES,
                               instruction="put apple in bowl", visible_entities=visible)
    if visible == (0, 1):
        raw, _ = planner.generate(example, mode="constrained", max_operations=2, object_constraints="none")
        assert json.loads(raw) == plan()
    else:
        with pytest.raises(NativePlanFailure):
            planner.generate(example, mode="constrained", max_operations=2, object_constraints="none")
