"""Independent goal/budget counterexamples for symbolic and native decoding."""
import json

import pytest
import torch

from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from ..native_planning import NativeConstraintProcessor, NativePlanFailure
from ..reachability import ReachablePlanDFA
from ..training import build_constraint_runtime

ENTITIES = ("apple", "bowl", "cup")
EOS = 200
TOKENS = {b: bytes([b]) for b in (9, 10, 13, *range(32, 127))}
TOKENS.update({201: b'[{"name":"pick","params":{"target_entity_name":',
               202: b'}},{"name":"place","params":{"target_container_name":', 203: b'}}]'})


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=3, max_operations=4, name_prefix="action_reachability")


def context(**overrides):
    facts = {
        "initial_state": {"held": {"right": None}, "negative": [["at", "cup", "bowl"]]},
        "capacities": {"right": 1},
        "entity_properties": {"apple": {"graspable": True}, "bowl": {"graspable": False},
                              "cup": {"graspable": False}},
        "destinations": {"apple": "bowl"},
        "compatibility": [{"subject": "apple", "destination": "bowl", "allowed": True}],
        "goals": {"positive": [["at", "apple", "bowl"]]},
    }
    facts.update(overrides)
    return {"entities": ENTITIES, "task_skill_sequence": ["pick", "place"], "planning_facts": facts}


def native(policy, **kwargs):
    return ReachablePlanDFA(TOKENS, EOS, policy=policy, entity_count=3,
                            max_operations=kwargs.pop("max_operations", 2),
                            patterns=(("pick", "place"),), causal=False, **kwargs)


def plan(destination=1, subject=0):
    return [{"name": "pick", "params": {"target_entity_name": subject}},
            {"name": "place", "params": {"target_container_name": destination}}]


def labels(value):
    return list(json.dumps(value, separators=(",", ":")).encode()) + [EOS]


def test_action_budget_counts_operations_not_tokens(runtime):
    query = runtime.dfa_for("vlabench", context()).action_reachability()
    assert query.can_reach(remaining_actions=2) is True
    assert query.can_reach(remaining_actions=1) is False
    assert query.can_reach(remaining_actions=0) is False


def test_native_goal_specific_masks_and_merged_tokens(runtime):
    policy = runtime.dfa_for("vlabench", context())
    dfa = native(policy)
    assert dfa.accepts(labels(plan()))
    assert not dfa.accepts(labels(plan(destination=2)))
    assert not dfa.accepts(labels(plan(subject=2)))
    assert dfa.accepts([201, ord("0"), 202, ord("1"), 203, EOS])
    state = dfa.step(dfa.start_state, 201)
    assert ord("0") in dfa.allowed_tokens(state)
    assert ord("2") not in dfa.allowed_tokens(state)
    state = dfa.start_state
    for token in labels(plan()):
        assert token in dfa.allowed_tokens(state)
        state = dfa.step(state, token)
    assert native(policy, max_operations=1).can_reach_goal() is False


def test_wrong_entity_goal_is_not_satisfied(runtime):
    query = runtime.dfa_for("vlabench", context(goals={"positive": [["at", "cup", "bowl"]]})).action_reachability()
    assert query.can_reach(remaining_actions=2) is False


def test_unknown_goal_is_distinct_from_impossibility(runtime):
    policy = runtime.dfa_for("vlabench", context(goals={"positive": [["unobserved", "apple"]]}))
    assert policy.action_reachability().can_reach(remaining_actions=2) is None
    assert policy.action_reachability(strict=False).can_reach(remaining_actions=2) is True
    dfa = native(policy)
    assert dfa.can_reach_goal() is None
    processor = NativeConstraintProcessor(dfa, 1, 200)
    with pytest.raises(NativePlanFailure, match="undetermined"):
        processor(torch.tensor([[0, ord("[")]]), torch.zeros(1, 204))


def test_missing_precondition_observations_are_not_proven_feasible(runtime):
    policy = runtime.dfa_for("vlabench", context(entity_properties={}))
    assert policy.action_reachability().can_reach(remaining_actions=2) is None
    assert policy.action_reachability(strict=False).can_reach(remaining_actions=2) is True


def test_already_satisfied_goal_allows_noop(runtime):
    ctx = context(initial_state={"positive": [["at", "apple", "bowl"]]})
    ctx.pop("task_skill_sequence")
    policy = runtime.dfa_for("vlabench", ctx)
    assert policy.action_reachability().can_reach(remaining_actions=0) is True
    assert native(policy).accepts(labels([]))


def test_expansion_limit_is_not_cached_as_unreachable(runtime):
    policy = runtime.dfa_for("vlabench", context())
    query = policy.action_reachability(max_state_expansions=1)
    for _ in range(2):
        with pytest.raises(SemanticSearchLimitExceeded):
            query.can_reach(remaining_actions=2)
    dfa = native(policy, max_state_expansions=1)
    with pytest.raises(SemanticSearchLimitExceeded):
        dfa.can_reach_goal()


def test_episode_caches_do_not_share_goals(runtime):
    good = runtime.dfa_for("vlabench", context()).action_reachability()
    bad = runtime.dfa_for("vlabench", context(goals={"positive": [["at", "cup", "bowl"]]})).action_reachability()
    assert good.can_reach(remaining_actions=2) is True
    assert bad.can_reach(remaining_actions=2) is False
    assert good.can_reach(remaining_actions=2) is True


def test_locally_executable_wrong_pick_has_no_goal_continuation(runtime):
    ctx = context(
        initial_state={"held": {"right": None}, "negative": [["at", "apple", "bowl"]]},
        entity_properties={"apple": {"graspable": True}, "cup": {"graspable": True}},
        destinations={"apple": "bowl", "cup": "bowl"},
        compatibility=[{"subject": x, "destination": "bowl", "allowed": True} for x in ("apple", "cup")])
    policy = runtime.dfa_for("vlabench", ctx)
    query = policy.action_reachability()
    wrong, known = query.advance(policy.start_state, "pick", {"target_entity_name": 2})
    assert wrong is not None and known  # Immediate compiled constraints allow picking the cup.
    assert query.can_reach(wrong, remaining_actions=1) is False
    dfa = native(policy)
    prefix = dfa.step(dfa.start_state, 201)
    assert ord("2") not in dfa.allowed_tokens(prefix)
    assert ord("0") in dfa.allowed_tokens(prefix)


def test_closed_container_requires_state_transition(runtime):
    entities = ("apple", "bowl", "cooler")
    ctx = context(
        initial_state={"held": {"right": None}, "positive": [["closed", "cooler"]],
                       "negative": [["at", "apple", "cooler"]]},
        entity_properties={"apple": {"graspable": True}, "cooler": {"graspable": False, "openable": True}},
        destinations={"apple": "cooler"},
        compatibility=[{"subject": "apple", "destination": "cooler", "allowed": True}],
        reusable_entities=["apple"],
        goals={"positive": [["at", "apple", "cooler"]]})
    ctx["entities"] = entities
    # Preserve the domain's two declared subtask patterns; do not invent a
    # pick/open/place pattern just because its physical effects are plausible.
    ctx["task_skill_sequence"] = ["pick", "open_door", "pick", "place"]
    policy = runtime.dfa_for("vlabench", ctx)
    query = policy.action_reachability()
    assert query.can_reach(remaining_actions=4) is True
    assert query.can_reach(remaining_actions=3) is False
    held, _ = query.advance(policy.start_state, "pick", {"target_entity_name": 0})
    assert query.advance(held, "place", {"target_container_name": 2})[0] is None
    dfa = ReachablePlanDFA(TOKENS, EOS, policy=policy, entity_count=3, max_operations=4,
                           patterns=(("pick", "open_door"), ("pick", "place")), causal=False)
    witness = [plan()[0], {"name": "open_door", "params": {"target_entity_name": 2}}, *plan(2)]
    assert dfa.accepts(labels(witness))
    assert not dfa.accepts(labels(plan(2)))


def test_logits_mask_survives_high_model_preference(runtime):
    dfa = native(runtime.dfa_for("vlabench", context()))
    processor = NativeConstraintProcessor(dfa, 1, 200)
    scores = torch.zeros(1, 204)
    scores[0, ord("2")] = 1e10
    masked = processor(torch.tensor([[0, 201]]), scores)
    assert masked[0, ord("2")].isneginf()
    assert masked[0, ord("0")].isfinite()


def test_disabled_goal_lc_changes_search_without_rule_duplication(runtime):
    from dataclasses import replace
    from ..constraint_compiler import WorldConstraintDFA
    original = runtime.dfa_for("vlabench", context(goals={"positive": [["at", "cup", "bowl"]]}))
    program = replace(original.program, rules=tuple(r for r in original.program.rules if r.name != "grounded__goal_completion"))
    relaxed = WorldConstraintDFA(original.base_dfa, runtime.vocabulary, runtime.world_bundle, program,
                                  ENTITIES, requirements=original.adapter.requirements)
    assert original.action_reachability().can_reach(remaining_actions=2) is False
    assert relaxed.action_reachability().can_reach(remaining_actions=2) is True


@pytest.mark.parametrize("destination", [1, 2])
def test_frozen_native_planner_uses_bound_policy(runtime, destination):
    from types import SimpleNamespace
    from ..native_planning import FrozenNativePlanner

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

    # A deterministic VLM stub exercises the real generator's policy binding
    # and post-generation acceptance without downloading a model or using GPU.
    planner = FrozenNativePlanner.__new__(FrozenNativePlanner)
    planner.processor = Processor()
    planner.tokenizer = SimpleNamespace(eos_token_id=EOS,
        decode=lambda ids, **kw: bytes(i for i in ids if i < 128).decode())
    planner.model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
        generate=lambda **kw: torch.tensor([[0, *labels(plan(destination))]]))
    planner.token_bytes = TOKENS
    planner.dfas, planner.tables = {}, {}
    planner.dp_cache_directory = None
    example = SimpleNamespace(segmented_image_paths=(), image_paths=(), entities=ENTITIES, instruction="put apple in bowl")
    kwargs = dict(mode="constrained", max_operations=2, object_constraints="none",
                  reachability_policy=runtime.dfa_for("vlabench", context()))
    if destination == 1:
        raw, output = planner.generate(example, **kwargs)
        assert json.loads(raw) == plan()
        assert output == labels(plan())
    else:
        with pytest.raises(NativePlanFailure):
            planner.generate(example, **kwargs)
