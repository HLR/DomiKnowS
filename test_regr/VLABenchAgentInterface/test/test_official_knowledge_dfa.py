import json
from itertools import product

import pytest

from ..native_planning import NativePlanDFA
from ..object_grounding import ObjectConstraints
from ..official.decoding_graph import build_decoding_graph
from ..official.knowledge_dfa import KnowledgePlanDFA

EOS = 200
TOKENS = {byte: bytes([byte]) for byte in [9, 10, 13, *range(32, 127)]}


def knowledge(**kwargs):
    kwargs.setdefault("entity_count", 4)
    kwargs.setdefault("max_operations", 4)
    kwargs.setdefault("causal", False)
    return KnowledgePlanDFA(TOKENS, EOS, **kwargs)


def native(**kwargs):
    kwargs.setdefault("entity_count", 4)
    kwargs.setdefault("max_operations", 4)
    return NativePlanDFA(TOKENS, EOS, **kwargs)


def op(name, entity=None, container=None):
    params = {}
    if entity is not None:
        params["target_entity_name"] = entity
    if container is not None:
        params["target_container_name"] = container
    return {"name": name, "params": params}


def labels(plan, indent=None):
    separators = None if indent else (",", ":")
    return list(json.dumps(plan, separators=separators, indent=indent).encode()) + [EOS]


SUBTASKS = ([[op("press", e)] for e in range(4)]
            + [[op("pick", e), op("place", container=c)] for e, c in product(range(4), repeat=2)]
            + [[op("pick", 0), op("pour", container=1)], [op("pick", 1), op("pour", container=2), op("place", container=3)]]
            + [[op("pick", 2), op("lift")]])


def plans(max_parts=2):
    for count in range(max_parts + 1):
        for parts in product(SUBTASKS, repeat=count):
            yield [operation for part in parts for operation in part]


def same_language(new, old, max_parts=2):
    for plan in plans(max_parts):
        assert new.accepts(labels(plan)) == old.accepts(labels(plan)), plan
    for plan in ([op("pick", 1), op("place", container=3)], [op("press", 2)]):
        for text in (labels(plan), labels(plan, indent=4)):
            left, right = new.start_state, old.start_state
            for label in text:
                assert new.allowed_tokens(left) == old.allowed_tokens(right)
                left, right = new.step(left, label), old.step(right, label)
                if left is None or right is None:
                    assert left is None and right is None
                    break


def test_press_answer_formulas_give_the_one_press_language():
    same_language(knowledge(press_answer=True), native(max_operations=1, patterns=(("press",),)))


def test_scene_bound_formula_gives_the_bounded_id_language():
    same_language(knowledge(entity_count=12, scene_bound=3), native(entity_count=3))


@pytest.mark.parametrize("chosen", [(("pick", "place"),), (("pick", "pour", "place"),),
                                    (("pick", "place"), ("press",)), (("pick", "pour"), ("pick", "pour", "place"))])
def test_chosen_pattern_formulas_give_the_restricted_grammar(chosen):
    same_language(knowledge(chosen_patterns=chosen), native(patterns=chosen))


def test_chosen_patterns_respect_the_operation_budget():
    # Two operations left: a three-operation pattern can no longer start.
    chosen = (("pick", "pour", "place"), ("press",))
    same_language(knowledge(chosen_patterns=chosen, max_operations=3), native(patterns=chosen, max_operations=3), max_parts=3)


def test_instruction_candidate_formula_gives_the_filtered_language():
    grounding = ObjectConstraints(restrictions=(("pick", "target_entity_name", (1,)),
                                                ("place", "target_container_name", (3,))))
    class ProceduralCandidates(KnowledgePlanDFA):
        # The former inventory filter, written out by hand, as an independent reference.
        def operation_admitted(self, action, count, extras):
            skill, pointer = self.actions[action]
            roles = self.skill_arguments[skill]
            ids = next((ids for s, r, ids in grounding.restrictions if roles and (s, r) == (skill, roles[0])), None)
            return ids is None or pointer in ids
    same_language(knowledge(object_candidates=grounding), ProceduralCandidates(TOKENS, EOS, entity_count=4, max_operations=4))


def test_without_episode_facts_the_language_is_the_grammar():
    same_language(knowledge(), native())


@pytest.mark.parametrize("rule,kwargs,plan", [
    ("press_answer_single_press", {"press_answer": True}, [op("pick", 0), op("place", container=1)]),
    ("press_answer_single_press", {"press_answer": True}, [op("press", 0), op("press", 1)]),
    ("scene_ids_bound_targets", {"scene_bound": 2}, [op("press", 3)]),
    ("chosen_patterns_complete", {"chosen_patterns": (("pick", "pour", "place"),)}, [op("pick", 0), op("pour", container=1)]),
    ("instruction_candidates_bound_targets",
     {"object_candidates": ObjectConstraints(restrictions=(("press", "target_entity_name", (1,)),))}, [op("press", 0)]),
])
@pytest.mark.parametrize("change", ["remove", "deactivate"])
def test_declarations_control_acceptance(rule, kwargs, plan, change):
    declarations = build_decoding_graph()
    strict = knowledge(decoding_graph=declarations, **kwargs)
    for key, constraint in list(declarations.graph.logicalConstrains.items()):
        if constraint.name == rule:
            if change == "remove":
                del declarations.graph.logicalConstrains[key]
            else:
                constraint.active = False
    relaxed = knowledge(decoding_graph=declarations, **kwargs)
    assert not strict.accepts(labels(plan))
    assert relaxed.accepts(labels(plan))


def test_causal_and_episode_formulas_combine():
    dfa = knowledge(causal=True, scene_bound=3)
    assert dfa.accepts(labels([op("pick", 1), op("place", container=2)]))
    assert not dfa.accepts(labels([op("pick", 1), op("place", container=1)]))  # causal
    assert not dfa.accepts(labels([op("pick", 1), op("place", container=3)]))  # scene bound


def test_effect_groups_follow_the_constrained_language():
    dfa = knowledge(scene_bound=2)
    state = dfa.start_state
    for label in b'[{"name":"pick","params":{"target_entity_name":':
        state = dfa.step(state, label)
    grouped = {label for labels in dfa.effect_groups(state).values() for label in labels}
    assert grouped == {ord("0"), ord("1")}  # the HMM lookahead only ranks ids the formulas admit
    state = dfa.step(dfa.step(state, ord("1")), ord("}"))  # the second brace closes the operation
    finished = {key[1] for key in dfa.effect_groups(state) if key[0] == "done"}
    assert {dfa.actions[action] for action in finished} == {("pick", 1)}


@pytest.mark.parametrize("change", ["remove", "deactivate"])
def test_continuation_formula_prunes_unnamed_patterns_as_they_start(change):
    # Completion alone fixes the accepted plans; continuation refuses an unnamed skill as soon as it is written.
    declarations = build_decoding_graph()
    strict = knowledge(decoding_graph=declarations, chosen_patterns=(("press",),))
    for key, constraint in list(declarations.graph.logicalConstrains.items()):
        if constraint.name == "chosen_patterns_continue":
            if change == "remove":
                del declarations.graph.logicalConstrains[key]
            else:
                constraint.active = False
    relaxed = knowledge(decoding_graph=declarations, chosen_patterns=(("press",),))
    prefix = list(b'[{"name":"pi')
    def run(dfa):
        state = dfa.start_state
        for label in prefix:
            state = dfa.step(state, label)
        return state
    assert run(strict) is None and run(relaxed) is not None


@pytest.mark.parametrize("kwargs", [
    {"causal": True},
    {"scene_bound": 2},
    {"press_answer": True},
    {"chosen_patterns": (("pick", "place"), ("press",))},
    {"object_candidates": ObjectConstraints(restrictions=(("pick", "target_entity_name", (1,)),))},
])
def test_the_hmm_action_lookahead_steers_the_knowledge_dfa(kwargs):
    from .test_action_lookahead import EOS as LOOKAHEAD_EOS, decode, random_hmm, table_for
    tokens = {byte: bytes([byte]) for byte in [9, 10, 13, *range(32, 127)]}
    dfa = KnowledgePlanDFA(tokens, LOOKAHEAD_EOS, entity_count=3, max_operations=6, **{"causal": False, **kwargs})
    hmm = random_hmm(8, 134, seed=5)
    table = table_for(dfa, hmm)
    for seed in range(3):
        output, processor = decode(dfa, table, hmm, 5.0, seed)
        assert dfa.accepts(output)
        assert processor.session.stats["steps"] > 0
