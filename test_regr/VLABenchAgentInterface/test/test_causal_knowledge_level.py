"""The ``causal`` knowledge level: three world-graph LCs, enforced by the graph DFA and the verifier alike."""
import pytest

from test_regr.generation.constraint_equivalence import assert_bounded_equivalence
from ..knowledge import LEVELS, group_active
from ..world_graph import build_vlabench_world_graph, set_knowledge_level
from .test_constraint_equivalence import compile_policy, encode, op, verified

ENTITIES = ("apple", "bowl", "cup")
CAUSAL = ("causal__anti_self_placement", "causal__distinct_subtask_destinations", "causal__destinations_stay_put")


def active(world):
    return {rule.name for rule in world.graph.logicalConstrains.values()
            if getattr(rule, "headLC", True) and rule.active}


def test_causal_is_a_branch_that_enables_only_its_own_group():
    assert LEVELS.index("grammar") < LEVELS.index("causal") < LEVELS.index("generic")
    assert [group for group in ("causal", "generic", "scene", "task") if group_active("causal", group)] == ["causal"]
    assert not any(group_active(level, "causal") for level in LEVELS if level != "causal")
    world = build_vlabench_world_graph("causal_level_groups", knowledge_level="causal")
    assert active(world) == set(CAUSAL)
    set_knowledge_level(world, "generic")
    assert not active(world) & set(CAUSAL) and "anti_self_placement" in active(world)


@pytest.mark.parametrize("plan,expected", [
    ([op("pick", 0), op("place", 1)], True),
    ([op("pick", 0), op("place", 0)], False),                                   # anti-self-placement
    ([op("pick", 0), op("pour", 1), op("place", 1)], False),                    # repeated destination
    ([op("pick", 0), op("place", 1), op("pick", 1), op("place", 2)], False),    # destination picked
    ([op("pick", 0), op("place", 1), op("pick", 2), op("place", 1)], True),     # a later subtask may reuse it
])
def test_graph_dfa_and_verifier_agree_on_the_causal_rules(plan, expected):
    world = build_vlabench_world_graph("causal_level_cases", knowledge_level="causal")
    vocab, grammar, monitor = compile_policy(world, ENTITIES, operations=4)
    sequence = encode(plan, vocab, world, ENTITIES)
    assert grammar.accepts(sequence), "the case must exercise the causal rules, not syntax"
    assert verified(plan, ENTITIES, world) == expected
    assert monitor.accepts(sequence) == expected


def test_repeated_picks_stay_legal_at_causal_but_not_at_generic():
    plan = [op("pick", 0), op("pour", 1), op("place", 2), op("pick", 0), op("pour", 1), op("place", 2)]
    for level, expected in (("causal", True), ("generic", False)):
        world = build_vlabench_world_graph(f"causal_repeat_{level}", knowledge_level=level)
        vocab, _, monitor = compile_policy(world, ENTITIES, operations=6)
        assert monitor.accepts(encode(plan, vocab, world, ENTITIES)) == expected
        assert verified(plan, ENTITIES, world) == expected


def test_all_small_paths_match_the_verifier_at_causal():
    world = build_vlabench_world_graph("causal_level_exhaustive", knowledge_level="causal")
    vocab, grammar, monitor = compile_policy(world, ("apple", "bowl"))
    from ..graph import tokens_to_plan

    def oracle(sequence):
        return verified(tokens_to_plan([vocab.token_for_label(label) for label in sequence], world=world),
                        ("apple", "bowl"), world)
    audit = assert_bounded_equivalence(grammar, monitor, oracle, max_steps=7, terminal_symbols={vocab.eos_label})
    assert audit.rejected > 0 and audit.accepted > 0


@pytest.mark.parametrize("name,plan", [
    ("causal__anti_self_placement", [op("pick", 0), op("place", 0)]),
    ("causal__distinct_subtask_destinations", [op("pick", 0), op("pour", 1), op("place", 1)]),
    ("causal__destinations_stay_put", [op("pick", 0), op("place", 1), op("pick", 1), op("place", 2)]),
])
def test_deactivating_a_causal_rule_changes_the_decoded_language(name, plan):
    world = build_vlabench_world_graph("causal_level_toggle", knowledge_level="causal")
    vocab, _, strict = compile_policy(world, ENTITIES, operations=4)
    next(rule for rule in world.graph.logicalConstrains.values() if rule.name == name).active = False
    _, _, relaxed = compile_policy(world, ENTITIES, operations=4)
    sequence = encode(plan, vocab, world, ENTITIES)
    assert not strict.accepts(sequence) and relaxed.accepts(sequence)


@pytest.mark.parametrize("flags,expected", [
    (["--knowledge", "causal"], "causal"),
    (["--knowledge", "grammar"], "grammar"),
    (["--constraints", "causal"], "causal"),
    (["--knowledge", "causal", "--constraints", "causal"], "causal"),
    ([], "none"),
])
def test_official_protocol_takes_the_level_from_knowledge(flags, expected):
    from .. import main
    args = main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "d", "--protocol", "official", *flags])
    assert main.official_constraints(args) == expected


@pytest.mark.parametrize("flags", [["--knowledge", "generic"], ["--knowledge", "causal", "--constraints", "grammar"]])
def test_official_protocol_rejects_levels_it_cannot_ground(flags):
    from .. import main
    args = main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "d", "--protocol", "official", *flags])
    with pytest.raises(ValueError):
        main.official_constraints(args)
