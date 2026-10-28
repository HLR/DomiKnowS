"""The complex-task knowledge as logical constraints rejects exactly what the former templates rejected."""
from dataclasses import replace

import pytest

from domiknows.generation.grounded import FAMILIES, PlanningFacts
from ..complex_grounding import complex_planning_facts
from ..knowledge.complex_lcs import ComplexMonitor, build_complex_knowledge
from ..semantics import PlanSemantics
from ..world_graph import SKILL_ARGUMENTS, set_knowledge_level
from .legacy_complex_oracle import complex_planning_facts as legacy_facts
from .test_complex_improvements import runtime  # noqa: F401 (fixture)

SCENARIOS = {
    "book": ("book_rearrange", "Arrange the books from earliest to latest publication",
             ("century17_first_book", "century19_second_book", "shelf", "table")),
    "book_reverse": ("book_rearrange", "Arrange the books from latest to earliest",
                     ("century17_first_book", "century19_second_book", "shelf")),
    "cook": ("cook_dishes", "Prepare tomato_and_egg_stir_fry", ("plate", "tomato", "egg", "fish")),
    "chemistry": ("take_chemistry_experiment", "Mix Na2CO3 solution and HCl solution to produce NaCl.",
                  ("Na2CO3", "HCl", "NaCl", "flask", "table")),
    "hammer": ("hammer_nail_and_hang_picture", "Hang the The Night on the wall",
               ("hammer", "nail", "counter", "the night_painting")),
    "unbound": ("cook_dishes", "Prepare an unknown dish", ("plate", "tomato")),
}
SKILLS = {"book_rearrange": ("pick", "push", "pull", "place"), "cook_dishes": ("pick", "place"),
          "take_chemistry_experiment": ("pick", "pour", "place"),
          "hammer_nail_and_hang_picture": ("pick", "push", "place")}


@pytest.fixture(scope="module")
def knowledge():
    return build_complex_knowledge()


def plannings(knowledge, task, instruction, entities):
    old = PlanningFacts.parse(legacy_facts(task, instruction, entities), entities=entities)
    raw = complex_planning_facts(task, instruction, entities)
    new = PlanningFacts.parse(raw, entities=entities)
    if "complex_request" in raw:
        new = replace(new, monitors=(ComplexMonitor(knowledge, task, raw["complex_request"]),))
    return old, new


def alphabet(task, entities):
    for skill in SKILLS[task]:
        roles = SKILL_ARGUMENTS[skill]
        if not roles:
            yield skill, {}
        for entity in entities:
            for role in roles:
                yield skill, {role: entity}


def violated(checks):
    return {check.family for check in checks if check.family in FAMILIES and check.satisfied is False}


def walk(knowledge, task, instruction, entities, depth, limit=6000):
    """Depth-first over plans the former model admits; returns the number of compared prefixes."""
    old_planning, new_planning = plannings(knowledge, task, instruction, entities)
    old, new = PlanSemantics(entities, SKILL_ARGUMENTS, old_planning), PlanSemantics(entities, SKILL_ARGUMENTS, new_planning)
    steps = tuple(alphabet(task, entities))
    compared = 0
    stack = [((), old.initial_state(), new.initial_state())]
    while stack and compared < limit:
        plan, old_state, new_state = stack.pop()
        old_end = old_planning.completion(old_state.grounded).satisfied is False
        new_end = new_planning.completion(new_state.grounded).satisfied is False
        assert old_end == new_end, (plan, "completion")
        compared += 1
        if len(plan) == depth:
            continue
        for skill, params in steps:
            _, old_checks = old.grounded_transition(old_state, skill, params)
            _, new_checks = new.grounded_transition(new_state, skill, params)
            assert violated(old_checks) == violated(new_checks), (plan, skill, params)
            if not violated(old_checks):
                stack.append((plan + ((skill, tuple(params.values())),),
                              old.advance(old_state, skill, params), new.advance(new_state, skill, params)))
    return compared


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_compiled_complex_rules_reject_what_the_templates_rejected(knowledge, name):
    task, instruction, entities = SCENARIOS[name]
    depth = {"book": 6, "book_reverse": 6, "hammer": 5}.get(name, 4)
    assert walk(knowledge, task, instruction, entities, depth) > 50


def test_rules_are_generated_from_the_tables(knowledge):
    names = set(knowledge.enforce_constraints)
    assert len([n for n in names if n.startswith("complex__cook__")]) == 15 * 4
    assert {"complex__book__order", "complex__chemistry__return_after_pour",
            "complex__hammer__step5_after_step4", "complex__hammer__steps_done"} <= names


@pytest.mark.parametrize("rule,name,prefix,step", [
    ("complex__book__order", "book", (), ("pick", {"target_entity_name": "century19_second_book"})),
    ("complex__hammer__step4_after_step3", "hammer", (), ("pick", {"target_entity_name": "the night_painting"})),
    ("complex__chemistry__return_after_pour", "chemistry",
     (("pick", {"target_entity_name": "HCl"}),), ("place", {"target_container_name": "table"})),
])
def test_deactivating_a_complex_rule_removes_its_order_violation(rule, name, prefix, step):
    task, instruction, entities = SCENARIOS[name]
    knowledge = build_complex_knowledge()

    def order_violated():
        _, planning = plannings(knowledge, task, instruction, entities)
        semantics = PlanSemantics(entities, SKILL_ARGUMENTS, planning)
        state = semantics.initial_state()
        for skill, params in prefix:
            state = semantics.advance(state, skill, params)
        return "milestone_order" in violated(semantics.grounded_transition(state, *step)[1])
    assert order_violated()
    knowledge.set_active(False, names={rule})
    assert not order_violated()


def test_knowledge_level_switches_the_complex_rules(runtime):
    world = runtime.world_bundle
    previous = world.knowledge_state.level
    try:
        set_knowledge_level(world, "scene")
        assert not any(lc.active for lc in world.complex_knowledge.get().enforce_constraints.values())
        set_knowledge_level(world, "task")
        assert all(lc.active for lc in world.complex_knowledge.get().enforce_constraints.values())
    finally:
        set_knowledge_level(world, previous)
