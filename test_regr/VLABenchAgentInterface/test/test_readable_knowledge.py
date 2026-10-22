"""Actual formulas, primitive evidence, and the documented tier contract."""
from dataclasses import replace
from itertools import product

import pytest
import torch

from domiknows.generation.grounded import Operation, PlanningFacts, requirement_events
from ..constraint_compiler import compile_world_constraints
from ..knowledge import LEVELS, group_active
from ..semantics import PlanSemantics
from ..training import build_constraint_runtime
from ..world_graph import build_vlabench_world_graph, materialize_plan, set_knowledge_level, verify_plan_constraints


def active_names(world):
    return {rule.name for rule in world.graph.logicalConstrains.values() if rule.headLC and rule.active}


def descendants(root):
    pending = list(root.getChildDataNodes() or ())
    while pending:
        node = pending.pop()
        yield node
        pending.extend(node.getChildDataNodes() or ())


@pytest.mark.parametrize('level', LEVELS)
def test_tier_toggle_matches_fresh_graph_including_grounded_families(level):
    toggled = build_vlabench_world_graph('readable_toggle_' + level, knowledge_level='grammar')
    assert len(toggled.grounded_constraints.constraints) == 8
    assert not any(rule.active for rule in toggled.grounded_constraints.constraints)
    set_knowledge_level(toggled, level)
    fresh = build_vlabench_world_graph('readable_fresh_' + level, knowledge_level=level)
    assert active_names(toggled) == active_names(fresh)
    assert {id(rule) for rules in toggled.knowledge_groups.values() for rule in rules} == {
        id(rule) for rule in toggled.graph.logicalConstrains.values() if rule.headLC
    }
    for group, rules in toggled.knowledge_groups.items():
        assert all(rule.active == group_active(level, group) for rule in rules)


def test_aggregate_summary_cannot_hide_a_false_requirement():
    world = build_vlabench_world_graph('readable_evidence')
    plan = [{'name': 'pick', 'params': {'target_entity_name': 0}}, {'name': 'lift', 'params': {}}]
    root = materialize_plan(plan, ('apple',), world,
                            task_context={'planning_facts': {'entity_properties': {'apple': {'graspable': False}}}})
    assert verify_plan_constraints(root, world).score < 1.0
    nodes = list(descendants(root))
    # Forge the old aggregate observation. The default LC reads the separate
    # evidence node and must still reject the known false graspability fact.
    for node in nodes:
        if node.ontologyNode is world.operation:
            key = '<' + world.grounded_constraints['affordance__satisfied'].name + '>'
            node.attributes[key] = torch.tensor([0., 1.])
    assert verify_plan_constraints(root, world).score < 1.0
    evidence = [node for node in nodes if node.ontologyNode is world.grounded_constraints.requirement]
    assert any('graspable' in str(node.instanceValue) for node in evidence)


def test_legacy_composite_flags_cannot_bypass_default_world_logic(monkeypatch):
    original = PlanSemantics.evaluate_operation
    def forged(self, *args):
        return replace(original(self, *args), is_self_placement=False, targets_static_fixture=False,
                       repeated_pick_entity=False, unopened_container_placement=False)
    monkeypatch.setattr(PlanSemantics, 'evaluate_operation', forged)
    world = build_vlabench_world_graph('readable_primitive')
    plan = [{'name': 'pick', 'params': {'target_entity_name': 0}},
            {'name': 'place', 'params': {'target_container_name': 0}}]
    assert verify_plan_constraints(materialize_plan(plan, ('apple',), world), world).score < 1.0
    runtime = build_constraint_runtime(world, max_entities=1, max_operations=2, name_prefix='readable_primitive')
    from ..plan_codec import plan_to_tokens
    policy = runtime.dfa_for('vlabench', {'entities': ('apple',)})
    labels = [runtime.vocabulary.label_for_token(token) for token in plan_to_tokens(plan, world=world)]
    assert not policy.accepts(labels)


def test_hand_formulas_match_independent_truth_tables():
    world = build_vlabench_world_graph('readable_hand_truth_tables')
    program = compile_world_constraints(world)
    for family in ('holding', 'capacity'):
        for hand_known, held, subject, idempotent, capacity in product(
            (False, True), (None, 'apple', 'cup'), (None, 'apple'), (False, True), (None, 0, 1)
        ):
            raw = {'initial_state': {'held': {'right': held}}} if hand_known else {}
            if capacity is not None:
                raw['capacities'] = {'right': capacity}
            planning = PlanningFacts.parse(raw, entities=('apple', 'cup'))
            op = Operation('lift' if family == 'holding' else 'pick', subject,
                           requires_held=family == 'holding', grasp=family == 'capacity', idempotent=idempotent)
            _, checks = planning.evaluate(planning.initial, op)
            check = next(check for check in checks if check.family == family)
            observed_subject = subject if subject is not None else held if hand_known else None
            if family == 'holding':
                known = idempotent or hand_known
                expected = not known or idempotent or (held is not None and (subject is None or held == subject))
            else:
                known = capacity is not None and (capacity == 0 or hand_known)
                expected = not known or (capacity > 0 and (held is None or held == observed_subject))
            assert program.check_events(tuple(requirement_events(world.grounded_constraints, (check,)))) == expected


def test_unknown_requirement_is_not_materialized_as_true():
    world = build_vlabench_world_graph('readable_unknown')
    plan = [{'name': 'pick', 'params': {'target_entity_name': 0}}, {'name': 'lift', 'params': {}}]
    root = materialize_plan(plan, ('apple',), world)
    evidence = [node for node in descendants(root) if node.ontologyNode is world.grounded_constraints.requirement]
    unknown = [node for node in evidence if int(node.attributes['<' + world.grounded_constraints.known.name + '>'].argmax()) == 0]
    assert unknown
    assert all(int(node.attributes['<' + world.grounded_constraints.holds.name + '>'].argmax()) == 0 for node in unknown)
    assert not root.attributes['grounding_report']['fully_grounded']


def test_low_tier_graph_can_enable_grounded_enforcement_later():
    world = build_vlabench_world_graph('readable_enable_task', knowledge_level='grammar')
    set_knowledge_level(world, 'task')
    runtime = build_constraint_runtime(world, max_entities=1, max_operations=2, name_prefix='readable_enable_task')
    policy = runtime.dfa_for('vlabench', {'entities': ('apple',),
                          'planning_facts': {'entity_properties': {'apple': {'graspable': False}}}})
    tokens = ('skill:pick', 'arg:target_entity_name', 'obj:0', 'skill:lift', '<eos>')
    assert not policy.accepts([runtime.vocabulary.label_for_token(token) for token in tokens])
