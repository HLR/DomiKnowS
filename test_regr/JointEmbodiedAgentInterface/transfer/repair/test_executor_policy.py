"""Counterexamples for capabilities and grounded rules across decoder surfaces."""
import json

import pytest

from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary

from ...generalization.executor_policy import (
    Action, CAPABILITIES, DESTINATION, EXECUTOR, ExecutorProfile, SUBJECT,
    bind_plan_policy, pick_place_profile,
)
from .surface import surface_decoder


@pytest.fixture(scope='module')
def setup():
    vocabulary = PlanVocabulary.default(max_entities=4)
    public = {'entities': ['a', 'b', 'c', 'd'], 'movable': ['a', 'b']}
    context = {'entity_table': public['entities'], 'generation_static_fixture_ids': [2, 3]}
    return vocabulary, public, context


def plan_labels(vocabulary, plan):
    tokens = []
    for operation in plan:
        tokens.append('skill:' + operation['name'])
        for role, pointer in operation['params'].items():
            tokens.extend(('arg:' + role, f'obj:{pointer}'))
    return [vocabulary.label_for_token(token) for token in (*tokens, vocabulary.eos_token)]


def operation(skill, role, pointer):
    return {'name': skill, 'params': {role: pointer}}


@pytest.mark.parametrize(('plan', 'capable', 'executable'), [
    ([operation('pick', SUBJECT, 0), operation('pour', DESTINATION, 2)], False, False),
    ([operation('place', DESTINATION, 2)], True, False),
    ([operation('pick', SUBJECT, 0), operation('pick', SUBJECT, 1)], True, False),
    ([operation('pick', SUBJECT, 0), operation('place', DESTINATION, 2)], True, True),
    ([operation('pick', SUBJECT, 1), operation('place', DESTINATION, 2)], True, True),
    ([operation('pick', SUBJECT, 0), operation('place', DESTINATION, 0)], False, False),
    ([operation('pick', SUBJECT, 2), operation('place', DESTINATION, 3)], False, False),
    ([operation('pick', SUBJECT, 0), operation('place', DESTINATION, 2),
      operation('pick', SUBJECT, 0), operation('place', DESTINATION, 3)], True, True),
])
def test_joint_and_native_accept_the_same_canonical_plans(setup, plan, capable, executable):
    vocabulary, public, context = setup
    profile = pick_place_profile()
    for policy, expected in ((CAPABILITIES, capable), (EXECUTOR, executable)):
        dfa = bind_plan_policy(policy, vocabulary, 4, context, profile)
        native = surface_decoder({1000: json.dumps(plan).encode()}, 0, public, vocabulary,
                                 4, policy_name=policy, executor_profile=profile)
        assert dfa.accepts(plan_labels(vocabulary, plan)) == expected
        assert native.accepts([1000, 0]) == expected


def test_known_false_affordance_and_compatibility_restrict_but_unknown_does_not(setup):
    vocabulary, _, context = setup
    context = {**context, 'executor_properties': [(0, 'graspable', False)],
               'executor_compatibility': [{'subject': 1, 'destination': 2, 'allowed': False}]}
    policy = bind_plan_policy(EXECUTOR, vocabulary, 4, context, pick_place_profile())
    for subject, destination, expected in ((0, 3, False), (1, 2, False), (1, 3, True)):
        labels = plan_labels(vocabulary, [operation('pick', SUBJECT, subject),
                                          operation('place', DESTINATION, destination)])
        assert policy.accepts(labels) == expected


def test_executor_declarations_change_compiled_language_and_bind_each_hand(setup):
    vocabulary, _, context = setup
    profile = ExecutorProfile((
        Action('pick', (SUBJECT,), hand='left', grasp=True),
        Action('pour', (DESTINATION,), hand='right', requires_held=True),
    ), capacities=(('left', 1), ('right', 1)), initial_holdings=(('left', None), ('right', 1)))
    labels = plan_labels(vocabulary, [operation('pick', SUBJECT, 0), operation('pour', DESTINATION, 2)])
    assert not bind_plan_policy(EXECUTOR, vocabulary, 4, context, pick_place_profile()).accepts(labels)
    assert bind_plan_policy(EXECUTOR, vocabulary, 4, context, profile).accepts(labels)
    empty_other_hand = ExecutorProfile(profile.actions, profile.capacities, (('left', None), ('right', None)))
    assert not bind_plan_policy(EXECUTOR, vocabulary, 4, context, empty_other_hand).accepts(labels)


def test_missing_executor_profile_uses_existing_shared_language(setup):
    vocabulary, _, context = setup
    reference = bind_plan_policy('shared-verified', vocabulary, 4, context)
    for name in (CAPABILITIES, EXECUTOR):
        policy = bind_plan_policy(name, vocabulary, 4, {**context, 'task': 'unused'})
        assert policy.base_dfa is reference.base_dfa
        assert policy.start_state == reference.start_state


def test_rlbench_observation_adapter_binds_without_task_identity(setup):
    from ...generalization.rlbench_protocol import planning_context

    vocabulary, _, _ = setup
    ids = [f'entity_{i:03d}' for i in range(4)]
    context = planning_context({'entities': ids, 'movable': ids[:2],
                                'positions': {name: [0, 0, 0] for name in ids},
                                'instruction': 'Pick an object and put it down.', 'images': []})
    dfa = bind_plan_policy(EXECUTOR, vocabulary, 4, context, pick_place_profile())
    for pointer, expected in ((0, True), (1, True), (2, False)):
        plan = [operation('pick', SUBJECT, pointer), operation('place', DESTINATION, 3)]
        assert dfa.accepts(plan_labels(vocabulary, plan)) == expected
    assert not dfa.accepts(plan_labels(vocabulary, [operation('place', DESTINATION, 3)]))
