import pytest
from .permutation import EntityPermutation, deterministic_permutations


def test_three_cycle_maps_world_bindings_gold_and_predictions_in_opposite_directions():
    perm = EntityPermutation((2, 0, 1))
    original = {'instruction': 'put 2 objects: entity_000 into entity_001',
        'entities': ['entity_000', 'entity_001', 'entity_002'],
        'movable': ['entity_000'], 'visible_entity_ids': [0, 2],
        'positions': {'entity_000': [10, 0, 0], 'entity_001': [20, 0, 0], 'entity_002': [30, 0, 0]},
        'entity_types': {'entity_000': 'fruit', 'entity_001': 'bowl', 'entity_002': 'table'},
        'initial_relations': [['on', 'entity_000', 'entity_002']]}
    changed = perm.public(original)
    assert changed['positions']['entity_000'] == [30, 0, 0]
    assert changed['entity_types']['entity_001'] == 'fruit'
    assert changed['movable'] == ['entity_001'] and changed['visible_entity_ids'] == [0, 1]
    assert changed['initial_relations'] == [['on', 'entity_001', 'entity_000']]
    assert changed['instruction'] == 'put 2 objects: entity_001 into entity_002'
    plan = [{'name': 'pick', 'params': {'target_entity_name': 0}},
            {'name': 'place', 'params': {'target_container_name': 1}}]
    new = perm.plan(plan)
    assert new[0]['params']['target_entity_name'] == 1
    assert new[1]['params']['target_container_name'] == 2
    assert perm.plan(new, inverse=True) == plan
    assert original['movable'] == ['entity_000']


def test_permutation_does_not_change_numeric_quantities_or_repair_invalid_predictions():
    perm = EntityPermutation((2, 0, 1))
    plan = [{'name': 'unsupported', 'params': {'count': 2, 'angle': 1.0, 'target_entity_name': 0}}]
    transformed = perm.plan(plan)
    assert transformed[0]['params'] == {'count': 2, 'angle': 1.0, 'target_entity_name': 1}
    bad = [{'name': 'pick', 'params': {'target_entity_name': 99}}]
    assert perm.plan(bad, inverse=True, strict=False) == bad
    with pytest.raises(ValueError): perm.plan(bad)
    with pytest.raises(ValueError): perm.encode_pointer(True)


def test_five_orders_are_reproducible_nonidentity_and_unique_when_possible():
    orders = deterministic_permutations('episode', 10)
    assert orders == deterministic_permutations('episode', 10)
    assert len({p.order for p in orders}) == 5
    assert all(p.order != tuple(range(10)) for p in orders)
    assert len({p.order for p in deterministic_permutations('episode', 3)}) == 5
    assert len({p.order for p in deterministic_permutations('episode', 2)}) == 1


@pytest.mark.parametrize('order', [(0, 0), (0, 2), (), (False, 1)])
def test_nonbijections_and_boolean_ids_rejected(order):
    with pytest.raises(ValueError, match='bijection'): EntityPermutation(order)
