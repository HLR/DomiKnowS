import pytest
from .bind_images import match_geometry, declared_libero_entities
from ..generalization import libero_protocol


def test_geometry_bijection_is_observational_and_not_inventory_order():
    public = {'entities': ['entity_000', 'entity_001', 'entity_002'], 'movable': ['entity_000'],
              'positions': {'entity_000': [10, 0, 0], 'entity_001': [20, 0, 0], 'entity_002': [30, 0, 0]}}
    assert match_geometry(public, [[30, 0, 0], [10, 0, 0], [20, 0, 0]], [False, True, False]) == [1, 2, 0]
    with pytest.raises(ValueError, match='ambiguous'):
        match_geometry(public, [[10, 0, 0]]*3, [True]*3)


def test_entity_bindings_do_not_change_with_goals_language_or_objects_of_interest(tmp_path):
    path = tmp_path/'scene.bddl'
    prefix = '(define (:objects apple banana - fruit) (:fixtures bowl - bowl) '
    path.write_text(prefix+'(:goal (in apple bowl)) (:obj_of_interest apple))')
    first = declared_libero_entities(path, libero_protocol)
    path.write_text(prefix+'(:goal (in banana bowl)) (:obj_of_interest banana) (:language changed instruction))')
    assert first == declared_libero_entities(path, libero_protocol) == ['apple', 'banana', 'bowl']
