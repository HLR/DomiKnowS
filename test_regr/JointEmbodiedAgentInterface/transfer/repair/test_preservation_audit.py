import copy
import pytest

from test_regr.JointEmbodiedAgentInterface.transfer.repair.preservation_audit import compare_episode, first_difference, load_rows
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary, plan_to_tokens
from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
from test_regr.JointEmbodiedAgentInterface.generalization.shared_constraints import bind_shared


@pytest.fixture(scope='module')
def compiled():
    world = build_vlabench_world_graph('test_repair_preservation')
    vocab = PlanVocabulary.from_world(world, max_entities=4)
    entities = ['entity_000', 'entity_001', 'entity_002', 'entity_003']
    policy = bind_shared(vocab, 12, {'entity_table': entities, 'generation_static_fixture_ids': [2, 3]})

    def accepts(plan):
        state = policy.start_state
        for token in plan_to_tokens(plan, entities, world=world):
            label = vocab.label_for_token(token)
            if state is None or label not in policy.allowed_tokens(state):
                return False
            state = policy.step(state, label)
        return state is not None and policy.is_accepting(state)
    return accepts


def row(source=0, destination=2):
    return {'episode_id': 'episode', 'input_hash': 'same', 'generation_error': None,
        'prediction': [{'name': 'pick', 'params': {'target_entity_name': source}},
                       {'name': 'place', 'params': {'target_container_name': destination}}],
        'labels': [source, destination, 100]}


def test_legal_wrong_object_must_remain_unchanged(compiled):
    # No goal is supplied: selecting either movable object remains legal.
    for source in (0, 1):
        original = row(source)
        assert compare_episode(original, copy.deepcopy(original), compiled)['preservation_required']
    with pytest.raises(ValueError, match='changed a compiled-legal'):
        compare_episode(row(0), row(1), compiled)


def test_illegal_free_plan_can_change_but_new_plan_must_be_legal(compiled):
    assert not compare_episode(row(2), row(0), compiled)['preservation_required']
    with pytest.raises(ValueError, match='violates the compiled'):
        compare_episode(row(2), row(0, 0), compiled)


def test_projection_and_token_spelling_preservation(compiled):
    original = row()
    with pytest.raises(ValueError, match='projection rejected'):
        compare_episode(original, original, compiled, lambda _: False)
    changed = copy.deepcopy(original)
    changed['labels'] = [1001, 100]
    with pytest.raises(ValueError, match='changed a compiled-legal'):
        compare_episode(original, changed, compiled, lambda _: True)


def test_input_and_identity_guards(compiled):
    for key in ('episode_id', 'input_hash'):
        changed = copy.deepcopy(row()); changed[key] = 'different'
        with pytest.raises(ValueError, match='differs'):
            compare_episode(row(), changed, compiled)


def test_generation_errors_keep_denominators(compiled):
    original = row(); original['generation_error'] = {'message': 'budget exhausted'}
    original['prediction'] = []; original['labels'] = []
    result = compare_episode(original, row(), compiled)
    assert not result['preservation_required'] and result['free_generation_error']


def test_duplicate_episode_rejected(tmp_path):
    import json
    p = tmp_path/'predictions.jsonl'
    p.write_text('\n'.join([json.dumps(row())]*2))
    with pytest.raises(ValueError, match='duplicate'):
        load_rows(p, ['episode', 'other'])


def test_prefix_length_difference():
    assert first_difference([1, 2], [1, 2, 3]) == 2
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1, 2], [1, 2]) is None
