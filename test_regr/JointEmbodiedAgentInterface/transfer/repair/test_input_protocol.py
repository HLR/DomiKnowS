from copy import deepcopy
import pytest

from .input_protocol import (IMAGE_INSTRUCTION, METADATA_ASSISTED, generation_context,
    constraint_context, constraint_public, contract_record, validate_resume_contract)
from ..matched_baseline.protocol import common_prompt
from ...generalization.shared_constraints import bind_shared
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary


@pytest.fixture(scope='module')
def vocabulary():
    return PlanVocabulary.default(max_entities=4)


@pytest.fixture
def public():
    return {'instruction': 'Put the red block into the basket.',
        'entities': ['entity_000', 'entity_001', 'entity_002', 'entity_003'],
        'images': ['clean.png', 'tags.png'], 'movable': ['entity_000', 'entity_001'],
        'initial_relations': [['On', 'entity_000', 'entity_003']],
        'entity_types': {'entity_000': 'block'}, 'positions': {'entity_000': [1, 2, 3]}}


@pytest.mark.parametrize('key,value', [
    ('movable', []), ('initial_relations', [['CANARY', 'entity_002', 'entity_001']]),
    ('entity_types', {'entity_001': 'CANARY'}), ('positions', {'CANARY': [9, 8, 7]}),
    ('visible_entity_ids', [3]), ('task_id', 'CANARY'), ('reference_plan', ['CANARY']),
    ('goals', ['CANARY']), ('obj_of_interest', ['CANARY']), ('target_entity_name', 'CANARY'),
    ('predicted_roles', {'source': 3, 'destination': 1}), ('button_hints', 'CANARY'),
    ('knowledge', 'CANARY')])
def test_metadata_and_reference_changes_cannot_change_model_inputs(public, vocabulary, key, value):
    original = generation_context(public, vocabulary)
    changed = deepcopy(public); changed[key] = value
    assert generation_context(changed, vocabulary) == original
    assert 'CANARY' not in original['_matched_core']


def test_no_semantic_names_inventory_relations_or_explanations(public, vocabulary):
    public['entities'] = ['PRIVATE_red_block', 'PRIVATE_blue_block', 'PRIVATE_basket', 'PRIVATE_table']
    bound = generation_context(public, vocabulary)
    prompt = bound['_matched_core']
    assert bound['entity_table'] == [f'entity_{i:03d}' for i in range(4)]
    for forbidden in ('PRIVATE_', 'Entities:', 'Initial scene relations', 'fixture',
                      'acquires', 'releases', 'source ID', 'destination ID', 'entity_000'):
        assert forbidden not in prompt
    assert public['instruction'] in prompt
    assert set(bound) == {'instruction', 'images', 'entity_table', '_matched_core', '_matched_roles'}


def test_model_path_does_not_access_constraint_fields(public, vocabulary):
    class Restricted(dict):
        def __getitem__(self, key):
            assert key in ('instruction', 'images', 'entities')
            return super().__getitem__(key)
        def get(self, key, *args):
            raise AssertionError('metadata access: '+key)
    generation_context(Restricted(public), vocabulary)


def test_constraint_evidence_changes_only_compiled_policy(public, vocabulary):
    original = generation_context(public, vocabulary)
    unknown = deepcopy(public); unknown.pop('movable')
    assert generation_context(unknown, vocabulary) == original
    assert constraint_context(public)['generation_static_fixture_ids'] == [2, 3]
    assert constraint_context(unknown)['generation_static_fixture_ids'] == []
    labels = [vocabulary.label_for_token(t) for t in ('skill:pick', 'arg:target_entity_name', 'obj:2', '<eos>')]
    def accepts(observation):
        policy = bind_shared(vocabulary, 12, constraint_context(observation)); state = policy.start_state
        for label in labels:
            if state is None or label not in policy.allowed_tokens(state):
                return False
            state = policy.step(state, label)
        return state is not None and policy.is_accepting(state)
    assert not accepts(public)
    assert accepts(unknown)


def test_constraint_view_drops_goals_roles_and_task_identity(public):
    public.update(goals=['CANARY'], task_id='CANARY', predicted_roles={'source': 2})
    assert set(constraint_public(public)) == {'entities', 'movable'}
    assert set(constraint_context(public)) == {'entity_table', 'generation_static_fixture_ids'}


def test_actual_prompt_override_uses_strict_core(public, vocabulary):
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return str(messages)
    class Planner:
        processor = Processor()
        def _prompt_base(self, *args): return 'PRIVATE_TEMPLATE'
        def _chat(self, *args): return 'PRIVATE_CHAT'
    planner = Planner(); bound = generation_context(public, vocabulary)
    with common_prompt(planner):
        rendered = planner._chat(planner._prompt_base('vlabench', bound), len(bound['images']))
        assert public['instruction'] in rendered
        assert 'PRIVATE_' not in rendered and 'fixture' not in rendered
    assert planner._prompt_base() == 'PRIVATE_TEMPLATE'


def test_versioned_resume_rejects_legacy_outputs(public, vocabulary):
    with pytest.raises(ValueError, match='input protocol differs'):
        validate_resume_contract({'policy': 'none'}, IMAGE_INSTRUCTION)
    validate_resume_contract({'input_protocol': IMAGE_INSTRUCTION}, IMAGE_INSTRUCTION)
    old = generation_context(public, vocabulary, METADATA_ASSISTED)
    assert 'Initial scene relations' in old['_matched_core']
    assert not contract_record(IMAGE_INSTRUCTION)['model_scene_metadata']
