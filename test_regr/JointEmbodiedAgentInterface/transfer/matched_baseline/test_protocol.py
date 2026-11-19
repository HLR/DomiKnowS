"""Semantic tests for matched inputs and exact, observation-only ID conversion."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from .protocol import prompt_core,normalize_ids,raw_format_valid,common_prompt,role_prompt


@pytest.fixture
def public():
    return {'instruction':'put the red block in the basket','entities':['entity_000','entity_001','entity_002'],
        'movable':['entity_000','entity_001'],'initial_relations':[['on','entity_000','entity_002']],
        'images':['raw.png','tags.png']}


@pytest.fixture
def vocabulary():
    args=(('pick',('target_entity_name',)),('place',('target_container_name',)))
    return SimpleNamespace(skill_arguments=args,skill_argument_map=dict(args))


def test_common_prompt_excludes_private_fields_and_string_id_aliases(public,vocabulary):
    original=prompt_core(public,vocabulary)
    poisoned={**public,'reference_plan':'SECRET','goals':'SECRET','task':'SECRET','native_success':True}
    assert prompt_core(poisoned,vocabulary)==original
    assert 'SECRET' not in original and 'entity_000' not in original
    assert '["on", 0, 2]' in original


def test_id_conversion_preserves_wrong_choices_and_raw_output(public):
    plan=[{'name':'pick','params':{'target_entity_name':'entity_001'}},
          {'name':'place','params':{'target_container_name':'2'}}]
    original=deepcopy(plan);converted,changes=normalize_ids(plan,public['entities'])
    assert plan==original
    assert converted[0]['params']['target_entity_name']==1  # never choose the red object from the instruction
    assert converted[1]['params']['target_container_name']==2
    assert len(changes)==2
    assert normalize_ids(converted,public['entities'])==(converted,[])


@pytest.mark.parametrize('value',['red block','entity_999','-1','1.0','001',True,17])
def test_unknown_or_ambiguous_ids_are_not_repaired(public,value):
    plan=[{'name':'pick','params':{'target_entity_name':value}}]
    assert normalize_ids(plan,public['entities'])==(plan,[])


def test_raw_format_does_not_decide_semantic_legality(public,vocabulary):
    raw=[{'name':'pick','params':{'target_entity_name':'entity_002'}}]
    assert not raw_format_valid(raw,vocabulary,3)
    converted,_=normalize_ids(raw,public['entities'])
    assert raw_format_valid(converted,vocabulary,3)  # static fixture: format-valid but semantically illegal


def test_shared_prompt_hook_restores_and_keeps_unmarked_query_path(vocabulary,public):
    class Processor:
        def apply_chat_template(self,messages,**kwargs):return str(messages)
    planner=SimpleNamespace(_prompt_base=lambda d,c:'original',_chat=lambda p,n:'old chat',processor=Processor())
    original=planner._prompt_base;core=prompt_core(public,vocabulary)
    with common_prompt(planner):
        assert planner._prompt_base('vlabench',{'instruction':'instruction-only query'})=='original'
        bound={'_matched_core':core,'_matched_roles':{'source':1,'destination':2}}
        assert planner._prompt_base('vlabench',bound)==role_prompt(core,bound['_matched_roles'])
    assert planner._prompt_base is original


def test_duplicate_ids_and_unbound_relations_stop(public,vocabulary):
    with pytest.raises(ValueError):prompt_core({**public,'entities':['entity_000']*3},vocabulary)
    with pytest.raises(ValueError):prompt_core({**public,'initial_relations':[['on','hidden_object','entity_002']]},vocabulary)
