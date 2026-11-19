"""Semantic projection checks: byte translation must preserve compiled rules."""
import json
import pytest
from .native_baseline import compiled_decoder, prompt, parse_plan
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary


@pytest.fixture(scope='module')
def setup():
    vocabulary = PlanVocabulary.default(max_entities=4)
    public = {'entities':['entity_000','entity_001','entity_002','entity_003'],
        'movable':['entity_000','entity_001'], 'instruction':'put the red object in the bowl',
        'initial_relations':[], 'images':['raw.png','numbered.png']}
    dfa = compiled_decoder({i:bytes([i]) for i in range(32,127)},0,public,vocabulary,4)
    return vocabulary,public,dfa


def plan(source=0,destination=2):
    return [{'name':'pick','params':{'target_entity_name':source}},
            {'name':'place','params':{'target_container_name':destination}}]


@pytest.mark.parametrize('value',[plan(),plan(1,3),plan(2,3),plan(0,0),[],
    [{'name':'place','params':{'target_container_name':2}}],
    [{'name':'pick','params':{'target_entity_name':0}}],plan(4,2)])
def test_json_lexer_matches_compiled_policy(setup,value):
    vocabulary,_,dfa = setup
    tokens = []
    for operation in value:
        tokens.append('skill:'+operation['name'])
        for role,pointer in operation['params'].items():
            tokens.extend(['arg:'+role,'obj:'+str(pointer)])
    try:
        labels = [vocabulary.label_for_token(t) for t in tokens]+[vocabulary.eos_label]
        expected = dfa.language.policy.accepts(labels)
    except KeyError:
        expected = False
    assert dfa.accepts(list(json.dumps(value).encode())+[0]) == expected


def test_semantic_counterexamples_and_legal_wrong_objects(setup):
    _,_,dfa = setup
    assert dfa.accepts(list(json.dumps(plan()).encode())+[0])
    assert dfa.accepts(list(json.dumps(plan(1,3)).encode())+[0])  # legal wrong alternative
    assert not dfa.accepts(list(json.dumps(plan(2,3)).encode())+[0])  # static fixture
    assert not dfa.accepts(list(json.dumps(plan(0,0)).encode())+[0])  # self placement


def test_merged_native_tokens_cannot_bypass_compiled_object_rules(setup):
    vocabulary,public,_ = setup
    values = [plan(),plan(1,3),plan(2,3),plan(0,0)]
    token_bytes = {i:bytes([i]) for i in range(32,127)}
    token_bytes.update({1000+i:json.dumps(value).encode() for i,value in enumerate(values)})
    dfa = compiled_decoder(token_bytes,0,public,vocabulary,4)
    assert [dfa.accepts([1000+i,0]) for i in range(4)] == [True,True,False,False]


def test_prompt_reference_and_object_name_separation(setup):
    vocabulary,public,_ = setup
    original = prompt(public,vocabulary)
    changed = {**public,'entities':['SECRET_TUNA','SECRET_BOWL','X','Y'],
        'goals':['SECRET_GOAL'],'reference_plan':plan(1,3),'task':'SECRET_TASK'}
    assert prompt(changed,vocabulary) == original
    assert 'SECRET' not in original


def test_strict_json_parser_does_not_silently_repair_answers():
    assert parse_plan(json.dumps(plan())) == plan()
    assert parse_plan('```json\n'+json.dumps(plan())+'\n```') == plan()
    for text in ('Here is the plan: '+json.dumps(plan()),'{"name":"pick"}',
                 '[{"name":"pick","params":null}]'):
        with pytest.raises(ValueError):parse_plan(text)
