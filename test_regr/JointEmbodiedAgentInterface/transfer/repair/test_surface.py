import json
import pytest
from ..matched_baseline.protocol import normalize_ids
from ..native_baseline import parse_plan
from .surface import surface_decoder
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
from ...generalization.shared_constraints import bind_shared


@pytest.fixture(scope='module')
def setup():
    v=PlanVocabulary.default(max_entities=4)
    public={'entities':['entity_000','entity_001','entity_002','entity_003'],'movable':['entity_000','entity_001']}
    return v,public


def plan(a,b):return [{'name':'pick','params':{'target_entity_name':a}},{'name':'place','params':{'target_container_name':b}}]


@pytest.mark.parametrize('a',[0,'0','entity_000','numbered object 0'],ids=['integer','decimal-string','entity-alias','prompt-alias'])
@pytest.mark.parametrize('b',[2,'2','entity_002','numbered object 2'],ids=['integer','decimal-string','entity-alias','prompt-alias'])
@pytest.mark.parametrize('wrapper',['{}','```json\n{}\n```','```\n{}\n```'])
def test_exact_aliases_and_wrappers_preserve_pointers(setup,a,b,wrapper):
    v,public=setup;raw=wrapper.format(json.dumps(plan(a,b)))
    tokens={i:bytes([i]) for i in range(9,127)};tokens[1000]=raw.encode()
    d=surface_decoder(tokens,0,public,v,4)
    assert d.accepts([1000,0])
    assert normalize_ids(parse_plan(raw),public['entities'])[0]==plan(0,2)


@pytest.mark.parametrize('value',[plan('entity_002','3'),plan('0','entity_000'),plan('entity_999',2),plan('red block',2)])
def test_aliases_do_not_bypass_compiled_constraints(setup,value):
    v,public=setup;raw=json.dumps(value)
    d=surface_decoder({1000:raw.encode()},0,public,v,4)
    assert not d.accepts([1000,0])


def test_legal_wrong_object_and_malformed_wrapper(setup):
    v,p=setup
    for text,expected in [(json.dumps(plan('1','3')),True),('```json\n'+json.dumps(plan(0,2))+'```',False),
                          ('```json\n'+json.dumps(plan(0,2))+'\n``` extra',False)]:
        d=surface_decoder({1000:text.encode()},0,p,v,4)
        assert d.accepts([1000,0])==expected


def test_escaped_alias_and_parser_recognized_unicode_header(setup):
    v,p=setup
    text='```JSON ☃\n'+json.dumps(plan('numbered object 0','2')).replace('object 0','object\\u00200')+'\n```'
    assert normalize_ids(parse_plan(text),p['entities'])[0]==plan(0,2)
    assert surface_decoder({1000:text.encode()},0,p,v,4).accepts([1000,0])


def test_alias_collision_uses_normalizer_precedence(setup):
    v,_=setup
    public={'entities':['entity_000','numbered object 0','entity_002','entity_003'],
            'movable':['entity_000','numbered object 0']}
    raw=plan('numbered object 0',0)
    assert normalize_ids(raw,public['entities'])[0]==plan(0,0)
    assert not surface_decoder({1000:json.dumps(raw).encode()},0,public,v,4).accepts([1000,0])


def test_declared_optional_pointer_uses_compiled_policy(setup):
    v,p=setup
    value=[{'name':'push','params':{'target_container_name':'entity_002'}}]
    policy=bind_shared(v,4,{'entity_table':p['entities'],'generation_static_fixture_ids':[2,3]})
    tokens=['skill:push','arg:target_container_name','obj:2',v.eos_token]
    state=policy.start_state
    for token in tokens:
        label=v.label_for_token(token)
        state=policy.step(state,label) if state is not None and label in policy.allowed_tokens(state) else None
    expected=state is not None and policy.is_accepting(state)
    assert expected, 'the graph declares this optional argument; projection must preserve it'
    decoder=surface_decoder({1000:json.dumps(value).encode()},0,p,v,4)
    assert decoder.accepts([1000,0])
