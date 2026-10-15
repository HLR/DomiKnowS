import itertools
import json

import pytest
import torch

from .native_planning import NativePlanDFA, NativeConstraintProcessor, byte_token_map, save_lookahead_table, restore_lookahead_table
from .planning_experiment import sequence_nll, read_records, append_record, load_graph_planner, scene_only_examples


def character_dfa(**kwargs):
    tokens = {byte: bytes([byte]) for byte in [9,10,13,*range(32, 127)]}
    tokens[130] = b'[{"name":"pick","params":{"target_entity_name":'
    tokens[131] = b'}},{"name":"place","params":{"target_container_name":'
    tokens[132] = b'}}]'
    return NativePlanDFA(tokens, 133, entity_count=2, **kwargs)


def plan(entity=0, container=1):
    return [{"name": "pick", "params": {"target_entity_name": entity}},
            {"name": "place", "params": {"target_container_name": container}}]


def labels(value):
    return list(json.dumps(value, separators=(",", ":")).encode()) + [133]


def test_byte_projection_handles_json_boundary_spanning_tokens():
    dfa = character_dfa()
    assert dfa.accepts(labels(plan()))
    assert dfa.accepts([130, ord("0"), 131, ord("1"), 132, 133])
    assert not dfa.accepts([130, ord("0"), 131, ord("1"), 132])
    assert not dfa.accepts([133])


@pytest.mark.parametrize("value", [[], plan(2), plan(-1), plan("0"), [plan()[0]], [plan()[1]], [{"name":"pick","params":{}}]])
def test_independent_invalid_plans_are_rejected(value):
    assert not character_dfa().accepts(labels(value))


def test_operation_bound_and_complete_subtask_extensions():
    assert character_dfa(max_operations=2).accepts(labels(plan()))
    assert not character_dfa(max_operations=1).accepts(labels(plan()))
    assert character_dfa(max_operations=4).accepts(labels(plan()+plan()))
    extended = [plan()[0], {"name":"pour","params":{"target_container_name":0}}, plan()[1]]
    assert character_dfa().accepts(labels(extended))


def test_dfa_does_not_start_subtask_that_cannot_fit_operation_budget():
    dfa=character_dfa(max_operations=1)
    state=dfa.start_state
    for byte in b'[{"name":"pi':
        state=dfa.byte_step(state,byte)
        if state is None:
            break
    assert state is None  # pick needs a second operation; press fits one.
    assert dfa.accepts(labels([{'name':'press','params':{'target_entity_name':0}}]))


def test_byte_level_surface_decoding():
    tokenizer = type("T", (), {"all_special_ids": [3], "get_vocab": lambda self: {"Ġ[":0, "abc":1, "Ċ":2, "<eos>":3}})()
    assert byte_token_map(tokenizer) == {0:b" [", 1:b"abc", 2:b"\n"}


def test_zero_hmm_weight_exactly_matches_dfa_only():
    dfa = character_dfa()
    scores = torch.arange(140).float().unsqueeze(0)
    a = NativeConstraintProcessor(dfa, 1, 200)
    b = NativeConstraintProcessor(dfa, 1, 200, hmm=(torch.ones(1),torch.ones(1,1),torch.ones(1,140)/140),weight=0)
    assert torch.equal(a(torch.tensor([[9]]),scores),b(torch.tensor([[9]]),scores))
    assert torch.isneginf(a(torch.tensor([[9]]), scores)[0,133])


def test_forward_likelihood_and_masking_against_latent_enumeration():
    from domiknows.generation.learners.hmm.discrete.discreteHMMLearner import HMMGenerationHead
    head = HMMGenerationHead(label_count=3,state_count=2,random_seed=42).double()
    x = torch.tensor([[0,1,2],[1,2,0]])
    lengths = torch.tensor([3,2])
    probability = []
    for row, length in zip(x, lengths):
        total = 0.
        for states in itertools.product(range(2),repeat=int(length)):
            value = head.initial_probs[states[0]] * head.emission_probs[states[0],row[0]]
            for index in range(1,int(length)):
                value = value * head.transition_probs[states[index-1],states[index]] * head.emission_probs[states[index],row[index]]
            total = total + value
        probability.append(total)
    expected = -torch.stack(probability).log().sum()/lengths.sum()
    actual = sequence_nll(head,x,lengths)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert head.emission_logits.grad.isfinite().all()
    x[1,2]=2
    torch.testing.assert_close(sequence_nll(head,x,lengths),actual)


def test_success_dp_against_exhaustive_observation_enumeration(tmp_path):
    from domiknows.generation.applications.hmm_dfa_decoder import HMMDFAStaticLookahead
    from domiknows.generation.dfa.stop_policy import StopPolicy
    initial=torch.tensor([.4,.6],dtype=torch.float64)
    transition=torch.tensor([[.7,.3],[.2,.8]],dtype=torch.float64)
    emission=torch.tensor([[.3,.7],[.6,.4]],dtype=torch.float64)
    # Language accepts label 0 followed by EOS label 1.
    table=HMMDFAStaticLookahead(transition,emission,StopPolicy(max_steps=3),1,3,
        lambda state, depth: ((0,1),) if state==0 else ((1,2),) if state==1 else (), lambda state: state==2)
    expected=sum(initial[i]*emission[i,0]*transition[i,j]*emission[j,1] for i,j in itertools.product(range(2),repeat=2))
    torch.testing.assert_close(table.success_after(initial,0,3),expected)
    assert table.success_after(initial,0,1)==0
    path=tmp_path/'cache.pt'
    save_lookahead_table(table,path,'same-model')
    restored=HMMDFAStaticLookahead(transition,emission,StopPolicy(max_steps=3),1,3,
        lambda state, depth: ((0,1),) if state==0 else ((1,2),) if state==1 else (), lambda state: state==2)
    assert not restore_lookahead_table(restored,path,'different-model')
    assert restore_lookahead_table(restored,path,'same-model')
    torch.testing.assert_close(restored.success_after(initial,0,3),expected)


def test_resume_repairs_only_torn_final_record_and_rejects_duplicates(tmp_path):
    path=tmp_path/'episodes.jsonl'
    append_record(path,{'episode_id':'one'})
    with path.open('a') as stream:
        stream.write('{"episode_id":')
    assert list(read_records(path))==['one']
    append_record(path,{'episode_id':'two'})
    assert list(read_records(path))==['one','two']
    append_record(path,{'episode_id':'two'})
    with pytest.raises(ValueError,match='duplicate'):
        read_records(path)


def test_checkpoint_version_gate_precedes_model_loading(tmp_path):
    path=tmp_path/'bad.pt'
    torch.save({'standalone_checkpoint_version':1},path)
    with pytest.raises(ValueError,match='version-5'):
        load_graph_planner(path,'unused')


def test_experimental_entity_tables_do_not_expose_gold_argument_ids(tmp_path):
    episode=tmp_path/'select_fruit'/'example0'
    (episode/'output').mkdir(parents=True)
    (episode/'env_config').mkdir()
    (episode/'env_config'/'scene.json').write_text(json.dumps({'entities':['apple','bowl','pear']}))
    output=episode/'output'/'operation_sequence.json'
    output.write_text(json.dumps({'skill_sequence':plan(0,1)}))
    first=scene_only_examples(tmp_path)[0]
    assert first.entities==('apple','bowl','pear')
    output.write_text(json.dumps({'skill_sequence':plan(2,1)}))
    second=scene_only_examples(tmp_path)[0]
    assert first.entities==second.entities
    assert first.operation_sequence!=second.operation_sequence


def test_benchmark_push_role_long_plans_and_unrestricted_scene_ids():
    tokens={byte:bytes([byte]) for byte in range(32,127)}
    dfa=NativePlanDFA(tokens,133,entity_count=70)
    push=[plan(65,69)[0],{'name':'push','params':{'target_container_name':69}}]
    assert dfa.accepts(labels(push))
    assert dfa.accepts(labels(plan()*5))  # Ten operations were previously impossible.
    assert not dfa.accepts(labels([plan()[0],{'name':'push','params':{}}]))
    legacy=NativePlanDFA(tokens,133,entity_count=70,grammar_profile='legacy')
    assert legacy.accepts(labels([plan()[0],{'name':'push','params':{}}]))
    assert not legacy.accepts(labels(push))
    assert not legacy.accepts(labels(plan()*5))


@pytest.mark.parametrize('text',[
    json.dumps(plan(),indent=2),
    ' \r\n[ { "params" : { "target_entity_name" : 0 }, "name" : "pick" },\t'
    '{"params":{"target_container_name":1},"name":"place"} ] \n',
    '[{"na\\u006de":"p\\u0069ck","params":{"target_entity_name":0}},'
    '{"params":{"target_container_name":1},"name":"place"}]',
])
def test_equivalent_json_formatting_and_member_order(text):
    assert json.loads(text)==plan()
    assert character_dfa().accepts(list(text.encode())+[133])


@pytest.mark.parametrize('text',[
    '[{"name":"pi ck","params":{"target_entity_name":0}}]',
    '[{"name":"pick","name":"pick","params":{"target_entity_name":0}}]',
    '[{"name":"pick","params":{"target_entity_name":0}},]',
    '[{"name":"pick","params":{"target_entity_name":01}}]',
    '[{"name":"pick","params":{"target_entity_name":1 0}}]',
    '[{"name":"pick","params":{"target_entity_name":1.0}}]',
    '[{"name":"pick","params":{"target_entity_name":true}}]',
    '[{"name":"pick","params":{"target_entity_name":0},"extra":0}]',
])
def test_flexible_json_does_not_accept_invalid_syntax_or_extra_members(text):
    assert not character_dfa().accepts(list(text.encode())+[133])


def test_whitespace_and_escape_tokens_cross_json_boundaries():
    tokens={0:b' [ {"params" : {"target_entity_name": 0}, "na\\u00',
            1:b'6de":"pick"},\n{"name":"place","params":{"target_container_name":1}}] \t'}
    dfa=NativePlanDFA(tokens,2,entity_count=2)
    assert dfa.accepts([0,1,2])
    assert not dfa.accepts([0,2])


def test_grounding_is_skill_role_specific_and_independent_of_goals():
    from .object_grounding import scene_objects, instruction_object_constraints
    entities=('table','apple','pear','plate_seen','apple_tag')
    config={'task':{'components':[
        {'name':'table','class':'Table'}, {'name':'apple','class':'Fruit'},
        {'name':'pear','class':'Fruit'}, {'name':'plate_seen','class':'Plate'},
        {'name':'apple_tag','class':'NameTag','content':'apple'}],
        'conditions':{'contain':{'entities':['pear'],'container':'table'}}}}
    grounding=instruction_object_constraints('Put the apple into the plate_seen',scene_objects(config,entities))
    assert grounding.ids('pick','target_entity_name',5)==(1,)
    assert grounding.ids('place','target_container_name',5)==(3,)
    assert grounding.ids('press','target_entity_name',5)==tuple(range(5))
    config['task']['conditions']={'target_entity':'apple_tag'}
    assert grounding==instruction_object_constraints('Put the apple into the plate_seen',scene_objects(config,entities))
    tokens={byte:bytes([byte]) for byte in range(32,127)}
    dfa=NativePlanDFA(tokens,133,entity_count=5,object_constraints=grounding)
    assert dfa.accepts(labels(plan(1,3)))
    assert not dfa.accepts(labels(plan(2,3)))
    assert not dfa.accepts(labels(plan(1,0)))


@pytest.mark.parametrize('instruction',[
    'Pick the fruit that is not an apple', 'Pick the second fruit from the left',
    'Put the apple in the tray into the plate_seen', 'Find something suitable',
    'Please add something besides an apple', 'Pick a fruit other than the apple',
])
def test_grounding_abstains_on_unsupported_interpretations(instruction):
    from .object_grounding import SceneObject,instruction_object_constraints
    constraints=instruction_object_constraints(instruction,(SceneObject(0,'apple','Fruit'),SceneObject(1,'pear','Fruit')))
    assert constraints.restrictions==()


def test_type_grounding_retains_candidates_instead_of_guessing_the_answer():
    from .object_grounding import SceneObject,instruction_object_constraints
    objects=(SceneObject(0,'table','Table'),SceneObject(1,'CuSO4','ChemistryTube'),
             SceneObject(2,'I2','ChemistryTube'),SceneObject(3,'CuSO4_tag','NameTag'))
    constraints=instruction_object_constraints('Take out the blue solution',objects)
    assert constraints.ids('pick','target_entity_name',4)==(1,2)
    assert constraints.ids('pour','target_container_name',4)==(0,1,2,3)


def test_scene_only_loader_does_not_append_goal_only_entities(tmp_path):
    episode=tmp_path/'select_fruit'/'example0'
    (episode/'output').mkdir(parents=True);(episode/'env_config').mkdir()
    (episode/'env_config/scene.json').write_text(json.dumps({'task':{'components':[
        {'name':'apple','class':'Fruit'},{'name':'plate','class':'Plate'}],
        'conditions':{'target_entity':'GOLD_ONLY_SECRET'}}}))
    (episode/'output/operation_sequence.json').write_text(json.dumps({'skill_sequence':plan()}))
    assert scene_only_examples(tmp_path)[0].entities==('apple','plate')


def test_frozen_generation_preserves_raw_logits_and_accepts_spaced_json(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from PIL import Image
    from .native_planning import FrozenNativePlanner
    from .dataset import PlanningExample
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(LogitsProcessorList=list))
    class Tokenizer:
        eos_token_id=133
        all_special_ids=[133]
        def get_vocab(self):
            return {**{chr(i):i for i in range(33,127)},'Ġ':32,'<eos>':133}
        def decode(self,ids,skip_special_tokens=True):
            return ''.join(chr(i) for i in ids if i!=133)
    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[1]]))
            self.input_ids=self['input_ids']
        def to(self,device): return self
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self,*args,**kwargs): return 'prompt'
        def __call__(self,**kwargs): return Inputs()
    class Model:
        def parameters(self): return iter([torch.zeros(1)])
        def generate(self,**kwargs):
            self.kwargs=kwargs
            output=list(json.dumps(plan()).encode())+[133]
            prefix=kwargs['input_ids']
            for token in output:
                for mask in kwargs.get('logits_processor',[]):
                    scores=mask(prefix,torch.zeros(1,134))
                    assert scores[0,token].isfinite()
                prefix=torch.cat([prefix,torch.tensor([[token]])],dim=1)
            return prefix
    image=tmp_path/'image.png'; Image.new('RGB',(2,2)).save(image)
    example=PlanningExample('one','Pick an object',tuple(plan()),(image,),(),None,entities=('apple','plate'))
    model=Model(); planner=FrozenNativePlanner(model,Processor())
    assert json.loads(planner.generate(example,mode='raw')[0])==plan()
    assert 'logits_processor' not in model.kwargs
    assert 'eos_token_id' not in model.kwargs
    assert model.kwargs['do_sample'] is False
    assert json.loads(planner.generate(example,mode='dfa')[0])==plan()
    assert model.kwargs['logits_processor']


def test_evaluator_persists_native_and_graph_validity_separately_and_resumes(tmp_path,monkeypatch):
    from argparse import Namespace
    from . import planning_experiment as module
    episode=tmp_path/'data/select_fruit/example0'
    (episode/'output').mkdir(parents=True);(episode/'env_config').mkdir();(episode/'input').mkdir()
    reference=[{'name':'pick','params':{'target_entity_name':1}},
               {'name':'push','params':{'target_container_name':3}}]
    (episode/'env_config/scene.json').write_text(json.dumps({'task':{'components':[
        {'name':'table','class':'Table'},{'name':'apple','class':'Fruit'},
        {'name':'pear','class':'Fruit'},{'name':'plate_seen','class':'Plate'}]}}))
    (episode/'input/instruction.txt').write_text('Put the apple into the plate_seen')
    (episode/'output/operation_sequence.json').write_text(json.dumps({'skill_sequence':reference}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'episode_ids':['select_fruit/example0']}))
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    class Planner:
        checksum='fake-tokenizer'
        def generate(self,*args,**kwargs): return json.dumps(reference),[0]
    monkeypatch.setattr(module.FrozenNativePlanner,'load',lambda *args:Planner())
    args=Namespace(output=str(tmp_path/'run'),planning_dir=str(tmp_path/'data'),manifest=str(manifest),
        limit=None,mode='dfa',max_tokens=768,max_operations=16,model=str(model),load_in_4bit=False,
        checkpoint=None,hmm=None,weight=0,seed=42,grammar_profile='benchmark',object_constraints='instruction')
    module.experiment(args)
    path=tmp_path/'run/episodes.jsonl';before=path.read_bytes();record=json.loads(before)
    assert not record['valid']  # Legacy graph push cannot carry an argument.
    assert record['native_valid'] and record['grounded_valid']
    assert record['scores']['score']==100
    assert record['instruction_constraints']['restrictions']
    module.experiment(args)
    assert path.read_bytes()==before
    args.object_constraints='none'
    with pytest.raises(ValueError,match='changed experiment settings'):
        module.experiment(args)


def test_hmm_acceptance_with_whitespace_cycles_against_exhaustive_enumeration():
    from domiknows.generation.applications.hmm_dfa_decoder import HMMDFAStaticLookahead
    from domiknows.generation.dfa.stop_policy import StopPolicy
    dfa=NativePlanDFA({0:json.dumps(plan()).encode(),1:b' '},2,entity_count=2)
    initial=torch.tensor([.4,.6],dtype=torch.float64)
    transition=torch.tensor([[.7,.3],[.2,.8]],dtype=torch.float64)
    emission=torch.tensor([[.3,.2,.5],[.2,.5,.3]],dtype=torch.float64)
    expected=initial.new_tensor(0.)
    for length in range(1,5):
        for tokens in itertools.product(range(3),repeat=length):
            if not dfa.accepts(tokens): continue
            belief=initial.clone()
            for token in tokens:
                belief=(belief*emission[:,token])@transition
            expected+=belief.sum()
    table=HMMDFAStaticLookahead(transition,emission,StopPolicy(max_steps=4),2,4,
        lambda state,depth:tuple((label,target) for label,target in dfa.transitions(state)
                                if depth!=1 or dfa.is_accepting(target)),dfa.is_accepting)
    torch.testing.assert_close(table.success_after(initial,dfa.start_state,4),expected)


@pytest.mark.parametrize('raw',[
    'not JSON', '{"name":"pick"}',
    '[{"name":"pick","params":{"target_entity_name":0},"extra":1}]',
    '[{"name":"pick","name":"pick","params":{"target_entity_name":0}}]',
    '[{"name":[],"params":{}}]',
    '[{"name":"press","params":{"target_entity_name":true}}]',
])
def test_native_diagnostic_rejects_malformed_raw_json_without_affecting_metrics(raw):
    from .native_planning import validate_native_plan
    valid,errors=validate_native_plan(raw,('apple','plate'))
    assert not valid and errors
