from contextlib import contextmanager
import pytest
import torch
from .objective import native_target,token_groups,balanced_ce,base_teacher,native_batch,native_logits
from .protocol import select,reject_reserved,checkpoint_kind,VERSION,normalize_anonymous_prediction


def test_unknown_merged_pointer_token_is_entirely_masked():
    text,spans=native_target([{'name':'pick','params':{'target_entity_name':12}}],[True]*12+[False])
    start,end,_,_=spans[0]
    assert text[start:end]=='12'
    assert token_groups([(start-3,end+2),(0,5),(len(text),len(text))],spans)==[-1,0,0]
    logits=torch.randn(3,20,requires_grad=True)
    loss=balanced_ce(logits,[12,1,2],[-1,0,1]);loss.backward()
    assert not logits.grad[0].any()
    assert logits.grad[1:].abs().sum()>0


def test_equal_group_weight_not_syntax_frequency():
    logits=torch.tensor([[0.,2.],[0.,2.],[2.,0.]])
    targets=[0,0,0]
    actual=balanced_ce(logits,targets,[0,0,1])
    expected=(torch.nn.functional.cross_entropy(logits[:2],torch.tensor(targets[:2]))+
              torch.nn.functional.cross_entropy(logits[2:],torch.tensor(targets[2:])))/2
    assert torch.allclose(actual,expected)


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(2,2);self.active=True
    @contextmanager
    def disable_adapter(self):
        self.active=False
        try:yield
        finally:self.active=True


def test_teacher_exception_restores_mixed_modes_and_trainability():
    model=Teacher().train();model.layer.eval();model.layer.bias.requires_grad_(False)
    before=[m.training for m in model.modules()];flags=[p.requires_grad for p in model.parameters()]
    with pytest.raises(RuntimeError),base_teacher(model):
        assert not model.active and not model.training and not torch.is_grad_enabled()
        model.layer.weight.requires_grad_(False)
        raise RuntimeError('teacher failed')
    assert model.active
    assert [m.training for m in model.modules()]==before
    assert [p.requires_grad for p in model.parameters()]==flags


def test_causal_native_alignment_and_future_exclusion():
    class Tokenizer:
        eos_token_id=9
        def __call__(self,*a,**kw):return {'input_ids':[3,4,5],'offset_mapping':[(0,1),(1,2),(2,3)]}
    class Planner:
        class Processor:tokenizer=Tokenizer()
        processor=Processor()
        def _prepare_inputs_and_boundaries(self,*a):
            return {'input_ids':torch.tensor([[1,1,1,1,2]]),'attention_mask':torch.ones(1,5,dtype=torch.long),
                    'mm_token_type_ids':torch.tensor([[0,1,1,1,0]])},[4]
        def _inputs_to_model_device(self,x):return x
    batch,targets,offsets,length=native_batch(Planner(),{},'abc')
    assert length==5 and targets==[3,4,5,9]
    assert batch['input_ids'].tolist()==[[1,1,1,1,2,3,4,5]]
    assert batch['mm_token_type_ids'].tolist()==[[0,1,1,1,0,0,0,0]]
    class Causal:
        def __call__(self,input_ids,**kwargs):
            class Output:pass
            output=Output();output.logits=input_ids.cumsum(1).unsqueeze(-1).float()[:,-kwargs['logits_to_keep']:];return output
    actual=native_logits(Causal(),batch,length,4)
    # A cached decoder sees only this prefix at each decision.
    cached=torch.stack([batch['input_ids'][0,:length+i].sum() for i in range(4)]).float()
    assert torch.equal(actual[:,0],cached)


def test_selection_fixed_gates_and_ties():
    rows=[{'epoch':i,'eai_successes':47,'progress_score':50.,'source_pointer_accuracy':.5} for i in range(1,5)]
    assert select(rows,51)['selected']['epoch']==1
    rows[0]['eai_successes']=46
    assert select(rows,51)['selected']['epoch']==2
    assert not select(rows,52)['qualified']


def test_reservation_and_decoder_guards():
    e={'suite':'libero_object','task':'x','state_index':35,'split':'validation'}
    with pytest.raises(ValueError):reject_reserved({'episodes':[e]},{'episodes':[e]})
    for arm in ('native','joint'):
        checkpoint_kind({'format':VERSION,'arm':arm},arm)
        with pytest.raises(ValueError):checkpoint_kind({'format':VERSION,'arm':arm},'joint' if arm=='native' else 'native')


def test_private_source_names_cannot_be_resolved_as_anonymous_ids():
    roles=('target_entity_name','target_container_name')
    for alias in ('private_red_block','table','00','red block',True,0.0):
        with pytest.raises(ValueError,match='unknown public'):
            normalize_anonymous_prediction([{'name':'pick','params':{'target_entity_name':alias}}],3,roles)
    for alias in (1,'1','entity_001','numbered object 1'):
        value,_=normalize_anonymous_prediction([{'name':'pick','params':{'target_entity_name':alias}}],3,roles)
        assert value[0]['params']['target_entity_name']==1


def test_source_prompt_does_not_expose_canonical_object_names():
    from types import SimpleNamespace
    from .common import source_context
    planner=SimpleNamespace(vocabulary=lambda domain:SimpleNamespace(skill_arguments=(('pick',('target_entity_name',)),)))
    item={'instruction':'pick the red block','entity_table':['SECRET_BLOCK','SECRET_TRAY'],'images':()}
    context=source_context(planner,{'item':item})
    assert 'SECRET' not in context['_matched_core']
    assert context['_public_entity_aliases']==['entity_000','entity_001']
    other=source_context(planner,{'item':{**item,'entity_table':['OTHER_PRIVATE','ANOTHER_PRIVATE']}})
    assert context['_matched_core']==other['_matched_core']
