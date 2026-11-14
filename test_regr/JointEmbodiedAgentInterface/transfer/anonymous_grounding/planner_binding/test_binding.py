from types import SimpleNamespace
import torch
from torch import nn
import pytest
from .binding import BindingHead,operation_bindings,operation_loss,pointer_role
from .objective import loss_terms


class Vocabulary:
    tokens=['<eos>','skill:pick','skill:place','arg:target_entity_name','arg:target_container_name',
            'obj:0','obj:1','obj:2','obj:3','skill:pull']
    label_count=len(tokens)
    def label_for_token(self,t):return self.tokens.index(t)
    def token_for_label(self,i):return self.tokens[i]


def test_full_vocabulary_ce_penalizes_illegal_logits_beyond_mass_regularizer():
    logits=torch.tensor([[0.,0.,7.]],requires_grad=True);targets=torch.tensor([0]);allowed={0:(0,1)}
    mask,_=loss_terms(logits,targets,allowed,torch.tensor([True]),'masked')
    full,_=loss_terms(logits,targets,allowed,torch.tensor([True]),'full')
    masked_gradient=torch.autograd.grad(mask,logits,retain_graph=True)[0]
    full_gradient=torch.autograd.grad(full,logits)[0]
    assert full>mask+5 and full_gradient[0,2]>masked_gradient[0,2]+.9


def test_unknown_pointer_targets_never_receive_ce_or_auxiliary_supervision():
    logits=torch.randn(2,3,requires_grad=True);targets=torch.tensor([0,2]);observed=torch.tensor([True,False])
    loss,terms=loss_terms(logits,targets,{0:(0,1),1:(2,)},observed,'full');loss.backward()
    assert torch.equal(logits.grad[1],torch.zeros(3)) and terms['unknown_pointer_tokens']==1
    values=torch.tensor([[2.,1.],[0.,3.],[-1.,-1.]],requires_grad=True)
    aux,coverage=operation_loss([{'step':0,'role':'source','column':0,'logits':values}],torch.tensor([6]),Vocabulary(),torch.tensor([True,False]))
    assert aux==0 and coverage['source']=={'cases':1,'observed':0,'correct':0}


def test_operation_targets_are_distinct_not_an_episode_union():
    v=Vocabulary();events=[];targets=torch.tensor([5,6,7,8]);losses=[]
    for i in range(4):
        scores=torch.zeros(5,2);column=i%2;scores[i,column]=4.;scores.requires_grad_()
        events.append({'step':i,'role':('source','destination')[column],'column':column,'logits':scores})
    loss,coverage=operation_loss(events,targets,v,torch.ones(4,dtype=torch.bool));loss.backward()
    assert coverage['source']['correct']==coverage['destination']['correct']==2
    # Swapping targets between two source operations must incur an error.
    wrong=targets[torch.tensor([2,1,0,3])]
    swapped,_=operation_loss(events,wrong,v,torch.ones(4,dtype=torch.bool))
    assert swapped>loss+1


def test_zero_initialized_residual_preserves_logits_and_never_forces_argmax():
    head=BindingHead(4,6,8);v=Vocabulary();visible=torch.tensor([True,True])
    scores=torch.tensor([[9.,0.],[-2.,0.],[-4.,0.]],requires_grad=True)
    assert torch.equal(head.residual(scores,0,visible,v),torch.zeros(v.label_count))
    head.gain.data.fill_(2);bias=head.residual(scores,0,visible,v)
    original=torch.zeros(v.label_count);original[6]=20
    assert int((original+bias).argmax())==6 and torch.isfinite(original+bias).all()
    assert bias.abs().max()<=4 and bias[9]==0
    (original+bias).sum().backward() # ordinary finite autograd graph


def test_teacher_forcing_hook_uses_prefix_and_cleans_up_after_failure():
    v=Vocabulary();head=BindingHead(4,6,8);base=nn.Linear(4,v.label_count)
    prepare=lambda *a:None
    planner=SimpleNamespace(label_heads={'vlabench':base},vocabulary=lambda d:v,_prepare_inputs_and_boundaries=prepare)
    tokens=['skill:pick','arg:target_entity_name','obj:0','skill:place','arg:target_container_name']
    features=torch.randn(4,6);visible=torch.ones(4,dtype=torch.bool);hidden=torch.randn(6,4,requires_grad=True)
    before=base(hidden).detach()
    with operation_bindings(planner,head,features,visible) as events:
        planner._prepare_inputs_and_boundaries('vlabench',{'instruction':'put it away'},tokens)
        actual=base(hidden)
        assert [(e['step'],e['role']) for e in events]==[(2,'source'),(5,'destination')]
        torch.testing.assert_close(actual,before) # zero gain preserves initialization
        auxiliary,_=operation_loss(events,torch.tensor([1,3,5,2,4,6]),v,visible)
        auxiliary.backward();assert hidden.grad.abs().sum()>0
    assert planner._prepare_inputs_and_boundaries is prepare and not base._forward_hooks
    with pytest.raises(ValueError,match='targets'):
        with operation_bindings(planner,head,features,visible):
            planner._prepare_inputs_and_boundaries('vlabench',{'reference_plan':[1]},tokens)
    assert planner._prepare_inputs_and_boundaries is prepare and not base._forward_hooks


def test_live_decoding_hook_uses_only_emitted_arguments():
    v=Vocabulary();head=BindingHead(4,6,8);layer=nn.Linear(4,v.label_count)
    planner=SimpleNamespace(label_heads={'vlabench':layer},vocabulary=lambda d:v,_prepare_inputs_and_boundaries=lambda *a:None)
    with operation_bindings(planner,head,torch.randn(4,6),torch.ones(4,dtype=torch.bool)) as events:
        for prefix in [[],['skill:pick'],['skill:pick','arg:target_entity_name']]:
            planner._prepare_inputs_and_boundaries('vlabench',{'instruction':'move it'},prefix);layer(torch.randn(4))
        assert [(e['step'],e['role']) for e in events]==[(2,'source')]
    assert pointer_role(['skill:place']) is None
