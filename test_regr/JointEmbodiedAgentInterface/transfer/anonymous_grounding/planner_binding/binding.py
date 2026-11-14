"""Causal operation grounding with a learned finite pointer-logit residual."""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from ..grounder import RoleGrounder

ROLES={'arg:target_entity_name':('source',0),'arg:target_container_name':('destination',1)}
FORBIDDEN=('reference','reference_plan','operation_sequence','target_plan_labels','native_success',
           'hidden_goal','goal','goals','obj_of_interest','scoring_private','reference_labels')


class BindingHead(nn.Module):
    def __init__(self,query_dim,visual_dim=1548,hidden_dim=256,max_gain=4.):
        super().__init__();self.role_head=RoleGrounder(query_dim,visual_dim,hidden_dim)
        self.gain=nn.Parameter(torch.zeros(()));self.max_gain=float(max_gain)
        self.query_dim=query_dim;self.visual_dim=visual_dim;self.hidden_dim=hidden_dim

    def forward(self,query,features,visible):return self.role_head(query,features,visible)

    def residual(self,logits,role,visible,vocabulary):
        # Every graph label stays available. Missing observations receive no bias.
        probabilities=F.softmax(logits[:,role].float(),dim=0)[:-1]
        active=visible.nonzero().flatten()
        result=logits.new_zeros(vocabulary.label_count)
        if len(active):
            pointers=torch.tensor([vocabulary.label_for_token('obj:'+str(int(i))) for i in active],device=logits.device)
            values=probabilities[active]-probabilities[active].mean()
            result=result.index_add(0,pointers,self.max_gain*torch.tanh(self.gain)*values)
        return result


def pointer_role(prefix):
    # A role is available from the emitted argument, before its object pointer.
    return ROLES.get(prefix[-1]) if prefix else None


@contextmanager
def operation_bindings(planner,head,features,visible):
    """Same head hook in teacher forcing and cached greedy decoding.

    Batched hidden row t sees prefix[:t]; decoding sees only emitted labels.
    No reference or future operation labels enter the residual provider.
    """
    original=planner._prepare_inputs_and_boundaries;state={'tokens':()};events=[]
    def prepare(domain,context,active_tokens=()):
        if domain=='vlabench':
            if any(k in context for k in FORBIDDEN):raise ValueError('binding context contains evaluator/training targets')
            state['tokens']=tuple(active_tokens)
        return original(domain,context,active_tokens)
    def hook(module,inputs,output):
        hidden=inputs[0];rows=hidden[None] if hidden.ndim==1 else hidden
        logits=output[None] if output.ndim==1 else output
        if hidden.ndim not in (1,2):raise ValueError('binding requires causal label-boundary hidden states')
        changed=[];tokens=state['tokens'];vocabulary=planner.vocabulary('vlabench')
        for i,(query,values) in enumerate(zip(rows,logits)):
            prefix=tokens if hidden.ndim==1 else tokens[:i]
            role=pointer_role(prefix)
            if role is None:changed.append(values);continue
            name,column=role;scores=head(query,features,visible)
            events.append({'step':len(prefix),'role':name,'column':column,'logits':scores})
            changed.append(values+head.residual(scores,column,visible,vocabulary))
        result=torch.stack(changed)
        return result[0] if output.ndim==1 else result
    planner._prepare_inputs_and_boundaries=prepare
    handle=planner.label_heads['vlabench'].register_forward_hook(hook)
    try:yield events
    finally:
        handle.remove();planner._prepare_inputs_and_boundaries=original


def operation_loss(events,targets,vocabulary,visible):
    losses=[];coverage={r:{'cases':0,'observed':0,'correct':0} for r in ('source','destination')}
    for event in events:
        step=event['step']
        if step>=len(targets):continue
        token=vocabulary.token_for_label(int(targets[step]))
        if not token.startswith('obj:'):raise ValueError('declared argument must be followed by its pointer target')
        pointer=int(token[4:]);metrics=coverage[event['role']];metrics['cases']+=1
        if pointer>=len(visible) or not bool(visible[pointer]):continue
        scores=event['logits'][:,event['column']];metrics['observed']+=1
        metrics['correct']+=int(int(scores.detach().argmax())==pointer)
        losses.append(F.cross_entropy(scores[None],torch.tensor([pointer],device=scores.device)))
    zero=events[0]['logits'][-1].sum()*0 if events else targets.new_zeros((),dtype=torch.float32)
    return torch.stack(losses).mean() if losses else zero,coverage
