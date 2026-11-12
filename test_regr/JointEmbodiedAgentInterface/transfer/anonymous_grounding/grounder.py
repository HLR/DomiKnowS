"""Index-free role scoring and predicted-only planner conditioning."""
import torch
from torch import nn
import torch.nn.functional as F


class RoleGrounder(nn.Module):
    def __init__(self,query_dim,visual_dim=1548,hidden_dim=256):
        super().__init__();self.query_dim=query_dim;self.visual_dim=visual_dim;self.hidden_dim=hidden_dim
        self.query=nn.Sequential(nn.Linear(query_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.visual=nn.Sequential(nn.Linear(visual_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.relations=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.relation_norm=nn.LayerNorm(hidden_dim)
        self.score=nn.Sequential(nn.Linear(hidden_dim*3,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,2))
        self.null=nn.Linear(hidden_dim,2)

    def forward(self,query,entities,visible):
        q=self.query(query.float()).reshape(1,-1);e=self.visual(entities.float())
        if bool(visible.any()):
            relation,_=self.relations(e[None],e[None],e[None],key_padding_mask=(~visible)[None],need_weights=False)
            e=self.relation_norm(e+relation[0])
        expanded=q.expand(e.shape[0],-1);logits=self.score(torch.cat([expanded,e,expanded*e],dim=-1))
        logits=logits.masked_fill(~visible[:,None],float('-inf'))
        return torch.cat([logits,self.null(q)],dim=0)


def instruction_query(planner,instruction):
    # Keep the original Tensor so auxiliary gradients reach Qwen LoRA; wrapping
    # it in Tensor._make_subclass would create a detached EncodedContext leaf.
    context={'instruction':instruction,'entity_table':(),'images':()}
    inputs,boundaries=planner._prepare_inputs_and_boundaries('vlabench',context,())
    output=planner.model(**planner._inputs_to_model_device(inputs),output_hidden_states=True,use_cache=False)
    hidden=getattr(output,'hidden_states',None) or getattr(output,'decoder_hidden_states',None)
    if hidden is None:raise ValueError('grounding query requires shared-backbone hidden states')
    return hidden[-1][0,min(boundaries[0],hidden[-1].shape[1]-1)].float()


def predict_roles(logits,visible):
    n=len(visible);result={}
    for role,column in [('source',0),('destination',1)]:
        best=int(logits[:,column].detach().argmax())
        result[role]=None if best==n or not bool(visible[best]) else best
    return result


def condition(context,predictions):
    # The only added binding is the head's prediction. No gold role, task
    # template, canonical name or reference plan enters this function.
    value=dict(context)
    value['entity_table']=tuple(f'numbered object {i}' for i in range(len(context['entity_table'])))
    source='unknown' if predictions['source'] is None else str(predictions['source'])
    destination='unknown' if predictions['destination'] is None else str(predictions['destination'])
    value['instruction']=context['instruction']+f'\nPredicted primary visual roles (uncertain): source ID {source}; destination ID {destination}. Verify against the images.'
    return value


def role_targets(plan,entities):
    result={'source':set(),'destination':set()}
    for operation in plan:
        for key,role in [('target_entity_name','source'),('target_container_name','destination')]:
            if key not in operation['params']:continue
            value=operation['params'][key]
            if type(value) is int or isinstance(value,str) and value.isdigit():pointer=int(value)
            elif isinstance(value,str) and value in entities:pointer=entities.index(value)
            else:raise ValueError('invalid source grounding pointer')
            if not 0<=pointer<len(entities):raise ValueError('source grounding pointer outside inventory')
            result[role].add(pointer)
    return result


def grounding_loss(logits,visible,targets):
    losses=[];coverage={}
    for column,role in enumerate(('source','destination')):
        gold=targets[role];observed=sorted(i for i in gold if bool(visible[i]))
        coverage[role]={'targets':len(gold),'observable_targets':len(observed)}
        if not gold:positions=[len(visible)]
        elif observed:positions=observed
        else:continue  # Unknown is not a gold substitute or a negative example.
        losses.append(-F.log_softmax(logits[:,column],dim=0)[positions].mean())
    # A connected, finite zero when neither role is observable.
    return (torch.stack(losses).mean() if losses else logits[-1].sum()*0),coverage


def known_pointer_steps(labels,vocabulary,visible):
    keep=torch.ones(len(labels),dtype=torch.bool,device=labels.device)
    for i,label in enumerate(labels.tolist()):
        token=vocabulary.token_for_label(int(label))
        if token.startswith('obj:'):
            pointer=int(token[4:])
            keep[i]=pointer<len(visible) and bool(visible[pointer])
    return keep
