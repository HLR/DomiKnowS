"""Full-vocabulary supervision and a mode-safe frozen native-Qwen teacher.

These functions define losses, never a decoding policy. No DFA is consulted.
"""
from contextlib import contextmanager
import json
import torch
import torch.nn.functional as F

SOURCE='target_entity_name'
DESTINATION='target_container_name'


def native_target(plan,visible):
    """Return canonical JSON plus pointer character spans, including unknowns."""
    chunks=['['];spans=[]
    for index,op in enumerate(plan):
        if index:chunks.append(',')
        chunks.append('{"name":'+json.dumps(op['name'])+',"params":{')
        for j,(role,value) in enumerate(op['params'].items()):
            if j:chunks.append(',')
            chunks.append(json.dumps(role)+':');start=sum(map(len,chunks))
            chunks.append(json.dumps(value));end=sum(map(len,chunks))
            if role in (SOURCE,DESTINATION):
                if type(value) is not int:raise ValueError('training target must use canonical integer pointers')
                spans.append((start,end,'source' if role==SOURCE else 'destination',
                              value<len(visible) and bool(visible[value])))
        chunks.append('}}')
    chunks.append(']')
    return ''.join(chunks),spans


def token_groups(offsets,spans):
    """Any BPE token overlapping an unknown pointer is entirely unscored."""
    groups=[]
    for left,right in offsets:
        hits=[s for s in spans if left<s[1] and right>s[0]]
        if any(not s[3] for s in hits):groups.append(-1)
        elif hits:
            roles={s[2] for s in hits}
            if len(roles)!=1:raise ValueError('one BPE token overlaps two pointer roles')
            groups.append(1 if roles=={'source'} else 2)
        else:groups.append(0)
    return groups


def graph_groups(labels,vocabulary,visible):
    groups=[];role=None
    for label in labels:
        token=vocabulary.token_for_label(int(label))
        if token.startswith('obj:'):
            pointer=int(token[4:])
            groups.append(-1 if pointer>=len(visible) or not bool(visible[pointer])
                          else 1 if role==SOURCE else 2 if role==DESTINATION else 0)
        else:groups.append(0)
        if token.startswith('arg:'):role=token[4:]
    return groups


def balanced_ce(logits,targets,groups):
    targets=torch.as_tensor(targets,device=logits.device,dtype=torch.long)
    groups=torch.as_tensor(groups,device=logits.device)
    losses=F.cross_entropy(logits.float(),targets,reduction='none')
    terms=[losses[groups==g].mean() for g in range(3) if bool((groups==g).any())]
    if not terms:raise ValueError('no observable supervision')
    return torch.stack(terms).mean()


@contextmanager
def base_teacher(model):
    """Adapters disabled; restore heterogeneous modes and trainability exactly."""
    modes=[(module,module.training) for module in model.modules()]
    flags=[(parameter,parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        with model.disable_adapter(),torch.no_grad():yield
    finally:
        for module,training in modes:module.training=training
        for parameter,requires_grad in flags:parameter.requires_grad_(requires_grad)


def native_batch(planner,context,text):
    """Append target BPE IDs to the exact generation prefix; never retokenize it.

    Position L-1 predicts target 0. Positions L..L+T-2 see only past target
    tokens. Processor image expansion therefore cannot offset target alignment.
    """
    inputs,_=planner._prepare_inputs_and_boundaries('vlabench',context,())
    tokenizer=planner.processor.tokenizer
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    targets=list(encoded['input_ids'])+[tokenizer.eos_token_id]
    offsets=list(encoded['offset_mapping'])+[(len(text),len(text))]
    inputs=planner._inputs_to_model_device(inputs);length=inputs['input_ids'].shape[1]
    suffix=torch.tensor([targets[:-1]],device=inputs['input_ids'].device)
    inputs['input_ids']=torch.cat([inputs['input_ids'],suffix],dim=1)
    inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(suffix)],dim=1)
    for name in ('mm_token_type_ids','token_type_ids'):
        if name in inputs:
            if inputs[name].shape!=inputs['input_ids'][:,:length].shape:raise ValueError('unaligned processor token types')
            inputs[name]=torch.cat([inputs[name],torch.zeros_like(suffix,dtype=inputs[name].dtype)],dim=1)
    # Recompute complete Qwen multimodal positions from the extended IDs/types.
    inputs.pop('position_ids',None);inputs.pop('cache_position',None)
    model=getattr(planner,'model',None)
    if model is not None:
        native=model.get_base_model() if hasattr(model,'get_base_model') else model
        prepare=getattr(native,'_prepare_position_ids_for_generation',None)
        if prepare is None:raise ValueError('native decoder must expose its generation position-ID interface')
        inputs['position_ids']=prepare(inputs['input_ids'],{k:v for k,v in inputs.items() if k!='input_ids'})
    return inputs,targets,offsets,length


def native_logits(model,inputs,length,count):
    # Appending T-1 gold tokens leaves exactly T next-token query positions at
    # the end. Avoid projecting thousands of image/prompt positions to 151k LM
    # logits when their logits do not contribute to either loss.
    return model(**inputs,use_cache=False,logits_to_keep=count).logits[0].float()


def graph_inputs(planner,domain,context,targets):
    vocabulary=planner.vocabulary(domain)
    active=[vocabulary.token_for_label(int(label)) for label in targets[:-1]]
    inputs,boundaries=planner._prepare_inputs_and_boundaries(domain,context,active)
    return planner._inputs_to_model_device(inputs),torch.tensor(boundaries[:len(targets)],device=planner.device)


def graph_logits(planner,domain,context,targets):
    inputs,positions=graph_inputs(planner,domain,context,targets)
    output=planner.model(**inputs,use_cache=False,output_hidden_states=True,logits_to_keep=1)
    hidden=output.hidden_states[-1][0].index_select(0,positions).float().to(planner.device)
    return planner.label_heads[domain](hidden)


def planning_loss(planner,context,plan,labels,visible,arm):
    text,spans=native_target(plan,visible)
    inputs,targets,offsets,length=native_batch(planner,context,text)
    # Both routes preserve native LM behavior on the identical native prefixes.
    with base_teacher(planner.model):
        teacher=native_logits(planner.model,inputs,length,len(targets)).detach()
    student=native_logits(planner.model,inputs,length,len(targets))
    kl=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction='batchmean')
    del teacher
    if arm=='native':ce=balanced_ce(student,targets,token_groups(offsets,spans))
    elif arm=='joint':
        labels=torch.as_tensor(labels,device=planner.device)
        eos=planner.vocabulary('vlabench').eos_label
        end=(labels==eos).nonzero().flatten()
        if not len(end):raise ValueError('missing graph EOS')
        labels=labels[:int(end[0])+1]
        logits=graph_logits(planner,'vlabench',context,labels)
        ce=balanced_ce(logits,labels,graph_groups(labels,planner.vocabulary('vlabench'),visible))
    else:raise ValueError('unknown decoder arm')
    return ce+.1*kl,{'ce':float(ce.detach()),'preservation_kl':float(kl.detach()),
                     'native_target_tokens':len(targets),'training_masks':False}


def eai_loss(planner,context,labels):
    """Same native preservation objective on EAI causal label prefixes.

    The graph head receives the student's hidden states; the frozen native LM
    head supplies the teacher and student distributions at the same boundaries.
    """
    vocabulary=planner.vocabulary('eai')
    targets=torch.as_tensor(labels,device=planner.device,dtype=torch.long)
    end=(targets==vocabulary.eos_label).nonzero().flatten()
    if not len(end):raise ValueError('missing EAI EOS')
    targets=targets[:int(end[0])+1]
    inputs,positions=graph_inputs(planner,'eai',context,targets)
    with base_teacher(planner.model):
        teacher=planner.model(**inputs,use_cache=False,logits_to_keep=positions).logits[0].float().detach()
    student=planner.model(**inputs,use_cache=False,output_hidden_states=True,logits_to_keep=positions)
    native=student.logits[0].float()
    hidden=student.hidden_states[-1][0].index_select(0,positions).float().to(planner.device)
    logits=planner.label_heads['eai'](hidden)
    ce=F.cross_entropy(logits,targets)
    kl=F.kl_div(F.log_softmax(native,dim=-1),F.softmax(teacher,dim=-1),reduction='batchmean')
    return ce+.1*kl,{'ce':float(ce.detach()),'preservation_kl':float(kl.detach()),'training_masks':False}
