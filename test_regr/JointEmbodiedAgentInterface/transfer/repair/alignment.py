"""Source-only real-model alignment checks; no optimizer updates or target data."""
import torch
from .objective import native_batch,native_logits,native_target,graph_logits

NATIVE_KL_LIMIT=1e-3


def distribution_divergence(expected,actual):
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():raise ValueError('nonfinite alignment logits')
    return float(torch.nn.functional.kl_div(torch.log_softmax(actual.float(),-1),
        torch.softmax(expected.float(),-1),reduction='sum').clamp_min(0))


def assert_future_independent(expected,changed,count):
    if not torch.allclose(expected[:count],changed[:count],atol=1e-5,rtol=1e-6):
        raise ValueError('future source labels affected an earlier teacher-forced decision')


def verify_alignment(planner,context,plan,labels,visible):
    from transformers import LogitsProcessorList
    from ...models import FreeDecoding
    planner.eval();text,_=native_target(plan,visible)
    inputs,targets,_,length=native_batch(planner,context,text)
    with torch.no_grad():full=native_logits(planner.model,inputs,length,len(targets)).detach()
    future=[]
    with torch.no_grad():
        replacement=planner.processor.tokenizer.encode(' anonymous',add_special_tokens=False)[0]
        for step in sorted({0,min(3,len(targets)-1)}):
            changed=dict(inputs);ids=inputs['input_ids'].clone();ids[:,length+step:]=replacement;changed['input_ids']=ids
            scored=native_logits(planner.model,changed,length,len(targets))
            assert_future_independent(full,scored,step+1)
            future.append({'step':step,'max_prior_logit_change':float((full[:step+1]-scored[:step+1]).abs().max())})
    prefix={k:(v[...,:length] if k in ('input_ids','attention_mask','mm_token_type_ids','token_type_ids','position_ids') else v)
            for k,v in inputs.items()}
    differences=[];divergences=[];top_matches=[];position_checks=[]
    native=planner.model.get_base_model()
    def check_positions(module,args,kwargs):
        step=len(position_checks);positions=kwargs['position_ids']
        expected=inputs['position_ids'][...,:length] if step==0 else inputs['position_ids'][...,length+step-1:length+step]
        if not torch.equal(positions,expected):raise ValueError('cached native causal/RoPE positions differ')
        position_checks.append(True)
    handle=native.model.language_model.register_forward_pre_hook(check_positions,with_kwargs=True)
    class ForcedSourcePrefix:
        def __call__(self,ids,scores):
            step=ids.shape[1]-length
            if step>=len(targets):raise ValueError('cached alignment exceeded source target')
            difference=(scores[0].float()-full[step]).abs().max()
            differences.append(float(difference))
            divergence=distribution_divergence(full[step],scores[0]);divergences.append(divergence)
            top_matches.append(int(scores[0].argmax())==int(full[step].argmax()))
            if divergence>NATIVE_KL_LIMIT:
                raise ValueError(f'native cached distribution differs: step={step}, KL={divergence}, max_abs={float(difference)}')
            result=torch.full_like(scores,-torch.inf);result[0,targets[step]]=0.;return result
    try:
        with torch.inference_mode():
            generated=planner.model.generate(**prefix,do_sample=False,use_cache=True,max_new_tokens=len(targets),
                logits_processor=LogitsProcessorList([ForcedSourcePrefix()]),
                eos_token_id=planner.processor.tokenizer.eos_token_id)[0,length:].tolist()
    finally:handle.remove()
    if generated!=targets:raise ValueError('forced cached source prefix differs')
    vocabulary=planner.vocabulary('vlabench');gold=torch.as_tensor(labels,device=planner.device)
    end=(gold==vocabulary.eos_label).nonzero().flatten();gold=gold[:int(end[0])+1]
    with torch.no_grad():graph=graph_logits(planner,'vlabench',context,gold)
    graph_differences=[]
    def force(module,args,output):
        if output.ndim!=1:raise ValueError('expected cached single-boundary hidden state')
        step=len(graph_differences)
        graph_differences.append(float((output-graph[step]).abs().max()))
        if not torch.allclose(output,graph[step],atol=.02,rtol=.01):
            raise ValueError('graph teacher-forcing / cached decoding misalignment')
        result=torch.full_like(output,-torch.inf);result[int(gold[step])]=0.;return result
    handle=planner.label_heads['vlabench'].register_forward_hook(force)
    try:
        planner.free_decoding=True
        with torch.inference_mode():
            emitted,_=planner.sample_labels('vlabench',context,FreeDecoding(vocabulary.label_count,vocabulary.eos_label),
                                           max_steps=len(gold),deterministic=True)
    finally:handle.remove()
    if emitted!=gold.tolist():raise ValueError('forced graph prefix differs')
    return {'passed':True,'native_max_absolute_error':max(differences),'graph_max_absolute_error':max(graph_differences),
        'native_max_kl':max(divergences),'native_kl_limit':NATIVE_KL_LIMIT,
        'native_greedy_disagreements':top_matches.count(False),'exact_cached_position_checks':len(position_checks),
        'future_label_exclusion':future,'precision_note':'Standard NF4/BF16; elementwise roundoff is reported separately from causal alignment',
        'native_steps':len(targets),'graph_steps':len(gold),'source_only':True,'optimizer_updates':0}
