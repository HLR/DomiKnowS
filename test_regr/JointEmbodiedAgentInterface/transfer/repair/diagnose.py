"""Historical-only interventions and native surface-projection verification."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from ..protocol import read,freeze,sha,episodes,public_input
from .surface import surface_decoder,SURFACE_VERSION


def replay(config,output):
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    from ..native_baseline import compiled_decoder
    c=read(config);archive=Path(c['historical_archive']);rows=[]
    source=archive/'evaluation/base/none/predictions.jsonl'
    originals={r['episode_id']:r for r in map(json.loads,source.read_text().splitlines())}
    vocabulary=PlanVocabulary.default(max_entities=c['max_entities'])
    for episode in episodes(read(c['historical_manifest'])):
        identity=episode['episode_id'];row=originals[identity];public=public_input(episode)
        # One merged token tests the entire byte projection, independent of
        # tokenizer segmentation. Actual decoding is a separate experiment.
        tokens={1:row['raw_text'].encode('utf-8')}
        old=compiled_decoder(tokens,0,public,vocabulary,12)
        new=surface_decoder(tokens,0,public,vocabulary,12)
        before,after=old.accepts([1,0]),new.accepts([1,0])
        if row['valid_plan'] and not row['generation_error'] and not after:
            raise ValueError('legal historical greedy output still rejected: '+identity)
        rows.append({'episode_id':identity,'historical_accepted':before,'repaired_accepted':after,
            'normalized_prediction':row['prediction'],'raw_text':row['raw_text']})
    freeze(Path(output)/'surface_replay.json',{'surface':SURFACE_VERSION,'source_sha256':sha(source),
        'episodes':rows,'greedy_preservation_required':True,'actual_constrained_decoding_required':True})


@contextmanager
def diagnostic_binding(planner,head,features,visible,enabled):
    """Capture planner logits and exact added residual at each causal pointer."""
    import torch
    from ..anonymous_grounding.planner_binding.binding import pointer_role
    original=planner._prepare_inputs_and_boundaries;state={'tokens':()};events=[]
    def prepare(domain,context,active_tokens=()):
        if domain=='vlabench':state['tokens']=tuple(active_tokens)
        return original(domain,context,active_tokens)
    def hook(module,inputs,output):
        if output.ndim!=1:raise ValueError('diagnostic hook is cached decoding only')
        role=pointer_role(state['tokens'])
        if role is None:return output
        name,column=role;scores=head(inputs[0],features,visible)
        correction=head.residual(scores,column,visible,planner.vocabulary('vlabench'))
        after=output+correction if enabled else output
        events.append({'step':len(state['tokens']),'prefix':list(state['tokens']),'role':name,
            'planner_logits':output.detach().float().cpu().tolist(),
            'grounding_scores':scores.detach().float().cpu().tolist(),
            'residual_logits':correction.detach().float().cpu().tolist(),
            'corrected_logits':after.detach().float().cpu().tolist(),'residual_enabled':enabled})
        return after
    planner._prepare_inputs_and_boundaries=prepare
    handle=planner.label_heads['vlabench'].register_forward_hook(hook)
    try:yield events
    finally:handle.remove();planner._prepare_inputs_and_boundaries=original


def intervention(config,arm,output):
    import torch
    from ..matched_baseline import evaluate as matched
    from ..anonymous_grounding.planner_binding import binding
    c=read(config);historical=read(c['historical_config']);historical['output']=str(output)
    historical['input_hashes']={**historical['input_hashes'],str(config):sha(config)}
    path=Path(output)/'config.json';freeze(path,historical)
    planner_key,residual=arm.split(':');real_load=matched.load_joint;real_binding=binding.operation_bindings
    def load(conf,key,device):
        planner,own,runtime,details=real_load(conf,key,device)
        _,head,_,_=real_load(conf,'B',device) if key!='B' else (None,own,None,None)
        details.update(diagnostic_intervention=True,planner_weights=key,residual_head='B',
                       residual_applied=residual=='B',head_transplant=key!='B')
        return planner,head,runtime,details
    def hook(planner,head,features,visible):return diagnostic_binding(planner,head,features,visible,residual=='B')
    # Wrap the hook to archive all pointer distributions. The original evaluator
    # continues to own prompts, frozen priors, scoring and decision records.
    @contextmanager
    def trace(*args):
        with hook(*args) as events:
            yield events
        log=Path(output)/'pointer_interventions.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:stream.write(json.dumps({'episode_index':trace.index,'events':events})+'\n')
        trace.index+=1
    trace.index=0
    matched.load_joint=load;binding.operation_bindings=trace
    original_argv=sys.argv
    try:
        sys.argv=['diagnostic','--config',str(path),'--model-key',planner_key,'--policy','none']
        matched.main()
    finally:matched.load_joint=real_load;binding.operation_bindings=real_binding;sys.argv=original_argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['replay','base-repaired','initialization:off','initialization:B','B:off','B:B'],required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();c=read(a.config)
    from .protocol import verify
    verify(c)
    if a.checkpoint:
        key=a.arm.split(':')[0]
        expected=c['model'] if key in ('replay','base-repaired') else read(c['historical_config'])['snapshots'][key]['path']
        if str(a.checkpoint)!=expected:raise ValueError('diagnostic checkpoint differs from frozen intervention')
    if a.seed!=42 or sha(a.manifest)!=sha(c['historical_manifest']):raise ValueError('historical diagnostic cohort differs')
    if a.arm=='replay':replay(a.config,a.output)
    elif a.arm=='base-repaired':
        from .. import native_baseline
        from ..matched_baseline import evaluate
        native_baseline.compiled_decoder=surface_decoder
        conf=read(c['historical_config']);conf['output']=str(a.output)
        conf['repair_surface']=SURFACE_VERSION;conf['input_hashes'][str(a.config)]=sha(a.config)
        path=a.output/'config.json';freeze(path,conf)
        sys.argv=['repaired','--config',str(path),'--model-key','base','--policy','shared-verified']
        evaluate.main()
    else:intervention(a.config,a.arm,a.output)


if __name__=='__main__':main()
