"""Fast source-only attention/precision probe; no optimization or goal scoring."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from ..protocol import read,freeze,sha
from ..matched_baseline.protocol import common_prompt,prompt_core
from ..anonymous_grounding.augmentation import identity_epoch
from ..source_transform import transform
from .common import fresh_planner
from .alignment import verify_alignment


def main():
    from domiknows.generation.dfa.vocabulary import TokenVocabulary
    from test_regr.VLABenchAgentInterface.dataset import load_planning_examples,planning_metadata
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary,plan_to_tokens,labels_to_plan
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--initialization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=read(a.config);payload=torch.load(a.initialization,map_location='cpu',weights_only=True)
    if not payload['initialization_before_updates']:raise ValueError('source-only pre-update probe required')
    vocabulary=PlanVocabulary.default(max_entities=c['max_entities'])
    eai=TokenVocabulary(payload['vocabularies']['eai'],eos_token='<eos>')
    runtime=SimpleNamespace(eai_vocabulary=eai,vlabench_vocabulary=vocabulary)
    record=next(r for r in read(c['source_observations'])['episodes'] if r['split']=='train')
    episode=Path(record['scene_path']).parents[1]
    original=load_planning_examples(episode,observation_mode='metadata-assisted')[0]
    task,dimension=planning_metadata(record['episode_id'])
    original=replace(original,episode_id=record['episode_id'],task=task,dimension=dimension)
    changed=not identity_epoch(original.episode_id,len(original.entities),0)
    example=transform(original,record,epoch=0,augmented=changed,pointer_keys=set(vocabulary.argument_keys))
    image=record['augmentations'][0]['image'] if changed else record['canonical_image']
    bound={'instruction':example.instruction,'entity_table':list(example.entities),'images':(record['raw_image'],image)}
    public={'instruction':example.instruction,'entities':list(example.entities),'images':list(bound['images']),
            'movable':list(example.entities),'initial_relations':[]}
    bound['_matched_core']=prompt_core(public,vocabulary);bound['_matched_roles']=None
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    world=build_vlabench_world_graph('alignment_probe')
    labels=[vocabulary.label_for_token(t) for t in plan_to_tokens(example.operation_sequence,example.entities,world=world)]
    plan=labels_to_plan(labels,vocabulary,world=world)
    del payload
    planner=fresh_planner(c,runtime,'native');planner.eval();base=planner.model.get_base_model();reports=[]
    for implementation in ('sdpa','eager'):
        base.set_attn_implementation(implementation)
        try:
            with common_prompt(planner):result=verify_alignment(planner,bound,plan,labels,[True]*len(example.entities))
        except (ValueError,RuntimeError) as exc:result={'passed':False,'error':str(exc)}
        reports.append({'attention':implementation,**result});print(json.dumps(reports[-1]),flush=True)
    freeze(a.output,{'source_only':True,'optimizer_updates':0,'config_sha256':sha(a.config),
        'initialization_sha256':sha(a.initialization),'episode_id':example.episode_id,'reports':reports,
        'tf32':torch.backends.cuda.matmul.allow_tf32,'bf16_reduced_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})


if __name__=='__main__':main()
