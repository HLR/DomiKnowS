"""Freeze confirmation scenes only after historical-use and provenance gates.

Run in the qualified LIBERO simulator environment. No planner is loaded here.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.repair'
from ..protocol import read,freeze,sha,digest
from ..anonymous_grounding.planner_binding.freeze_external import scene,carry_calibration


def prepare(c,config_path,output):
    usage=read(c['confirmation_usage_audit'])
    if not usage.get('passed') or usage.get('unresolved') or usage.get('prior_model_use'):
        raise ValueError('confirmation provenance unresolved; no scoring or substitution permitted')
    base=Path(c['libero_base']);bridge_path=base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    sys.path.insert(0,str(bridge_path.parent));bridge=importlib.import_module('libero_campaign')
    from libero.libero import benchmark
    from libero.libero.envs import SegmentationRenderEnv
    original=read(c['historical_manifest']);records=[];fingerprints=[]
    example=next(e for e in original['episodes'] if e['split']=='evaluation')
    with Image.open(read(example['generation_input'])['images'][0]) as image:width,height=image.size
    cases={f"{e['suite']}/{e['task']}":e for e in original['episodes']}
    for case,previous in sorted(cases.items()):
        if sha(previous['bddl'])!=previous['source_bddl_sha256']:raise ValueError('native specification changed')
        suite_name,task_name=case.split('/',1);suite=benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        index=next(i for i in range(suite.n_tasks) if suite.get_task(i).name==task_name)
        states=suite.get_task_init_states(index)
        if len(states)!=50:raise ValueError('required official states unavailable')
        spec=bridge.specification(previous['bddl'])
        env=SegmentationRenderEnv(bddl_file_name=previous['bddl'],camera_heights=height,camera_widths=width,
            horizon=1000,control_freq=20,use_camera_obs=True)
        try:
            env.reset()
            for state_index in range(50):
                env.sim.set_state_from_flattened(np.asarray(states[state_index]));env.sim.forward()
                fingerprints.append({'case':case,'state_index':state_index,
                    'split':'confirmation' if 35<=state_index<=39 else 'other',
                    'scene_sha256':scene(env,spec),
                    'official_state_sha256':hashlib.sha256(np.asarray(states[state_index]).tobytes()).hexdigest()})
            for state_index in range(35,40):
                identity=f'{case}/state{state_index:02d}';folder=output/'evaluation'/identity;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'record.json').exists():
                    record=read(folder/'record.json')
                    if any(sha(p)!=h for p,h in record['files'].items()):raise ValueError('frozen scene changed')
                    records.append(record);continue
                seed=int(bridge.digest(identity)[:8],16);random.seed(seed);np.random.seed(seed);env.seed(seed);env.reset()
                obs=env.set_init_state(states[state_index])
                for _ in range(10):obs,_,_,_=env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
                state_path=folder/'initial_state.npy';np.save(state_path,env.get_sim_state(),allow_pickle=False)
                raw,numbered,visible=bridge.numbered_image(obs,env,spec['entities'])
                raw_path=folder/'raw.png';numbered_path=folder/'numbered.png';Image.fromarray(raw).save(raw_path);numbered.save(numbered_path)
                public=bridge.generation_input(spec,[raw_path,numbered_path]);public['visible_entity_ids']=visible
                freeze(folder/'generation_input.json',public);freeze(folder/'scoring_private.json',spec)
                paths=[state_path,raw_path,numbered_path,folder/'generation_input.json',folder/'scoring_private.json']
                record={**{k:v for k,v in previous.items() if k not in ('episode_id','state_index','split','state','generation_input','scoring_private','files')},
                    'episode_id':identity,'state_index':state_index,'split':'evaluation','state':str(state_path),
                    'generation_input':str(folder/'generation_input.json'),'scoring_private':str(folder/'scoring_private.json'),
                    'files':{str(f):sha(f) for f in paths}}
                freeze(folder/'record.json',record);records.append(record)
                print(f'[repair-freeze] {len(records)}/100 {identity}',flush=True)
        finally:env.close()
    for field in ('scene_sha256','official_state_sha256'):
        held=[r[field] for r in fingerprints if r['split']=='confirmation']
        other={r[field] for r in fingerprints if r['split']=='other'}
        if len(held)!=100 or len(set(held))!=100 or set(held)&other:
            raise ValueError('confirmation overlaps another official scene: '+field)
    manifest={**original,'episodes':records+[e for e in original['episodes'] if e['split']=='calibration'],
        'protocol':'Predeclared confirmation states 35-39, excluded from subsequent training/validation',
        'usage_audit_sha256':sha(c['confirmation_usage_audit']),'config_sha256':sha(config_path)}
    path=output/'manifest.json';freeze(path,manifest)
    freeze(output/'split_audit.json',{'passed':True,'fingerprints':fingerprints,'precision_digits':5})
    gate=carry_calibration(original,manifest,read(base/'calibration_v3/summary.json'),Path(c['historical_manifest']),path,bridge_path)
    freeze(output/'calibration_reuse.json',gate)
    reservation={'campaign':c['output'],'manifest_sha256':sha(path),'episodes':[
        {k:r[k] for k in ('episode_id','suite','task','state_index')} for r in records],
        'prohibited_uses':['training','validation','selection','executor_repair','calibration']}
    freeze(output/'reservation.json',reservation)
    freeze(Path(c['reservation_directory'])/(Path(c['output']).name+'.json'),reservation)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--arm',choices=['both'],default='both')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.seed!=42:raise ValueError('seed is frozen')
    c=read(a.config)
    if a.manifest and sha(a.manifest)!=sha(c['historical_manifest']):raise ValueError('preparation source manifest differs')
    if a.checkpoint and str(a.checkpoint)!=c['model']:raise ValueError('preparation does not adapt a trained checkpoint')
    prepare(c,a.config,a.output)


if __name__=='__main__':main()
