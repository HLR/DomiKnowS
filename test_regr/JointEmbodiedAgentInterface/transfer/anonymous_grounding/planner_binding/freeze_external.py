"""Standalone native-runtime freeze of unused LIBERO states 40--44."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def freeze(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and read(p)!=v:raise ValueError('frozen external artifact changed: '+str(p))
    p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')


def carry_calibration(old_manifest,new_manifest,gate,old_path,new_path,bridge):
    before=[e for e in old_manifest['episodes'] if e['split']=='calibration']
    after=[e for e in new_manifest['episodes'] if e['split']=='calibration']
    if len(before)!=100 or before!=after:raise ValueError('calibration states or observations changed')
    if gate['manifest_sha256']!=sha(old_path) or gate['success_rate']<.9:raise ValueError('original calibration not qualified')
    if gate['executor_source_sha256']!=sha(bridge.parent/'fixed_executor.py') or gate['bridge_source_sha256']!=sha(bridge):
        raise ValueError('qualified native executor or reset bridge changed')
    return {**gate,'manifest_sha256':sha(new_path),'calibration_reused':True,
        'original_manifest_sha256':sha(old_path),'calibration_records_sha256':digest(before),
        'reason':'Identical 100 calibration records and native executor/reset code; only held-out evaluation records replaced.'}


def scene(env,spec):
    values=[]
    for name in spec['entities']:
        model=env.env.objects_dict.get(name) or env.env.fixtures_dict.get(name)
        if model is not None:body=model.root_body
        elif name in getattr(env.sim.model,'body_names',()):body=name
        elif name in getattr(env.sim.model,'geom_names',()):
            i=env.sim.model.geom_name2id(name)
            values.append([name,'static_geom',np.round(env.sim.data.geom_xpos[i],5).tolist(),np.round(env.sim.data.geom_xmat[i],5).tolist()])
            continue
        elif spec['entity_types'][name] in getattr(env.sim.model,'body_names',()):
            # E.g. a declared main_table uses the native arena body named table.
            body=spec['entity_types'][name]
        else:raise ValueError('cannot fingerprint declared entity '+name)
        i=env.sim.model.body_name2id(body)
        values.append([name,np.round(env.sim.data.body_xpos[i],5).tolist(),np.round(env.sim.data.body_xquat[i],5).tolist()])
    return digest(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config)
    base=Path(c['libero_base']);bridge_path=base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    sys.path.insert(0,str(bridge_path.parent));bridge=importlib.import_module('libero_campaign')
    from libero.libero import benchmark
    from libero.libero.envs import SegmentationRenderEnv
    original=read(c['libero_manifest']);root=Path(c['output'])/'fresh_libero';records=[];fingerprints=[]
    example=next(e for e in original['episodes'] if e['split']=='evaluation')
    with Image.open(read(example['generation_input'])['images'][0]) as image:width,height=image.size
    cases={f"{e['suite']}/{e['task']}":e for e in original['episodes']}
    for case,previous in sorted(cases.items()):
        if sha(previous['bddl'])!=previous['source_bddl_sha256']:raise ValueError('original native task specification changed')
        suite_name,task_name=case.split('/',1);suite=benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        index=next(i for i in range(suite.n_tasks) if suite.get_task(i).name==task_name)
        states=suite.get_task_init_states(index)
        if len(states)<45:raise ValueError('required unused official states unavailable')
        spec=bridge.specification(previous['bddl'])
        env=SegmentationRenderEnv(bddl_file_name=previous['bddl'],camera_heights=height,camera_widths=width,
            horizon=1000,control_freq=20,use_camera_obs=True)
        try:
            env.reset()
            # Compare scene geometry, not robot-state/noise/time differences.
            for state_index in range(45):
                env.sim.set_state_from_flattened(np.asarray(states[state_index]));env.sim.forward()
                fingerprints.append({'case':case,'state_index':state_index,'split':'fresh' if state_index>=40 else 'reserved',
                    'scene_sha256':scene(env,spec),'official_state_sha256':hashlib.sha256(np.asarray(states[state_index]).tobytes()).hexdigest()})
            for state_index in range(40,45):
                identity=f'{case}/state{state_index:02d}';folder=root/'evaluation'/identity;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'record.json').exists():
                    previous_record=read(folder/'record.json')
                    if any(sha(p)!=h for p,h in previous_record['files'].items()):raise ValueError('partial frozen episode changed')
                    if previous_record['source_bddl_sha256']!=sha(previous['bddl']):raise ValueError('partial frozen native specification changed')
                    records.append(previous_record);continue
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
                print(f'[fresh-libero] {len(records)}/100 {identity}',flush=True)
        finally:env.close()
    for field in ('scene_sha256','official_state_sha256'):
        reserved={e[field] for e in fingerprints if e['split']=='reserved'}
        if any(e[field] in reserved for e in fingerprints if e['split']=='fresh'):raise ValueError('fresh evaluation overlaps reserved '+field)
        fresh=[e[field] for e in fingerprints if e['split']=='fresh']
        if len(fresh)!=len(set(fresh)):raise ValueError('duplicate physical scenes inside fresh evaluation')
    if len(records)!=100:raise ValueError('fresh external coverage differs')
    manifest={**original,'episodes':records+[e for e in original['episodes'] if e['split']=='calibration'],
        'protocol':'Fresh held-out LIBERO states 40--44; source-only planner training; fixed qualified executor',
        'fresh_states':[40,41,42,43,44],'training_states_used':False}
    path=root/'manifest.json';freeze(path,manifest)
    freeze(root/'split_audit.json',{'passed':True,'fingerprints':fingerprints,'evaluated_episodes':100,
        'reserved_official_states':list(range(40)),'precision_digits':5,'scene_entities':'all declared entities; robot and time excluded'})
    gate=carry_calibration(original,manifest,read(base/'calibration_v3/summary.json'),Path(c['libero_manifest']),path,bridge_path)
    freeze(root/'calibration_reuse.json',gate)


if __name__=='__main__':main()
