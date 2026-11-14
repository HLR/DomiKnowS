"""Standalone geometry audit of actual frozen states, not just official seeds."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def verify(rows):
    fresh=[r for r in rows if r['cohort']=='fresh'];reserved=[r for r in rows if r['cohort']=='reserved']
    if len(fresh)!=100 or len(reserved)!=800:raise ValueError('actual frozen-scene coverage differs')
    for field in ('episode_id','scene_sha256','input_sha256'):
        known={r[field] for r in reserved}
        if any(r[field] in known for r in fresh):raise ValueError('fresh/reserved settled overlap: '+field)
    if len({r['scene_sha256'] for r in fresh})!=100:raise ValueError('duplicate settled fresh scenes')
    return {'passed':True,'fresh':100,'reserved':800,'scene_precision_digits':5,'robot_and_time_excluded':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--freezer',required=True)
    p.add_argument('--target-manifest',default='/home/auszok/experiments/joint_transfer_20261110_attempt3/target_data/libero_sharded/manifest.json')
    a=p.parse_args();spec=importlib.util.spec_from_file_location('frozen_native_preparer',a.freezer)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    c=helper.read(a.config);root=Path(c['output']);base=Path(c['libero_base'])
    sys.path.insert(0,str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'))
    bridge=importlib.import_module('libero_campaign')
    old=helper.read(c['libero_manifest'])['episodes'];target=helper.read(a.target_manifest)['episodes']
    fresh=[r for r in helper.read(root/'fresh_libero/manifest.json')['episodes'] if r['split']=='evaluation']
    grouped={}
    for cohort,records in [('reserved',old+target),('fresh',fresh)]:
        for record in records:grouped.setdefault(record['episode_id'].rsplit('/',1)[0],[]).append((cohort,record))
    fingerprints=[]
    for case,records in sorted(grouped.items()):
        env=bridge.environment(records[0][1]['bddl'],render=False)
        try:
            env.reset();task=bridge.specification(records[0][1]['bddl'])
            for cohort,record in records:
                if any(helper.sha(p)!=h for p,h in record['files'].items()):raise ValueError('reserved/fresh frozen input changed')
                env.sim.set_state_from_flattened(np.load(record['state'],allow_pickle=False));env.sim.forward()
                public=helper.read(record['generation_input']);image_hashes=[]
                for path in public['images']:
                    with Image.open(path) as image:
                        array=np.asarray(image.convert('RGB'));image_hashes.append(helper.digest([array.shape,helper.hashlib.sha256(array.tobytes()).hexdigest()]))
                fingerprints.append({'episode_id':record['episode_id'],'cohort':cohort,
                    'scene_sha256':helper.scene(env,task),
                    'input_sha256':helper.digest({**{k:v for k,v in public.items() if k!='images'},'image_pixels':image_hashes})})
            print(f'[settled-audit] {len(fingerprints)}/900 {case}',flush=True)
        finally:env.close()
    result=verify(fingerprints)
    helper.freeze(root/'fresh_libero/settled_split_audit.json',{**result,'records':fingerprints,
        'manifest_sha256':helper.sha(root/'fresh_libero/manifest.json'),'target_manifest_sha256':helper.sha(a.target_manifest),
        'original_manifest_sha256':helper.sha(c['libero_manifest']),'audit_source_sha256':helper.sha(__file__),
        'native_geometry_resolver_sha256':helper.sha(a.freezer)})


if __name__=='__main__':main()
