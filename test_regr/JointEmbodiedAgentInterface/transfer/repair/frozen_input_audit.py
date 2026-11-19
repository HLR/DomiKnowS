"""Standalone settled-scene and decoded-image confirmation audit, no scoring."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def images(record):
    if any(sha(path)!=expected for path,expected in record['files'].items()):raise ValueError('frozen input changed')
    public=read(record['generation_input']);pixels=[]
    for path in public['images']:
        with Image.open(path) as image:
            array=np.asarray(image.convert('RGB'))
            pixels.append(digest([array.shape,hashlib.sha256(array.tobytes()).hexdigest()]))
    return digest({**{k:v for k,v in public.items() if k!='images'},'image_pixels':pixels}),digest(pixels)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--freezer',required=True);p.add_argument('--settled-audit',required=True)
    p.add_argument('--target-manifest',required=True);p.add_argument('--fresh-manifest',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();c=read(a.config)
    spec=importlib.util.spec_from_file_location('qualified_geometry',a.freezer)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    known=read(a.settled_audit)
    if not known['passed'] or known['native_geometry_resolver_sha256']!=sha(a.freezer):
        raise ValueError('previous settled audit or geometry resolver differs')
    base=Path(c['libero_base']);sys.path.insert(0,str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'))
    bridge=importlib.import_module('libero_campaign')
    new=[r for r in read(Path(c['output'])/'confirmation/manifest.json')['episodes'] if r['split']=='evaluation']
    ids={r['episode_id'] for r in new}
    if len(new)!=100 or len(ids)!=100 or {r['state_index'] for r in new}!={35,36,37,38,39}:raise ValueError('confirmation cohort changed')
    reserved=[r for r in known['records'] if r['episode_id'] not in ids]
    retired=[r for r in known['records'] if r['episode_id'] in ids]
    if len(reserved)!=800 or len(retired)!=100:raise ValueError('prepared-only validation retirement differs')
    native_records={r['episode_id']:r for path in (c['historical_manifest'],a.target_manifest,a.fresh_manifest)
        for r in read(path)['episodes']}
    old_pairs={};old_inputs={}
    for row in reserved:
        record=native_records[row['episode_id']];input_hash,pair=images(record)
        if input_hash!=row['input_sha256']:raise ValueError('previous decoded input fingerprint changed')
        old_pairs.setdefault(pair,[]).append(row['episode_id']);old_inputs[input_hash]=row['episode_id']
    grouped={}
    for record in new:grouped.setdefault(record['episode_id'].rsplit('/',1)[0],[]).append(record)
    rows=[];pairs={};scenes={r['scene_sha256'] for r in reserved}
    for case,records in sorted(grouped.items()):
        env=bridge.environment(records[0]['bddl'],render=False)
        try:
            env.reset();task=bridge.specification(records[0]['bddl'])
            for record in records:
                env.sim.set_state_from_flattened(np.load(record['state'],allow_pickle=False));env.sim.forward()
                geometry=helper.scene(env,task);input_hash,pair=images(record)
                if geometry in scenes or input_hash in old_inputs or pair in old_pairs:
                    raise ValueError('confirmation overlaps an earlier settled scene or decoded image: '+record['episode_id'])
                if pair in pairs:raise ValueError('duplicate confirmation image pair')
                pairs[pair]=record['episode_id']
                rows.append({'episode_id':record['episode_id'],'cohort':'fresh','scene_sha256':geometry,
                    'input_sha256':input_hash,'decoded_image_pair_sha256':pair})
            print(f'[confirmation-settled-audit] {len(rows)}/100 {case}',flush=True)
        finally:env.close()
    if len({r['scene_sha256'] for r in rows})!=100:raise ValueError('duplicate confirmation settled geometry')
    result={'passed':True,'confirmation':100,'reserved':800,'records':rows,'goals_scored':False,
        'retired_prepared_only_validation_ids':[r['episode_id'] for r in retired],
        'audit_source_sha256':sha(__file__),'geometry_source_sha256':sha(a.freezer),
        'prior_settled_audit_sha256':sha(a.settled_audit),'confirmation_manifest_sha256':sha(Path(c['output'])/'confirmation/manifest.json'),
        'checks':['settled geometry at five decimal places','decoded RGB image pairs','public-input fingerprints'],
        'reused_known_scene_fingerprints':True}
    helper.freeze(a.output,result)


if __name__=='__main__':main()
