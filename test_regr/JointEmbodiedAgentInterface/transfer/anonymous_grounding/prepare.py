"""Observation-only source preprocessing. Never reads operation plans."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import json

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.anonymous_grounding'

from ..protocol import read, freeze, sha, digest
from ..source_views import allocate
from ..permutation import deterministic_permutations
from .markers import atlas,recognize,render
from .provenance import validate_source


def prepare_one(config,record,output):
    from test_regr.VLABenchAgentInterface.dataset import _extract_entities
    identity=record['episode_id'];root=Path(config['planning_dir'])/identity
    scenes=sorted((root/'env_config').glob('*.json'))
    if len(scenes)!=1:raise ValueError('unambiguous original source configuration required')
    scene=read(scenes[0]);task=scene.get('task',scene)
    names=list(_extract_entities({k:task[k] for k in ('components','scene','entities') if k in task}))
    if not 2<=len(names)<=64:raise ValueError('unsupported canonical entity count')
    raw,mask=root/'input/input.png',root/'input/input_mask.png'
    folder=output/'episodes'/digest(identity)[:24];path=folder/'record.json'
    if path.exists():
        saved=read(path)
        if any(sha(p)!=h for p,h in saved['files'].items()):raise ValueError('resumed source observation differs')
        return saved
    proof=recognize(raw,mask,len(names));folder.mkdir(parents=True,exist_ok=True)
    original=list(range(len(names)));render(raw,proof['markers'],original,folder/'canonical.png')
    augmentations=[]
    for epoch,permutation in enumerate(deterministic_permutations(identity,len(names),seed=42,trials=4)):
        image=folder/f'epoch_{epoch:03d}.png';render(raw,proof['markers'],permutation.order,image)
        augmentations.append({'epoch':epoch,'new_to_old':list(permutation.order),'image':str(image)})
    files=[raw,mask,scenes[0],*folder.glob('*.png')]
    saved={**record,'canonical_entities':names,'scene_path':str(scenes[0]),'scene_sha256':sha(scenes[0]),
           'raw_image':str(raw),'canonical_image':str(folder/'canonical.png'),'augmentations':augmentations,
           'binding':proof,'files':{str(p):sha(p) for p in files}}
    freeze(path,saved);return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();config=read(args.config);splits=read(config['split_manifest']);records=allocate(splits)
    validate_source(config)
    for identity,spec in atlas()['training_sources'].items():
        if identity not in splits['splits']['train']:raise ValueError('glyph atlas must use training images only')
        if sha(Path(config['planning_dir'])/identity/spec['image'])!=spec['sha256']:
            raise ValueError('frozen training typography source differs')
    freeze(args.output/'settings.json',{'split_sha256':sha(config['split_manifest']),
        'atlas_sha256':sha(Path(__file__).with_name('glyph_atlas.json')),
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('markers.py')]},
        'input':'original clean RGB and reannotated RGB; no simulator reconstruction, no ID-coded colors',
        'ambiguous_bindings':'unknown; no target-derived substitutions',
        'pointer_supervision':'unknown image-to-ID bindings are masked, preserving every episode'})
    completed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index,row in enumerate(pool.map(lambda r:prepare_one(config,r,args.output),records)):
            completed.append(row)
            if (index+1)%25==0:print(f'[anonymous-grounding] prepared {index+1}/4050',flush=True)
    if len({r['episode_id'] for r in completed})!=4050:raise ValueError('source identity coverage differs')
    freeze(args.output/'manifest.json',{'episodes':completed,'complete':True,'train':3600,'validation':450,
        'test_used':False,'references_read':False,'split_sha256':sha(config['split_manifest']),
        'mean_visible_entities':sum(len(r['binding']['visible_ids']) for r in completed)/4050,
        'zero_binding_episodes':sum(not r['binding']['visible_ids'] for r in completed)})
    print('[anonymous-grounding] original-image preparation complete',flush=True)


if __name__=='__main__':main()
