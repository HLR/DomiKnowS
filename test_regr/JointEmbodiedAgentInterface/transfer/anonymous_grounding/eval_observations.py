"""Frozen evaluation observations; no plans, goals or simulator targets are read."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from ..protocol import read,freeze,sha,digest
from .prepare import prepare_one


def verify_configuration(config):
    if sha(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('selected snapshot changed')
    if sha(config['grounding_head'])!=config['grounding_sha256']:raise ValueError('selected grounding head changed')
    if sha(config['evaluation_source_manifest'])!=config['evaluation_source_manifest_sha256']:
        raise ValueError('evaluation source manifest changed')
    for name,expected in read(config['evaluation_source_manifest'])['files'].items():
        if sha(Path(config['source'])/name)!=expected:raise ValueError('frozen evaluator changed: '+name)


def observed_markers(raw_path,numbered_path,count,style):
    if style not in ('libero-v3','rlbench-v33'):raise ValueError('unsupported public annotation style')
    width,height,offset=(18,17,3) if style=='libero-v3' else (19,17,2)
    with Image.open(raw_path) as im:raw=np.asarray(im.convert('RGB'))
    with Image.open(numbered_path) as im:numbered=np.asarray(im.convert('RGB'))
    if raw.shape!=numbered.shape:raise ValueError('raw and numbered observations differ in shape')
    h,w=raw.shape[:2];markers=[];ambiguous=[]
    for pointer in range(count):
        canvas=Image.new('RGB',(width,height));draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,width-1,height-1),fill='white',outline='black');draw.text((offset,offset),str(pointer),fill='black')
        stamp=np.asarray(canvas);possible=np.ones((h-height+1,w-width+1),dtype=bool)
        for y,x in ((0,0),(0,width-1),(height-1,0),(height-1,width-1),(1,1),(height-2,width-2)):
            possible&=np.all(numbered[y:y+h-height+1,x:x+w-width+1]==stamp[y,x],axis=-1)
        ys,xs=np.where(possible)
        matches=[(int(x),int(y)) for y,x in zip(ys,xs)
                 if np.array_equal(numbered[y:y+height,x:x+width],stamp)
                 and not np.array_equal(raw[y:y+height,x:x+width],stamp)]
        if len(matches)==1:
            x,y=matches[0];markers.append({'id':pointer,'box':[x,y,width,height],'view':0})
        elif matches:ambiguous.append(pointer)
    return {'markers':markers,'ambiguous_omitted':ambiguous,'visible_ids':sorted(m['id'] for m in markers),
        'protocol':'unique complete public '+style+' numeric stamps; no geometry or reference lookup'}


def libero_markers(raw_path,numbered_path,count):
    return observed_markers(raw_path,numbered_path,count,'libero-v3')


def prepare(config,output):
    training=read(config['training_configuration']);split=read(training['split_manifest'])
    freeze(output/'settings.json',{'config_sha256':sha(config['configuration_path']),
        'split_sha256':sha(training['split_manifest']),'libero_manifest_sha256':sha(config['libero_manifest']),
        'source_sha256':sha(__file__),'source_inputs':'original RGB and exact public numeric stamps only'})
    identities=split['splits']['test']
    if len(identities)!=450 or len(set(identities))!=450:raise ValueError('require frozen 450 test identities')
    def one(identity):
        return prepare_one(training,{'episode_id':identity,'split':'test'},output/'vlabench')
    with ThreadPoolExecutor(max_workers=4) as pool:
        vla=[]
        for i,row in enumerate(pool.map(one,identities)):
            vla.append(row)
            if (i+1)%25==0:print(f'[grounding-eval] VLA observations {i+1}/450',flush=True)
    from ..protocol import episodes,public_input
    external=[]
    for episode in episodes(read(config['libero_manifest'])):
        public=public_input(episode);raw,numbered=public['images']
        proof=libero_markers(raw,numbered,len(public['entities']))
        folder=output/'libero'/digest(episode['episode_id'])[:24];path=folder/'crop_rgb.png'
        with Image.open(raw) as im:
            w,h=im.size
            if w!=h:raise ValueError('LIBERO normalized camera adapter requires square views')
            folder.mkdir(parents=True,exist_ok=True)
            if not path.exists():im.convert('RGB').resize((480,480),Image.Resampling.BILINEAR).save(path)
        markers=[{**m,'box':[x*480/w,y*480/h,mw*480/w,mh*480/h]}
                 for m in proof['markers'] for x,y,mw,mh in [m['box']]]
        # Integer crop anchors preserve the same normalized-camera convention.
        for m in markers:m['box']=[round(v) for v in m['box']]
        external.append({'episode_id':episode['episode_id'],'split':'evaluation',
            'canonical_entities':public['entities'],'raw_image':str(path),
            'binding':{**proof,'markers':markers},'files':{raw:sha(raw),numbered:sha(numbered),str(path):sha(path)},
            'public_input':episode['generation_input'],'public_sha256':sha(episode['generation_input'])})
    freeze(output/'manifest.json',{'complete':True,'vlabench':vla,'libero':external,
        'test_training_used':False,'references_read':False,
        'libero_adapter':'single public camera scaled to 480 for anchor crops; original planner images unchanged; other three camera slots absent',
        'coverage':{k:{'episodes':len(rows),'zero_binding':sum(not r['binding']['markers'] for r in rows),
            'visible_entities':sum(len(r['binding']['visible_ids']) for r in rows),
            'entities':sum(len(r['canonical_entities']) for r in rows)} for k,rows in [('vlabench',vla),('libero',external)]}})


def features(config,manifest,output,device):
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    from ..train import tensor_digest
    from .features import source_crops
    proof=read(manifest);payload=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    encoder=FrozenSigLIPEncoder(config['vision_model'],local_files_only=True,image_size=384).to(device).eval()
    state={k[len('image_encoder.'):]:v for k,v in payload['controller'].items() if k.startswith('image_encoder.')}
    encoder.load_state_dict(state,strict=True);del payload,state
    encoder_sha=tensor_digest(encoder)
    freeze(output/'settings.json',{'checkpoint_sha256':sha(config['checkpoint']),'observation_manifest_sha256':sha(manifest),
        'encoder_sha256':encoder_sha,'source_sha256':sha(__file__),'feature_algorithm_sha256':sha(Path(__file__).with_name('features.py'))})
    records=[]
    rows=[(dataset,row) for dataset in ('vlabench','libero','rlbench') if dataset in proof for row in proof[dataset]]
    for i,(dataset,row) in enumerate(rows):
        if any(sha(p)!=h for p,h in row['files'].items()):raise ValueError('frozen image changed')
        path=output/dataset/(digest(row['episode_id'])[:24]+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('preserve unfinished feature attempt in a fresh directory')
        count=len(row['canonical_entities']);patches,owners=source_crops(row['raw_image'],row['binding']['markers'],count)
        values=torch.zeros(count,2,encoder.output_dim);counts=torch.zeros(count,2)
        for start in range(0,len(patches),16):
            with torch.no_grad():embeddings=encoder(torch.stack(patches[start:start+16]).to(device)).cpu()
            for (pointer,scale),embedding in zip(owners[start:start+16],embeddings):
                values[pointer,scale]+=embedding;counts[pointer,scale]+=1
        values=(values/counts.clamp_min(1)[...,None]).flatten(1);geometry=torch.zeros(count,4,3)
        for marker in row['binding']['markers']:
            x,y,w,h=marker['box'];geometry[marker['id'],marker['view']]=torch.tensor([((x+w/2)%480)/480,((y+h/2)%480)/480,1.])
        values=torch.cat([values,geometry.flatten(1)],-1)
        torch.save({'features':values.half(),'visible':counts.sum(-1)>0},path)
        records.append({'dataset':dataset,'episode_id':row['episode_id'],'path':str(path),'sha256':sha(path)})
        if (i+1)%25==0:print(f'[grounding-eval] features {i+1}/{len(rows)}',flush=True)
    freeze(output/'manifest.json',{'complete':True,'episodes':records,'encoder_sha256':encoder_sha,
        'observations_sha256':sha(manifest),'checkpoint_sha256':sha(config['checkpoint'])})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','features']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();c=read(a.config);c['configuration_path']=str(a.config)
    verify_configuration(c)
    if a.mode=='prepare':prepare(c,a.output)
    else:features(c,a.manifest,a.output,torch.device(a.device))


if __name__=='__main__':main()
