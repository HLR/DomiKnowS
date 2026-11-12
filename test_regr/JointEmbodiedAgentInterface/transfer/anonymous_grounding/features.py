"""Frozen SigLIP features from observation-only label-anchor crops."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ..protocol import read,freeze,sha,digest


def source_crops(raw_path,markers,count):
    groups=defaultdict(list)
    for marker in markers:groups[marker['id']].append(marker)
    patches=[];owners=[]
    with Image.open(raw_path) as raw:
        raw=raw.convert('RGB')
        for pointer,values in sorted(groups.items()):
            for marker in sorted(values,key=lambda x:x['view'])[:2]:
                x,y,w,h=marker['box'];cx,cy=x+w//2,y+h//2;vx,vy=(marker['view']%2)*480,(marker['view']//2)*480
                for scale,size in enumerate((64,192)):
                    box=(max(vx,cx-size//2),max(vy,cy-size//2),min(vx+480,cx+size//2),min(vy+480,cy+size//2))
                    patch=raw.crop(box).resize((384,384),Image.Resampling.BILINEAR)
                    patches.append(torch.from_numpy(np.asarray(patch).copy()).permute(2,0,1).float()/255.)
                    owners.append((pointer,scale))
    return patches,owners


def prepare_features(config,manifest_path,output,device):
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    from ..train import tensor_digest
    proof=read(manifest_path)
    if not proof['complete'] or proof['test_used']:raise ValueError('frozen source observations required')
    encoder=FrozenSigLIPEncoder(config['vision_model'],local_files_only=True,image_size=384).to(device).eval()
    # Use the exact frozen encoder saved with the selected initialization,
    # rather than whichever revision a model-name cache currently resolves to.
    initialization=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    state={key[len('image_encoder.'):]:value for key,value in initialization['controller'].items()
           if key.startswith('image_encoder.')}
    if not state:raise ValueError('initialization has no frozen visual encoder')
    encoder.load_state_dict(state,strict=True);del initialization,state
    encoder_sha256=tensor_digest(encoder)
    freeze(output/'settings.json',{'manifest_sha256':sha(manifest_path),'model':config['vision_model'],
        'encoder_sha256':encoder_sha256,
        'initialization_sha256':config['checkpoint_sha256'],
        'source_sha256':sha(__file__),'crop_sizes':[64,192],'views_per_entity':2,
        'order':'canonical entity order; no index or canonical-name feature','device':str(device)})
    records=[]
    for index,record in enumerate(proof['episodes']):
        folder=output/'episodes'/digest(record['episode_id'])[:24];path=folder/'features.pt'
        if path.exists():
            item=read(folder/'record.json')
            if sha(path)!=item['sha256'] or sha(record['raw_image'])!=record['files'][record['raw_image']]:
                raise ValueError('cached visual features or source RGB changed')
            records.append(item);continue
        if any(sha(p)!=h for p,h in record['files'].items()):raise ValueError('source image binding changed')
        count=len(record['canonical_entities']);patches,owners=source_crops(record['raw_image'],record['binding']['markers'],count)
        values=torch.zeros(count,2,encoder.output_dim);counts=torch.zeros(count,2)
        for start in range(0,len(patches),16):
            with torch.no_grad():embeddings=encoder(torch.stack(patches[start:start+16]).to(device)).cpu()
            for (pointer,scale),embedding in zip(owners[start:start+16],embeddings):
                values[pointer,scale]+=embedding;counts[pointer,scale]+=1
        values=values/counts.clamp_min(1)[...,None]
        visible=counts.sum(dim=-1)>0;values=values.flatten(1)
        geometry=torch.zeros(count,4,3)
        for marker in record['binding']['markers']:
            x,y,w,h=marker['box'];view=marker['view']
            geometry[marker['id'],view]=torch.tensor([((x+w/2)%480)/480,((y+h/2)%480)/480,1.])
        values=torch.cat([values,geometry.flatten(1)],dim=-1)
        folder.mkdir(parents=True);torch.save({'features':values.half(),'visible':visible},path)
        item={'episode_id':record['episode_id'],'path':str(path),'sha256':sha(path),
              'raw_sha256':record['files'][record['raw_image']],'visible_ids':visible.nonzero().flatten().tolist()}
        freeze(folder/'record.json',item);records.append(item)
        if (index+1)%25==0:print(f'[anonymous-grounding] visual features {index+1}/4050',flush=True)
    freeze(output/'manifest.json',{'episodes':records,'complete':True,'visual_dim':encoder.output_dim*2+12,
        'encoder_sha256':encoder_sha256,'source_manifest_sha256':sha(manifest_path)})
    del encoder;torch.cuda.empty_cache()
