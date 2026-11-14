"""Prepare frozen public priors and fresh evaluation features before training."""
import argparse
from pathlib import Path
import sys
import torch
from PIL import Image
from ...protocol import read,freeze,sha,digest,episodes,public_input
from ...train import tensor_digest
from ..eval_observations import libero_markers,features as visual_features
from ..object_fixes.grounding import paired_features


def observation_manifest(rows):
    # Retained runtimes iterate both keys even for a LIBERO-only evaluation.
    return {'complete':True,'vlabench':[],'libero':rows,'references_read':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config)
    from .text_features import main as text_features
    root=Path(c['output']);rows=[]
    prior_path=root/'text_priors/manifest.json'
    if prior_path.exists():
        prior=read(prior_path)
        if prior['references_read'] or sha(prior['path'])!=prior['sha256'] or prior['observations_sha256']!=sha(c['observations']):
            raise ValueError('reused public text features changed')
        if prior['encoder_sha256']!=read(c['features'])['encoder_sha256']:raise ValueError('reused paired encoder changed')
        if any(sha(f)!=h for f,h in prior['public_instruction_files'].items()):raise ValueError('public source text changed')
        if any(sha(Path(c['vision_model'])/f)!=h for f,h in prior['tokenizer_files'].items()):raise ValueError('public tokenizer changed')
    else:
        sys.argv=[sys.argv[0],'--config',str(a.config)];text_features();torch.cuda.empty_cache()
    for e in episodes(read(c['external_manifest'])):
        public=public_input(e);raw,numbered=public['images'];proof=libero_markers(raw,numbered,len(public['entities']))
        if not proof['markers']:raise ValueError('fresh episode has no verified public numeric markers')
        folder=root/'fresh_observations'/digest(e['episode_id'])[:24];folder.mkdir(parents=True,exist_ok=True)
        path=folder/'crop_rgb.png'
        with Image.open(raw) as im:
            if im.width!=im.height:raise ValueError('square public camera required')
            factor=480/im.width;im.convert('RGB').resize((480,480),Image.Resampling.BILINEAR).save(path)
        markers=[{**m,'box':[round(x*factor) for x in m['box']]} for m in proof['markers']]
        rows.append({'episode_id':e['episode_id'],'canonical_entities':public['entities'],'raw_image':str(path),
            'binding':{**proof,'markers':markers},'files':{raw:sha(raw),numbered:sha(numbered),str(path):sha(path)}})
    manifest=root/'fresh_observations/manifest.json';freeze(manifest,observation_manifest(rows))
    visual_features(c,manifest,root/'fresh_features',torch.device('cuda:0'));torch.cuda.empty_cache()
    from transformers import AutoTokenizer
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    encoder=FrozenSigLIPEncoder(c['vision_model'],local_files_only=True,image_size=384).to('cuda:0').eval()
    payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
    encoder.load_state_dict({k[len('image_encoder.'):]:v for k,v in payload['controller'].items() if k.startswith('image_encoder.')},strict=True)
    del payload
    if tensor_digest(encoder)!=read(c['features'])['encoder_sha256']:raise ValueError('primary encoder differs')
    tokenizer=AutoTokenizer.from_pretrained(c['vision_model'],local_files_only=True);bindings={r['episode_id']:r for r in rows};predictions=[]
    for i,e in enumerate(episodes(read(c['external_manifest']))):
        predictions.append({'episode_id':e['episode_id'],**paired_features(encoder,tokenizer,public_input(e),bindings[e['episode_id']],'cuda:0')})
        if (i+1)%10==0:print(f'[binding-inputs] frozen fresh priors {i+1}/100',flush=True)
    freeze(root/'fresh_priors.json',{'episodes':predictions,'references_read':False,'manifest_sha256':sha(c['external_manifest']),
        'encoder_sha256':tensor_digest(encoder),'generation_method':'frozen step-3 RGB/phrase retrieval; no scoring'})
    freeze(root/'inputs_complete.json',{'passed':True,'fresh_manifest_sha256':sha(c['external_manifest']),
        'text_manifest_sha256':sha(root/'text_priors/manifest.json'),'fresh_observations_sha256':sha(manifest),
        'fresh_features_sha256':sha(root/'fresh_features/manifest.json'),'fresh_priors_sha256':sha(root/'fresh_priors.json')})


if __name__=='__main__':main()
