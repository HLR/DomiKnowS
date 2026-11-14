"""Freeze paired SigLIP retrieval without Qwen or any training/evaluation labels."""
import argparse
from pathlib import Path
import torch
from ...protocol import read,sha,freeze,episodes,public_input
from ..eval_observations import verify_configuration
from .grounding import paired_features
from .provenance import validate_inputs


def main():
    from transformers import AutoTokenizer
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    from ...train import tensor_digest
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();c=read(a.configuration);verify_configuration(c);validate_inputs(c)
    proof=read(c['observations']);bindings={r['episode_id']:r for r in proof['libero']}
    encoder=FrozenSigLIPEncoder(c['vision_model'],local_files_only=True,image_size=384).to(a.device).eval()
    payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
    state={k[len('image_encoder.'):]:v for k,v in payload['controller'].items() if k.startswith('image_encoder.')}
    encoder.load_state_dict(state,strict=True);del payload,state
    tokenizer=AutoTokenizer.from_pretrained(c['vision_model'],local_files_only=True)
    tokenizer_files={p.name:sha(p) for p in Path(c['vision_model']).iterdir() if p.name in (
        'tokenizer.json','spiece.model','tokenizer_config.json','special_tokens_map.json')}
    if len(tokenizer_files)!=4:raise ValueError('freeze complete tokenizer files')
    settings={'configuration_sha256':sha(a.configuration),'observations_sha256':sha(c['observations']),
        'encoder_sha256':tensor_digest(encoder),'tokenizer_files':tokenizer_files,'source_sha256':sha(Path(__file__).with_name('grounding.py')),
        'updates':0,'queries':'instruction phrases only','pixels':'original RGB; numeric public draw anchors; RGB GrabCut',
        'candidate_scoring':'mean cosine of region and 192px context; all visible entities; no object-type lookup'}
    if settings['encoder_sha256']!=read(c['features'])['encoder_sha256']:raise ValueError('paired and learned encoders differ')
    out=Path(c['output'])/'aligned';freeze(out/'settings.json',settings);rows=[]
    for i,episode in enumerate(episodes(read(c['libero_manifest']))):
        public=public_input(episode);binding=bindings[episode['episode_id']]
        if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('observation changed')
        row={'episode_id':episode['episode_id'],**paired_features(encoder,tokenizer,public,binding,a.device)}
        freeze(out/'episodes'/f'{i:04d}.json',row);rows.append(row)
        if (i+1)%10==0:print(f'[object-fixes] paired RGB/text features {i+1}/100',flush=True)
    freeze(out/'manifest.json',{'settings':settings,'episodes':rows,'complete':True,'references_read':False})


if __name__=='__main__':main()
