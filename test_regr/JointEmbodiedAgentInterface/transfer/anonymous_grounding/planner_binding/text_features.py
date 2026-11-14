"""Public source instructions only; never open output plans for grounding priors."""
import argparse
from pathlib import Path
import torch
from ...protocol import read,freeze,sha
from ...train import tensor_digest
from ..object_fixes.grounding import phrases


def main():
    from transformers import AutoTokenizer
    from test_regr.VLABenchAgentInterface.dataset import _read_instruction
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config)
    manifest=read(c['observations']);instructions={};files={}
    for record in manifest['episodes']:
        folder=Path(record['raw_image']).parent
        for f in folder.iterdir():
            if f.is_file() and 'instruction' in f.name.lower():files[str(f)]=sha(f)
        text=_read_instruction(folder);instructions[text]=phrases(text)
    if not instructions or not files:raise ValueError('public source instructions unavailable')
    model=FrozenSigLIPEncoder(c['vision_model'],local_files_only=True,image_size=384).to('cuda:0').eval()
    payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict({k[len('image_encoder.'):]:v for k,v in payload['controller'].items() if k.startswith('image_encoder.')},strict=True)
    del payload
    if tensor_digest(model)!=read(c['features'])['encoder_sha256']:raise ValueError('paired text/image encoder mismatch')
    tokenizer=AutoTokenizer.from_pretrained(c['vision_model'],local_files_only=True);rows={}
    texts=sorted(instructions)
    for start in range(0,len(texts),16):
        selected=texts[start:start+16];queries=['a photo of '+(instructions[t][r] or 'an unidentified object')+'.' for t in selected for r in ('source','destination')]
        tokens=tokenizer(queries,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
        with torch.no_grad():vectors=model.model.get_text_features(**{k:v.to('cuda:0') for k,v in tokens.items()})
        if not torch.is_tensor(vectors):vectors=getattr(vectors,'text_embeds',None) if getattr(vectors,'text_embeds',None) is not None else vectors.pooler_output
        for j,text in enumerate(selected):rows[text]={'phrases':instructions[text],'features':vectors[j*2:j*2+2].cpu().half()}
        if start%160==0:print(f'[binding-priors] {start+len(selected)}/{len(texts)}',flush=True)
    output=Path(c['output'])/'text_priors';output.mkdir(exist_ok=True);path=output/'features.pt'
    torch.save(rows,path)
    freeze(output/'manifest.json',{'path':str(path),'sha256':sha(path),'instructions':len(rows),'public_instruction_files':files,
        'encoder_sha256':tensor_digest(model),'observations_sha256':sha(c['observations']),'references_read':False,
        'tokenizer_files':{f.name:sha(f) for f in Path(c['vision_model']).iterdir() if f.name in ('tokenizer.json','spiece.model','tokenizer_config.json','special_tokens_map.json')}})


if __name__=='__main__':main()
