"""Instruction phrases and paired SigLIP retrieval over observed RGB regions."""
import re
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def phrases(instruction):
    # Instruction grammar only: no benchmark task IDs, object lists or reference names.
    text=instruction.strip().rstrip('.');patterns=[
        r'^(?:please\s+)?(?:pick(?:\s+up)?|grab|take|select|choose)\s+(.+?)\s+and\s+(?:place|put|drop)\s+(?:it|them)\s+(?:into|onto|in|on)\s+(.+)$',
        r'^(?:please\s+)?(?:put|place|move)\s+(.+?)\s+(?:into|onto|in|on)\s+(.+)$']
    for pattern in patterns:
        match=re.match(pattern,text,re.I)
        if match:
            clean=lambda s:re.sub(r'^(?:the|an|a)\s+','',s.strip(),flags=re.I)
            return {'source':clean(match[1]),'destination':clean(match[2])}
    return {'source':None,'destination':None}


def region(raw,anchor,window=192):
    """RGB-only GrabCut around a PUBLIC numeric-tag draw anchor; deterministic fallback."""
    import cv2
    x,y=anchor;w,h=raw.size;half=window//2
    box=(max(0,x-half),max(0,y-half),min(w,x+half),min(h,y+half));array=np.array(raw.crop(box).convert('RGB'))
    px,py=x-box[0],y-box[1];hh,ww=array.shape[:2]
    mask=np.full((hh,ww),cv2.GC_PR_BGD,np.uint8);mask[[0,-1],:]=cv2.GC_BGD;mask[:,[0,-1]]=cv2.GC_BGD
    mask[max(1,py-2):min(hh-1,py+3),max(1,px-2):min(ww-1,px+3)]=cv2.GC_FGD
    cv2.setRNGSeed(42)
    try:
        cv2.grabCut(array,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
        fg=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD);ys,xs=np.where(fg)
        if len(xs)>=32 and fg.mean()<.75 and fg[py,px]:
            left,right,top,bottom=xs.min(),xs.max()+1,ys.min(),ys.max()+1
            pad=max(3,round(.1*max(right-left,bottom-top)))
            result=(max(0,box[0]+left-pad),max(0,box[1]+top-pad),min(w,box[0]+right+pad),min(h,box[1]+bottom+pad))
            return raw.crop(result),{'method':'rgb_grabcut','box':list(map(int,result)),'anchor':list(anchor)}
    except cv2.error:pass
    result=(max(0,x-32),max(0,y-32),min(w,x+32),min(h,y+32))
    return raw.crop(result),{'method':'public_anchor_fallback','box':list(result),'anchor':list(anchor)}


def rank(text_features,visual_features,visible):
    # All other observed candidates are negatives; record the hardest competitors.
    q=F.normalize(text_features.float(),dim=-1);v=F.normalize(visual_features.float(),dim=-1)
    scores=torch.einsum('rd,nsd->rns',q,v).mean(-1)
    scores[:,~visible]=float('-inf');predictions={};diagnostics={}
    for column,role in enumerate(('source','destination')):
        candidates=sorted((i for i in range(len(visible)) if bool(visible[i])),key=lambda i:(-float(scores[column,i]),i))
        predictions[role]=candidates[0] if candidates else None
        diagnostics[role]={'ranked_candidates':[{'id':i,'score':float(scores[column,i])} for i in candidates],
            'nearest_competitor_ids':candidates[1:4],
            'margin':float(scores[column,candidates[0]]-scores[column,candidates[1]]) if len(candidates)>1 else None}
    return predictions,diagnostics


def paired_features(encoder,tokenizer,public,observed,device):
    query=phrases(public['instruction']);count=len(public['entities']);visible=torch.zeros(count,dtype=torch.bool)
    features=torch.zeros(count,2,encoder.output_dim);regions=[];raw_path=public['images'][0]
    with Image.open(raw_path) as im:
        raw=im.convert('RGB').resize((480,480),Image.Resampling.BILINEAR)
        # LIBERO draws text at (object_x, object_y), rectangle starts three pixels earlier.
        factor=480/im.width
        for marker in observed['binding']['markers']:
            i=marker['id'];x,y,_,_=marker['box'];anchor=(round(x+3*factor),round(y+3*factor))
            roi,proof=region(raw,anchor);context=raw.crop((max(0,anchor[0]-96),max(0,anchor[1]-96),min(480,anchor[0]+96),min(480,anchor[1]+96)))
            patches=[roi,context];tensors=torch.stack([torch.from_numpy(np.asarray(p.resize((384,384),Image.Resampling.BILINEAR)).copy()).permute(2,0,1).float()/255 for p in patches])
            with torch.no_grad():features[i]=encoder(tensors.to(device)).cpu()
            visible[i]=True;regions.append({'id':i,**proof})
    texts=['a photo of '+(query[k] or 'an unidentified object')+'.' for k in ('source','destination')]
    tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    with torch.no_grad():out=encoder.model.get_text_features(**{k:v.to(device) for k,v in tokens.items()})
    if not torch.is_tensor(out):out=getattr(out,'text_embeds',None) if getattr(out,'text_embeds',None) is not None else out.pooler_output
    predicted,diagnostics=rank(out.cpu(),features,visible)
    for role in predicted:
        if query[role] is None:predicted[role]=None
    return {'predictions':predicted,'phrases':query,'diagnostics':diagnostics,'regions':regions,
            'text_inputs':texts,'token_ids':tokens['input_ids'].tolist(),'visible_ids':visible.nonzero().flatten().tolist()}
