"""Conservative exact tag recognition; original RGB supplies every output pixel.

Incomplete or ambiguous old tags are omitted, not guessed. Old annotations and
ID-coded instance colors cannot survive because output starts from clean RGB.
"""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, find_objects

from ..protocol import read, sha


@lru_cache(maxsize=1)
def atlas():
    return read(Path(__file__).with_name('glyph_atlas.json'))


@lru_cache(maxsize=4)
def templates(count=64):
    digits=atlas()['digits'];result=[]
    for value in range(count):
        text=str(value)
        for advance in ([14] if len(text)==1 else [14,15]):
            for shift in ([0] if len(text)==1 else [0,1]):
                for width in ([27,28] if len(text)==1 else [41,42]):
                    stamp=np.zeros((27,width),dtype=np.uint8)
                    for index,digit in enumerate(text):
                        glyph=np.array([[c=='1' for c in row] for row in digits[digit]['rows']])
                        x=digits[digit]['offset_x']+shift+advance*index
                        if x+glyph.shape[1]>width:break
                        stamp[2:20,x:x+glyph.shape[1]]=glyph*255
                    else:
                        result.append((value,stamp,digits[text[0]]['offset_x']+shift))
    return result


def recognize(raw_path,mask_path,entity_count):
    raw=np.array(Image.open(raw_path).convert('RGB'));mask=np.array(Image.open(mask_path).convert('RGB'))
    if raw.shape!=mask.shape or raw.shape!=(960,960,3):raise ValueError('four matching 480-pixel source views required')
    white=np.all(mask==255,axis=-1);black=np.all(mask==0,axis=-1);binary=white|black
    components,n=label(white,np.ones((3,3)));glyphs=atlas()['digits'];locations=defaultdict(list)
    for index,region in enumerate(find_objects(components)):
        y,x=region
        if y.stop-y.start!=18 or not 4<=x.stop-x.start<=15:continue
        bitmap=components[region]==index+1
        for digit,setting in glyphs.items():
            expected=np.array([[c=='1' for c in row] for row in setting['rows']])
            if bitmap.shape==expected.shape and np.array_equal(bitmap,expected):locations[digit].append((x.start,y.start))
    matches={}
    for value,stamp,offset in templates():
        height,width=stamp.shape
        for gx,gy in locations[str(value)[0]]:
            x,y=gx-offset,gy-2
            if x<1 or y<1 or x+width>=960 or y+height>=960:continue
            if x//480!=(x+width-1)//480 or y//480!=(y+height-1)//480:continue
            patch=mask[y:y+height,x:x+width]
            if not np.array_equal(patch,np.repeat(stamp[...,None],3,axis=-1)):continue
            # A surviving digit of an occluded multi-digit tag is NOT a new
            # single-digit ID: both full word boundaries must be observable.
            if not np.any(~binary[y:y+height,x-1] & ~binary[y:y+height,x+width]):continue
            if np.array_equal(patch,raw[y:y+height,x:x+width]):continue
            ink=frozenset((int(px+x),int(py+y)) for py,px in zip(*np.where(stamp==255)))
            key=(value,ink)
            candidate={'id':value,'box':[x,y,width,height],'view':(y//480)*2+x//480}
            if key not in matches or tuple(candidate['box'])<tuple(matches[key]['box']):matches[key]=candidate
    # Colored foreground geometry can overwrite a tag's black background while
    # leaving a complete digit. Accept it only if no valid multi-digit ID could
    # contain that digit: an exposed '1' from '11' is deliberately NOT ID 1.
    for digit,points in locations.items():
        value=int(digit)
        if value>=entity_count or any(digit in str(other) for other in range(10,entity_count)):continue
        setting=glyphs[digit]
        glyph=np.array([[c=='1' for c in row] for row in setting['rows']])
        for gx,gy in points:
            x,y=gx-setting['offset_x'],gy-2;width=27
            if x<0 or y<0 or x+width>960 or y+27>960:continue
            if x//480!=(x+width-1)//480 or y//480!=(y+26)//480:continue
            # Require the characteristic opaque annotation backing, changed
            # from the clean RGB. Physical text copied from RGB is excluded.
            bg=black[y:y+27,x:x+width]
            if bg.mean()<.5:continue
            original_black=np.all(raw[y:y+27,x:x+width]==0,axis=-1)
            if np.mean(~original_black[bg])<.5:continue
            original_ink=raw[gy:gy+18,gx:gx+glyph.shape[1]][glyph]
            if np.mean(np.any(original_ink!=255,axis=-1))<.95:continue
            ink=frozenset((int(px+gx),int(py+gy)) for py,px in zip(*np.where(glyph)))
            key=(value,ink)
            matches.setdefault(key,{'id':value,'box':[x,y,width,27],
                                    'view':(y//480)*2+x//480,'background_occluded':True})
    # Prefer the complete observed number over any possible digit sub-stamps.
    keys=[k for k in matches if not any(k[1]<other[1] for other in matches)]
    groups=defaultdict(list)
    for key in keys:
        candidate=matches[key]
        if candidate['id']>=entity_count:raise ValueError('printed canonical ID exceeds the source entity table')
        groups[candidate['id'],candidate['view']].append(candidate)
    accepted=[];ambiguous=[]
    for key,values in sorted(groups.items()):
        if len(values)==1:accepted.append(values[0])
        else:ambiguous.append({'id':key[0],'view':key[1],'candidates':len(values)})
    return {'raw_sha256':sha(raw_path),'mask_sha256':sha(mask_path),'atlas_sha256':sha(Path(__file__).with_name('glyph_atlas.json')),
            'markers':accepted,'visible_ids':sorted({m['id'] for m in accepted}),
            'ambiguous_omitted':ambiguous,'protocol':'complete exact printed stamps; observable word boundaries; no reference lookup'}


def render(raw_path,markers,order,output):
    if sorted(order)!=list(range(len(order))):raise ValueError('bijective ID permutation required')
    inverse={old:new for new,old in enumerate(order)}
    with Image.open(raw_path) as raw:image=raw.convert('RGB').copy()
    draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=20)
    for marker in markers:
        old=marker['id'];x,y,w,h=marker['box'];cx,cy=x+w//2,y+h//2
        text=str(inverse[old]);bounds=draw.textbbox((0,0),text,font=font);tw,th=bounds[2]-bounds[0],bounds[3]-bounds[1]
        width,height=tw+8,th+8;vx,vy=(marker['view']%2)*480,(marker['view']//2)*480
        left=min(max(vx+1,cx-width//2),vx+479-width);top=min(max(vy+1,cy-height//2),vy+479-height)
        draw.rectangle((left,top,left+width,top+height),fill='white',outline='black')
        draw.text((left+4-bounds[0],top+4-bounds[1]),text,fill='black',font=font)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('preserve frozen rendered input')
    image.save(output)
    return sha(output)
