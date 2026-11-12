"""Exact reconstruction gates for observation-only image-marker permutations."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .protocol import sha


def render(raw, centers, *, style, permutation=None):
    image = raw.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    if style not in ('libero-v3', 'rlbench-v33'):
        raise ValueError('unsupported annotation style; no guessed marker mappings')
    seen = set()
    for old, x, y in centers:
        if any(type(value) is not int for value in (old, x, y)) or old < 0:
            raise ValueError('integer canonical marker bindings required')
        if not 0 <= x < image.width or not 0 <= y < image.height:
            raise ValueError('marker center is outside its observed image')
        if (old, x, y) in seen:
            raise ValueError('duplicate marker binding')
        seen.add((old, x, y))
        value = old if permutation is None else permutation.encode_pointer(old)
        if style == 'libero-v3':
            draw.rectangle([x-3, y-3, x+14, y+13], fill='white', outline='black')
        else:
            draw.rectangle((x-2, y-2, x+16, y+14), fill='white', outline='black')
        draw.text((x, y), str(value), fill='black')
    return image


def verify(raw_path, numbered_path, centers, *, style, entity_count):
    if any(not 0 <= row[0] < entity_count for row in centers):
        raise ValueError('marker ID outside canonical inventory')
    with Image.open(raw_path) as raw, Image.open(numbered_path) as numbered:
        predicted = np.asarray(render(raw, centers, style=style))
        original = np.asarray(numbered.convert('RGB'))
        if predicted.shape != original.shape or not np.array_equal(predicted, original):
            count = int(np.count_nonzero(np.any(predicted != original, axis=-1))) if predicted.shape == original.shape else -1
            raise ValueError(f'canonical annotation reconstruction differs ({count} pixels); stop ID perturbation')
        return {'passed': True, 'style': style, 'entity_count': entity_count,
                'centers': centers, 'raw_sha256': sha(raw_path), 'numbered_sha256': sha(numbered_path),
                'decoded_numbered_sha256': hashlib.sha256(original.tobytes()).hexdigest()}


def recover_centers(raw_path, numbered_path, projected, *, style, entity_count, radius=8):
    """Locate the original public marker stamps, retaining exact pixel verification.

    Match public numeric stamps exactly. Prefer local projected locations; a
    unique complete stamp elsewhere in the same observation is also sufficient.
    Restored segmentation centers can differ from frozen annotation centers.
    Ambiguous or incomplete global matches are refused. No task reference,
    hidden target, or approximate image acceptance is used.
    """
    if radius < 0 or radius > 8:
        raise ValueError('marker recovery radius must be between 0 and 8 pixels')
    with Image.open(numbered_path) as image:
        original = np.asarray(image.convert('RGB'))
    height, width = original.shape[:2]
    occupied = np.zeros((height, width), dtype=bool)
    resolved = []; pending = list(projected); hidden = []; global_anchors = []
    left, top, right, bottom = (3, 3, 14, 13) if style == 'libero-v3' else (2, 2, 16, 14)
    probe = ImageDraw.Draw(Image.new('RGB', (left+right+1, top+bottom+1)))
    for pointer in range(entity_count):
        x0, y0, x1, y1 = probe.textbbox((left, top), str(pointer))
        if x0 < 0 or y0 < 0 or x1 > left+right+1 or y1 > top+bottom+1:
            raise ValueError('marker glyph exceeds its opaque rectangle; cannot prove occlusion equivalence')
    # Native dictionary traversal need not preserve the old annotation's paint
    # order. Recover upper stamps first from pixels, not current traversal order.
    while pending:
        choices = []; occluded = []
        for old, px, py in pending:
            stamp = Image.new('RGB', (left+right+1, top+bottom+1))
            stamp = np.asarray(render(stamp, [(old, left, top)], style=style))
            candidates = []; covered = []
            for y in range(max(0, py-radius), min(height, py+radius+1)):
                for x in range(max(0, px-radius), min(width, px+radius+1)):
                    x0, y0, x1, y1 = max(0, x-left), max(0, y-top), min(width, x+right+1), min(height, y+bottom+1)
                    expected = stamp[y0-(y-top):y1-(y-top), x0-(x-left):x1-(x-left)]
                    visible = ~occupied[y0:y1, x0:x1]
                    if not visible.any(): covered.append((old, x, y)); continue
                    if np.array_equal(expected[visible], original[y0:y1, x0:x1][visible]):
                        candidates.append((old, x, y, int(visible.sum())))
            if len(candidates) == 1:
                choices.append(candidates[0])
            else:
                # The printed canonical ID is itself a public observation. A
                # unique FULL stamp is independent of projection drift; never
                # infer a global position from a partially occluded glyph.
                sh, sw = stamp.shape[:2]
                possible = np.ones((height-sh+1, width-sw+1), dtype=bool)
                for sy, sx in ((0, 0), (0, sw-1), (sh-1, 0), (sh-1, sw-1), (1, 1), (sh-2, sw-2)):
                    possible &= np.all(original[sy:sy+height-sh+1, sx:sx+width-sw+1] == stamp[sy, sx], axis=-1)
                ys, xs = np.where(possible)
                exact = [(old, int(x)+left, int(y)+top, sh*sw) for y, x in zip(ys, xs)
                         if np.array_equal(original[y:y+sh, x:x+sw], stamp)]
                if len(exact) == 1:
                    choices.append(exact[0]); global_anchors.append(old)
                    continue
            if not candidates and covered and not any(c[0] == old for c in choices):
                # Entire fixed-size marker rectangles are overwritten by already
                # recovered opaque stamps. All these positions are output-
                # equivalent for every ID, not approximate visible matches.
                occluded.append((min(covered, key=lambda c:(c[1]-px)**2+(c[2]-py)**2), old))
        if choices:
            choice = max(choices, key=lambda c:(c[3], -c[0])); marker = choice[:3]
        elif occluded:
            marker, old = min(occluded, key=lambda pair:pair[1]); hidden.append(old)
        else:
            raise ValueError(f'public markers {[p[0] for p in pending]} have no unique exact locations; no guessed binding')
        pending = [p for p in pending if p[0] != marker[0]]
        resolved.append(marker)
        _, x, y = marker
        occupied[max(0, y-top):min(height, y+bottom+1), max(0, x-left):min(width, x+right+1)] = True
    resolved.reverse()
    proof = verify(raw_path, numbered_path, resolved, style=style, entity_count=entity_count)
    return {**proof, 'projected_centers': projected, 'recovery_radius_px': radius,
            'fully_occluded_markers': hidden,
            'global_full_stamp_markers': sorted(set(global_anchors)),
            'anchor_source': 'exact public annotation stamps; local projection or unique complete observed stamp',
            'paint_order_source': 'verified public pixels, not native dictionary iteration'}


def write_counterfactual(raw_path, numbered_path, centers, *, style, permutation, output):
    proof = verify(raw_path, numbered_path, centers, style=style, entity_count=len(permutation.order))
    output = Path(output)
    if output.exists():
        raise ValueError('preserve existing counterfactual; use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw_path) as raw:
        render(raw, centers, style=style, permutation=permutation).save(output)
    return {**proof, 'new_to_old': list(permutation.order), 'old_to_new': list(permutation.old_to_new),
            'counterfactual_sha256': sha(output)}
