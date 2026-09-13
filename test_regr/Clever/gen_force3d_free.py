"""Free generation of 3D-FORCE-style puzzles on scenes the released splits do not use.

No distractor or uniqueness constraints: programs are sampled at random and
their answers computed exactly from the scene geometry (object positions,
headings and ring cameras), the same oracle that reproduces 100% of the
released labels.  Output is the released ``3DForcePuzzle.json`` layout so
``force3d_dataset.load_force3d`` reads it unchanged.

Example:
    python gen_force3d_free.py --out free_puzzles.json --num-scenes 200 --per-scene 10 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import force3d_dataset as F

SHARED_PREFIX = F.SHARED_PREFIX
COLORS = F.FORCE3D_ATTRIBUTE_CONCEPTS["color"]
SHAPES = F.FORCE3D_ATTRIBUTE_CONCEPTS["shape"]
DIRECTIONS = list(F.DIRECTIONS)

SHAPE_TEXT = {
    "airliner": "airliner", "biplane": "biplane", "double": "double-decker bus", "fighter": "fighter jet",
    "horse": "horse", "minivan": "minivan car", "school": "school bus", "scooter": "scooter motorcycle",
    "sedan": "sedan car", "suv": "SUV", "tank": "tank", "truck": "pickup truck",
}
DIR_TEXT = {"left": "left of", "right": "right of", "front": "in front of", "behind": "behind"}


def used_scene_dirs(root: Path) -> set:
    used = set()
    for name in ("3DForcePuzzle.json", "3DForceRef.json"):
        path = root / name
        if not path.exists():
            continue
        for task in json.load(open(path))["tasks"]:
            for q in task["data"]["questions"]:
                used.add(q["image_filename"][0].split("/")[-3])
    return used


def load_scene(root: Path, scene_dir: str, mode: str):
    d = root / "multiview" / scene_dir
    scene = json.load(open(d / "scene.json"))
    views = json.load(open(d / "views.json"))[mode]
    views = sorted(views, key=lambda v: v["view_index"])
    cameras = [json.load(open(d / v["hash"] / "camera.json")) for v in views]
    hashes = [v["hash"] for v in views]
    bfile = d / hashes[0] / "bboxes.json"
    boxes = None
    if bfile.exists():
        bobjs = sorted(json.load(open(bfile))["objects"], key=lambda o: o["slot"])
        if len(bobjs) == len(scene["objects"]):
            boxes = [[round(float(x), 1) for x in o["bbox_2d_pixels"]] for o in bobjs]
    return scene, cameras, hashes, boxes


def descriptor(obj: dict, rng: random.Random, bare_prob: float) -> Tuple[List[str], str]:
    """Unary predicates for a variable, sampled to be TRUE of ``obj`` (so
    positives exist) — negatives arise from relations and from swapping."""
    r = rng.random()
    if r < bare_prob:
        return [], "object"
    kind = rng.choice(["color", "shape", "both"])
    if kind == "color":
        return [obj["color"]], f"{obj['color']} object"
    if kind == "shape":
        return [obj["shape"]], SHAPE_TEXT[obj["shape"]]
    return [obj["color"], obj["shape"]], f"{obj['color']} {SHAPE_TEXT[obj['shape']]}"


def sample_program(scene, n_views: int, rng: random.Random, max_vars: int, bare_prob: float,
                   perturb_prob: float):
    """Sample k variables anchored on k distinct objects, plus random relations.

    Unaries are drawn true of their anchor object; then with ``perturb_prob``
    one unary is replaced by a random value, so negatives are not only
    relation failures.  Truth is decided afterwards by exact evaluation.
    """
    objs = scene["objects"]
    n = len(objs)
    k = rng.randint(1, min(max_vars, n))
    anchors = rng.sample(range(n), k)
    unaries: List[List[str]] = []
    texts: List[str] = []
    for a in anchors:
        preds, text = descriptor(objs[a], rng, bare_prob)
        unaries.append(preds)
        texts.append(text)
    if k > 0 and rng.random() < perturb_prob:
        i = rng.randrange(k)
        if unaries[i]:
            j = rng.randrange(len(unaries[i]))
            pool = COLORS if unaries[i][j] in COLORS else SHAPES
            unaries[i][j] = rng.choice(pool)
            texts[i] = " ".join(p if p in COLORS else SHAPE_TEXT[p] for p in unaries[i]) or "object"
            if len(unaries[i]) == 1 and unaries[i][0] in COLORS:
                texts[i] += " object"
    # relations: 0..k-1 edges among distinct pairs, random perspective
    n_rel = rng.randint(0, max(0, k - 1)) if k >= 2 else 0
    pairs = list(itertools.permutations(range(k), 2))
    rng.shuffle(pairs)
    relations = []
    persp = set()
    for (i, j) in pairs[:n_rel]:
        d = rng.choice(DIRECTIONS)
        if rng.random() < 0.5:
            name, kind = f"obj_{d}", "object"
        else:
            cam = rng.randrange(n_views)
            name, kind = f"{d}_{cam}", "camera"
        relations.append((name, i, j))
        persp.add(kind)
    perspective = None if not persp else ("mixed" if len(persp) > 1 else persp.pop())
    return k, unaries, texts, relations, perspective


def evaluate(scene, rel_matrix, rel_index, k, unaries, relations):
    """All distinct assignments satisfying the program (exact semantics)."""
    objs = scene["objects"]
    n = len(objs)

    def sat(o, preds):
        return all(o["color"] == p or o["shape"] == p for p in preds)

    cands = [[i for i in range(n) if sat(objs[i], unaries[v])] for v in range(k)]
    sols = []

    def ok(assign):
        for name, a, b in relations:
            if a in assign and b in assign and rel_matrix[assign[a] * n + assign[b], rel_index[name]] < 0.5:
                return False
        return True

    def bt(v, assign):
        if v == k:
            sols.append(dict(assign))
            return len(sols) >= 2  # we only need to know: none / at least one
        for i in cands[v]:
            if i in assign.values():
                continue
            assign[v] = i
            if ok(assign) and bt(v + 1, assign):
                return True
            del assign[v]
        return False

    bt(0, {})
    return sols


def program_string(k, unaries, relations):
    parts = []
    for v in range(k):
        parts += [f"{p}(x{v + 1})" for p in unaries[v]]
    for name, a, b in relations:
        parts.append(f"{name}(x{a + 1}, x{b + 1})")
    body = " and ".join(parts) if parts else "True"
    prog = body
    for v in range(k, 0, -1):
        prog = f"exists(Object, lambda x{v}: {prog})" if v < k else f"exists(Object, lambda x{v}: {prog} )"
    return prog


def question_text(k, texts, relations):
    words = ["one", "two", "three", "four", "five", "six"]
    parts = [f"object {v + 1} is a {texts[v]}" if not texts[v].startswith(("a", "e", "i", "o", "u", "S")) or texts[v] == "object"
             else f"object {v + 1} is an {texts[v]}" for v in range(k)]
    for name, a, b in relations:
        if name.startswith("obj_"):
            d = name[4:]
            parts.append(f"object {a + 1} is {DIR_TEXT[d]} object {b + 1} from object {b + 1}'s perspective")
        else:
            d, cam = name.rsplit("_", 1)
            parts.append(f"object {a + 1} is {DIR_TEXT[d]} object {b + 1} from camera {cam} perspective")
    return f"Can you find {words[k - 1]} object{'s' if k > 1 else ''} from the image such that: " + "; ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(F.FORCE3D_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-scenes", type=int, default=200)
    ap.add_argument("--per-scene", type=int, default=10)
    ap.add_argument("--views", default="mixed", choices=["fixed_2", "fixed_3", "fixed_4", "mixed"])
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--bare-prob", type=float, default=0.1)
    ap.add_argument("--perturb-prob", type=float, default=0.3)
    ap.add_argument("--balance", type=float, default=0.5, help="target share of answer=True")
    ap.add_argument("--include-used", action="store_true", help="also use scenes of the released splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=60)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    used = used_scene_dirs(root) if not args.include_used else set()
    scenes = sorted(d for d in os.listdir(root / "multiview") if d.startswith("scene_") and d not in used)
    rng.shuffle(scenes)
    scenes = scenes[: args.num_scenes]
    rel_index = {n: i for i, n in enumerate(F.FORCE3D_RELATIONS)}
    conv = F.RelationConvention()

    questions = []
    stats = Counter()
    for scene_dir in scenes:
        mode = args.views if args.views != "mixed" else rng.choice(["fixed_2", "fixed_3", "fixed_4"])
        try:
            scene, cameras, hashes, boxes = load_scene(root, scene_dir, mode)
        except (FileNotFoundError, KeyError):
            stats["scene_skipped"] += 1
            continue
        rel_matrix = F.compute_relation_labels(scene, cameras, F.FORCE3D_RELATIONS, conv)
        n_views = len(cameras)
        got_true = got_false = 0
        want_true = round(args.per_scene * args.balance)
        tries = 0
        while got_true + got_false < args.per_scene and tries < args.max_tries:
            tries += 1
            k, unaries, texts, relations, persp = sample_program(scene, n_views, rng, args.max_vars,
                                                                  args.bare_prob, args.perturb_prob)
            if not any(unaries) and not relations:
                continue  # trivially true "find k objects": no content to learn
            sols = evaluate(scene, rel_matrix, rel_index, k, unaries, relations)
            answer = bool(sols)
            if answer and got_true >= want_true:
                continue
            if not answer and got_false >= args.per_scene - want_true:
                continue
            got_true += answer
            got_false += not answer
            sol = sols[0] if answer else None
            questions.append({
                "image_index": int(scene_dir.split("_")[1]),
                "image_filename": [f"{SHARED_PREFIX}{scene_dir}/{h}/image.png" for h in hashes],
                "slot_dict": {**{f"OBJ{v + 1}": texts[v] for v in range(k)},
                              **{f"R{i}": [a + 1, b + 1, name] for i, (name, a, b) in enumerate(relations)}},
                "solution": {str(v + 1): sol[v] for v in range(k)} if sol else None,
                "question": question_text(k, texts, relations),
                "program": program_string(k, unaries, relations),
                "answer": answer,
                "solution_indexes": [sol[v] for v in range(k)] if sol else [],
                "relation_perspective": persp,
                "bboxes": boxes,
                "n_vars": k,
                "n_relations": len(relations),
            })
            stats[f"k={k}"] += 1
            stats["true" if answer else "false"] += 1
    out = {"tasks": [{"subset": f"free_{args.views}_seed{args.seed}", "data": {"questions": questions}}]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(questions)} questions from {len(scenes)} scenes -> {args.out}")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
