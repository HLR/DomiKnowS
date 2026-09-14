"""Contrastive free generation of 3D-FORCE-style puzzles on scenes the released splits do not use.

Every question is emitted as a pair with identical structure:

* a positive whose variables are anchored on real, distinct objects, with
  descriptors and relations that hold for those anchors, and
* a negative made by one minimal edit that renders the formula unsatisfiable:
  flip one relation to its opposite on the same axis and perspective
  (``left_1`` -> ``right_1``, ``obj_front`` -> ``obj_behind``), or swap one
  descriptor value for another value of the same attribute.

Both members have the same number of variables, descriptor tokens, bare
variables and relations per perspective, so question structure alone predicts
the answer at chance.  The earlier unpaired sampler leaked the answer through
structure: longer formulas were mostly false and a structure-only lookup
reached 73.7% held-out, above the trained model.

No distractor or uniqueness constraints.  Answers and solutions are computed
exactly from scene geometry (100% agreement with the released labels).  Output
uses the released ``3DForcePuzzle.json`` layout plus ``pair_id`` and
``negative_edit``, readable by ``force3d_dataset.load_force3d``.

Example:
    python gen_force3d_free.py --out generated/free_pairs_all.json --num-scenes 5000 --per-scene 10 --seed 1
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

import force3d_dataset as F

SHARED_PREFIX = F.SHARED_PREFIX
COLORS = F.FORCE3D_ATTRIBUTE_CONCEPTS["color"]
SHAPES = F.FORCE3D_ATTRIBUTE_CONCEPTS["shape"]
AXES = [("left", "right"), ("front", "behind")]

SHAPE_TEXT = {
    "airliner": "airliner", "biplane": "biplane", "double": "double-decker bus", "fighter": "fighter jet",
    "horse": "horse", "minivan": "minivan car", "school": "school bus", "scooter": "scooter motorcycle",
    "sedan": "sedan car", "suv": "SUV", "tank": "tank", "truck": "pickup truck",
}
DIR_TEXT = {"left": "left of", "right": "right of", "front": "in front of", "behind": "behind"}

Relation = Tuple[str, int, int]  # (question relation name, variable index a, variable index b)


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
            boxes = [[round(float(x), 1) for x in (o.get("bbox_2d_clipped") or o["bbox_2d_pixels"])]
                     for o in bobjs]
    return scene, cameras, hashes, boxes


def descriptor_text(preds: Sequence[str]) -> str:
    if not preds:
        return "object"
    words = [p if p in COLORS else SHAPE_TEXT[p] for p in preds]
    if all(p in COLORS for p in preds):
        words.append("object")
    return " ".join(words)


def evaluate(scene, rel_matrix, rel_index, k, unaries, relations):
    """Up to two distinct assignments satisfying the program (exact semantics)."""
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
            return len(sols) >= 2
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


def sample_positive(scene, rel_matrix, rel_index, n_views: int, rng: random.Random, max_vars: int,
                    bare_prob: float):
    """Variables anchored on distinct objects; every descriptor and relation holds for the anchors."""
    objs = scene["objects"]
    n = len(objs)
    k = rng.randint(1, min(max_vars, n))
    anchors = rng.sample(range(n), k)
    unaries: List[List[str]] = []
    for a in anchors:
        if rng.random() < bare_prob:
            unaries.append([])
            continue
        o = objs[a]
        kind = rng.choice(["color", "shape", "both"])
        unaries.append([o["color"]] if kind == "color" else
                       [o["shape"]] if kind == "shape" else [o["color"], o["shape"]])
    n_rel = rng.randint(0, k - 1) if k >= 2 else 0
    pairs = list(itertools.permutations(range(k), 2))
    rng.shuffle(pairs)
    relations: List[Relation] = []
    for (i, j) in pairs:
        if len(relations) >= n_rel:
            break
        axis = rng.choice(AXES)
        if rng.random() < 0.5:
            names = [f"obj_{d}" for d in axis]
        else:
            cam = rng.randrange(n_views)
            names = [f"{d}_{cam}" for d in axis]
        holding = [nm for nm in names if rel_matrix[anchors[i] * n + anchors[j], rel_index[nm]] > 0.5]
        if holding:  # exactly one direction of an axis holds unless the objects are aligned
            relations.append((holding[0], i, j))
    return k, unaries, relations


def negative_twin(scene, rel_matrix, rel_index, k, unaries, relations, rng: random.Random):
    """One structure-preserving edit that makes the formula unsatisfiable, or None."""
    edits = [("relation", r) for r in range(len(relations))]
    edits += [("descriptor", v, t) for v in range(k) for t in range(len(unaries[v]))]
    rng.shuffle(edits)
    for edit in edits:
        new_unaries = [list(u) for u in unaries]
        new_relations = list(relations)
        if edit[0] == "relation":
            name, a, b = new_relations[edit[1]]
            new_relations[edit[1]] = (F.opposite_relation(name), a, b)
        else:
            v, t = edit[1], edit[2]
            token = new_unaries[v][t]
            pool = COLORS if token in COLORS else SHAPES
            new_unaries[v][t] = rng.choice([p for p in pool if p != token])
        if not evaluate(scene, rel_matrix, rel_index, k, new_unaries, new_relations):
            return new_unaries, new_relations, edit[0]
    return None


def program_string(k, unaries, relations):
    parts = []
    for v in range(k):
        parts += [f"{p}(x{v + 1})" for p in unaries[v]]
    for name, a, b in relations:
        parts.append(f"{name}(x{a + 1}, x{b + 1})")
    prog = " and ".join(parts) if parts else "True"
    for v in range(k, 0, -1):
        prog = f"exists(Object, lambda x{v}: {prog})" if v < k else f"exists(Object, lambda x{v}: {prog} )"
    return prog


def question_text(k, texts, relations):
    words = ["one", "two", "three", "four", "five", "six"]
    parts = [f"object {v + 1} is an {texts[v]}" if texts[v][0] in "aeiouAEIOUS" and texts[v] != "object"
             else f"object {v + 1} is a {texts[v]}" for v in range(k)]
    for name, a, b in relations:
        if name.startswith("obj_"):
            parts.append(f"object {a + 1} is {DIR_TEXT[name[4:]]} object {b + 1} from object {b + 1}'s perspective")
        else:
            d, cam = name.rsplit("_", 1)
            parts.append(f"object {a + 1} is {DIR_TEXT[d]} object {b + 1} from camera {cam} perspective")
    return (f"Can you find {words[k - 1]} object{'s' if k > 1 else ''} from the image such that: "
            + "; ".join(parts) + ".")


def make_question(scene_dir, hashes, boxes, k, unaries, relations, answer, solution, pair_id, edit):
    texts = [descriptor_text(u) for u in unaries]
    kinds = {"object" if name.startswith("obj_") else "camera" for name, _, _ in relations}
    perspective = None if not kinds else ("mixed" if len(kinds) > 1 else next(iter(kinds)))
    return {
        "image_index": int(scene_dir.split("_")[1]),
        "image_filename": [f"{SHARED_PREFIX}{scene_dir}/{h}/image.png" for h in hashes],
        "slot_dict": {**{f"OBJ{v + 1}": texts[v] for v in range(k)},
                      **{f"R{i}": [a + 1, b + 1, name] for i, (name, a, b) in enumerate(relations)}},
        "solution": {str(v + 1): solution[v] for v in range(k)} if solution else None,
        "question": question_text(k, texts, relations),
        "program": program_string(k, unaries, relations),
        "answer": answer,
        "solution_indexes": [solution[v] for v in range(k)] if solution else [],
        "relation_perspective": perspective,
        "bboxes": boxes,
        "n_vars": k,
        "n_relations": len(relations),
        "pair_id": pair_id,
        "negative_edit": edit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(F.FORCE3D_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-scenes", type=int, default=200)
    ap.add_argument("--per-scene", type=int, default=10, help="questions per scene (even: emitted as pairs)")
    ap.add_argument("--views", default="mixed", choices=["fixed_2", "fixed_3", "fixed_4", "mixed"])
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--bare-prob", type=float, default=0.1)
    ap.add_argument("--include-used", action="store_true", help="also use scenes of the released splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=80)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    used = used_scene_dirs(root) if not args.include_used else set()
    scenes = sorted(d for d in os.listdir(root / "multiview") if d.startswith("scene_") and d not in used)
    rng.shuffle(scenes)
    scenes = scenes[: args.num_scenes]
    rel_index = {n: i for i, n in enumerate(F.QUESTION_RELATIONS)}
    conv = F.RelationConvention()
    n_pairs = max(1, args.per_scene // 2)

    questions = []
    stats = Counter()
    for scene_dir in scenes:
        mode = args.views if args.views != "mixed" else rng.choice(["fixed_2", "fixed_3", "fixed_4"])
        try:
            scene, cameras, hashes, boxes = load_scene(root, scene_dir, mode)
        except (FileNotFoundError, KeyError):
            stats["scene_skipped"] += 1
            continue
        rel_matrix = F.compute_relation_labels(scene, cameras, F.QUESTION_RELATIONS, conv)
        made = tries = 0
        while made < n_pairs and tries < args.max_tries:
            tries += 1
            k, unaries, relations = sample_positive(scene, rel_matrix, rel_index, len(cameras), rng,
                                                    args.max_vars, args.bare_prob)
            if not any(unaries) and not relations:
                continue  # "find k objects" with no content
            solutions = evaluate(scene, rel_matrix, rel_index, k, unaries, relations)
            if not solutions:
                stats["positive_check_failed"] += 1  # cannot happen: the anchors satisfy it
                continue
            negative = negative_twin(scene, rel_matrix, rel_index, k, unaries, relations, rng)
            if negative is None:
                stats["no_structure_preserving_negative"] += 1
                continue
            neg_unaries, neg_relations, edit = negative
            pair_id = f"{scene_dir}_{mode}_p{made}"
            questions.append(make_question(scene_dir, hashes, boxes, k, unaries, relations, True,
                                           solutions[0], pair_id, None))
            questions.append(make_question(scene_dir, hashes, boxes, k, neg_unaries, neg_relations, False,
                                           None, pair_id, edit))
            made += 1
            stats[f"k={k}"] += 2
            stats[f"negative_by_{edit}"] += 1
    out = {"tasks": [{"subset": f"free_pairs_{args.views}_seed{args.seed}", "data": {"questions": questions}}]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(questions)} questions ({len(questions) // 2} pairs) from {len(scenes)} scenes -> {args.out}")
    print(dict(sorted(stats.items())))

    # Self-check: question structure must not predict the answer.
    light = [{"program_str": q["program"], "answer": q["answer"],
              "scene_dir": q["image_filename"][0].split("/")[-3]} for q in questions]
    train, test = F.split_by_scene(light, max(1, len({s["scene_dir"] for s in light}) // 10), seed=0)
    struct_acc, majority_acc, n_eval = F.structure_only_baseline(train, test)
    print(f"structure-only baseline on 10% held-out scenes: {struct_acc:.1f}% "
          f"(majority class {majority_acc:.1f}%, n={n_eval})")


if __name__ == "__main__":
    main()
