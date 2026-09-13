"""3D-FORCE adapter for the Clever/DomiKnowS pipeline.

Loads the 3D-FORCE Puzzle (yes/no ``exists`` programs) and REF (object
selection ``point``/``iota`` programs) splits into the sample format consumed
by ``main.py``'s sensors, translates the lambda-style programs into DomiKnowS
executable logic strings, and derives oracle spatial-relation labels from the
3D scene geometry.

Sample contract (what ``main.py`` reads):
    pil_image, image_filename, image_index, objects_raw (float32 n x 4 in the
    transformed image), all_objects (list of dicts with color/shape/...),
    logic_str, answer, question_raw, program (list, for curriculum sizing),
    relation_spatial_relation (float32 (n*n, R), oracle mode only) and, for
    REF, a precomputed one-hot ``logic_label``.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

FORCE3D_ROOT = Path("/localscratch/kamalida/projects/SaPy/datasets/3D-FORCE")
SHARED_PREFIX = "/egr/research-hlr2/shared/data/Spatial457/multiview/"

DIRECTIONS = ("left", "right", "front", "behind")
N_CAMERAS = 4

# Vocabulary actually used by the 3D-FORCE programs (verified by scanning
# every scene.json and every program): 7 colors and 12 shape classes.  Size
# and material exist in scene.json but no program refers to them.
FORCE3D_ATTRIBUTE_CONCEPTS: Dict[str, List[str]] = {
    "color": ["blue", "brown", "gray", "green", "purple", "red", "yellow"],
    "shape": [
        "airliner", "biplane", "double", "fighter", "horse", "minivan",
        "school", "scooter", "sedan", "suv", "tank", "truck",
    ],
}

# Relation vocabulary.  Plain names are aliases of the chosen view's camera
# (camera 0 in single-view mode) so graph.py's opposite/inverse constraints
# still attach to them.  ``obj_*`` are anchor-object perspective relations and
# ``*_k`` are camera-k perspective relations (k indexes the question's
# image_filename list).
# ``distinct`` is a fixed (never learned) identity relation: distinct(i, j)
# iff i != j.  The puzzle/REF semantics require every logical variable to
# bind a different object, which the existential reading of andL does not
# enforce on its own; the translator adds distinct('x', 'y') for every pair
# of variables.  main.py wires it through OracleDummyLearner in all modes.
DISTINCT = "distinct"
FORCE3D_RELATIONS: List[str] = (
    list(DIRECTIONS)
    + [f"obj_{d}" for d in DIRECTIONS]
    + [f"{d}_{k}" for k in range(N_CAMERAS) for d in DIRECTIONS]
    + [DISTINCT]
)
FORCE3D_RELATIONAL_CONCEPTS: Dict[str, List[str]] = {
    "spatial_relation": FORCE3D_RELATIONS,
}

_OPPOSITE = {"left": "right", "right": "left", "front": "behind", "behind": "front"}


def opposite_relation(name: str) -> str:
    """left_2 -> right_2, obj_front -> obj_behind, left -> right, distinct -> distinct."""
    if name == DISTINCT:
        return DISTINCT
    for d in DIRECTIONS:
        if name == d:
            return _OPPOSITE[d]
        if name.startswith(f"obj_{d}"):
            return f"obj_{_OPPOSITE[d]}"
        if name.startswith(f"{d}_"):
            return f"{_OPPOSITE[d]}_{name[len(d) + 1:]}"
    raise ValueError(f"Not a 3D-FORCE spatial relation: {name}")


# ---------------------------------------------------------------------------
# Program translation
# ---------------------------------------------------------------------------

_UNARY = set(v for vals in FORCE3D_ATTRIBUTE_CONCEPTS.values() for v in vals)
_BINARY = set(FORCE3D_RELATIONS)
_QUANTIFIERS = {"exists", "point", "iota"}


def _var_letter(index: int) -> str:
    if index >= 26:
        raise ValueError("more than 26 logical variables in one program")
    return chr(ord("a") + index)


class _Translator:
    """AST walker turning 3D-FORCE lambda programs into DomiKnowS logic."""

    def __init__(self, camera_alias: Optional[int] = 0):
        self.camera_alias = camera_alias
        self.var_map: Dict[str, str] = {}
        self.unary: Dict[str, List[str]] = defaultdict(list)  # letter -> terms
        self.binary: List[str] = []
        self.order: List[str] = []  # letters in binding order
        self.top: Optional[str] = None  # 'exists' or 'point'

    # -- helpers ----------------------------------------------------------
    def _bind(self, name: str) -> str:
        if name in self.var_map:
            raise ValueError(f"variable {name} bound twice")
        letter = _var_letter(len(self.var_map))
        self.var_map[name] = letter
        self.order.append(letter)
        return letter

    def _rel_name(self, name: str) -> str:
        if name not in _BINARY:
            raise ValueError(f"unknown binary predicate {name!r}")
        if self.camera_alias is not None:
            suffix = f"_{self.camera_alias}"
            base = name[: -len(suffix)] if name.endswith(suffix) else None
            if base in DIRECTIONS:
                return base
        return name

    # -- walkers ----------------------------------------------------------
    def _quantifier(self, node: ast.Call) -> str:
        """Handle exists/point/iota(Object, lambda v: body); return letter."""
        fn = node.func.id
        if len(node.args) != 2 or not isinstance(node.args[1], ast.Lambda):
            raise ValueError(f"malformed {fn}(...) call")
        lam = node.args[1]
        if len(lam.args.args) != 1:
            raise ValueError(f"{fn} lambda must bind exactly one variable")
        letter = self._bind(lam.args.args[0].arg)
        self._body(lam.body, letter)
        return letter

    def _body(self, node: ast.AST, subject: str) -> None:
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ast.And):
                raise ValueError("only 'and' is supported in program bodies")
            for value in node.values:
                self._body(value, subject)
            return
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")
        fn = node.func.id
        if fn in _QUANTIFIERS:
            if fn == "iota":
                raise ValueError("bare iota(...) outside a relation argument")
            self._quantifier(node)  # nested exists inside exists
            return
        if len(node.args) == 1:
            if fn not in _UNARY:
                raise ValueError(f"unknown unary predicate {fn!r}")
            letter = self._arg(node.args[0])
            self.unary[letter].append(f"{fn}('{letter}')")
            return
        if len(node.args) == 2:
            rel = self._rel_name(fn)
            a = self._arg(node.args[0])
            b = self._arg(node.args[1])
            self.binary.append(f"{rel}('{a}', '{b}')")
            return
        raise ValueError(f"unsupported call arity for {fn}")

    def _arg(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id not in self.var_map:
                raise ValueError(f"unbound variable {node.id}")
            return self.var_map[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "iota":
            # Flatten "rel(x, iota(Object, lambda y: SUB))": bind y, recurse.
            return self._quantifier(node)
        raise ValueError(f"unsupported predicate argument: {ast.dump(node)[:80]}")

    # -- entry ------------------------------------------------------------
    def translate(self, program_str: str) -> str:
        tree = ast.parse(program_str.strip(), mode="eval").body
        if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name)
                and tree.func.id in ("exists", "point")):
            raise ValueError("program must start with exists(...) or point(...)")
        self.top = tree.func.id
        self._quantifier(tree)

        terms: List[str] = []
        for letter in self.order:
            if self.unary[letter]:
                terms.extend(self.unary[letter])
            else:
                terms.append(f"obj('{letter}')")  # Clever convention
        terms.extend(self.binary)
        for i, x in enumerate(self.order):
            for y in self.order[i + 1:]:
                terms.append(f"{DISTINCT}('{x}', '{y}')")
        inner = terms[0] if len(terms) == 1 else f"andL({', '.join(terms)})"
        if self.top == "exists":
            return f"existsL({inner})"
        return f"miotaL({inner}, threshold=0.5, hard=False)"


def translate_force3d_program(program_str: str, camera_alias: Optional[int] = 0) -> str:
    """Translate a 3D-FORCE program string into a DomiKnowS logic string.

    Puzzle (``exists``) programs become ``existsL(andL(...))``; REF (``point``)
    programs become ``miotaL(andL(...), threshold=0.5, hard=False)`` whose
    first variable ``'a'`` is the selected object.  Nested ``iota`` anchors are
    flattened into extra existential variables.  ``camera_alias`` maps
    ``<dir>_<k>`` for that camera index onto the plain ``<dir>`` relation.
    """
    return _Translator(camera_alias).translate(program_str)


def program_predicates(program_str: str) -> List[str]:
    """Predicate tokens of a program (used as ``program`` for curriculum sizing)."""
    return [p for p in re.findall(r"([a-z_]+\d*)\(", program_str) if p not in _QUANTIFIERS]


# ---------------------------------------------------------------------------
# Geometry oracle
# ---------------------------------------------------------------------------

def _rotate_xy(vec: Sequence[float], angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float64)


class RelationConvention:
    """Tunable sign/offset conventions, calibrated against Puzzle solutions."""

    # Defaults are the values found by ``calibrate_convention`` on the Puzzle
    # split (object relations 469/469, camera relations 970/970 agreements).
    def __init__(self, camera_sign: int = 1, object_heading_offset_deg: float = 90.0,
                 object_lr_sign: int = 1, object_fb_sign: int = 1):
        self.camera_sign = camera_sign
        self.object_heading_offset_deg = object_heading_offset_deg
        self.object_lr_sign = object_lr_sign
        self.object_fb_sign = object_fb_sign

    def __repr__(self):
        return (f"RelationConvention(camera_sign={self.camera_sign}, "
                f"heading_offset={self.object_heading_offset_deg}, "
                f"lr_sign={self.object_lr_sign}, fb_sign={self.object_fb_sign})")


def camera_direction_vectors(scene: dict, camera: dict) -> Dict[str, np.ndarray]:
    """CLEVR-style ground-plane direction vectors for one ring camera.

    Derived from the camera location and the ring target (validated at 100 %
    against the Puzzle solutions): ``front`` points from the target toward the
    camera (an object in front is closer to the camera), ``behind`` is the
    opposite, ``right`` is the camera's right-hand direction on the ground
    plane and ``left`` its opposite.  ``scene.json['directions']`` (hero frame)
    does *not* reproduce the dataset's labels and is intentionally unused.
    """
    target = np.array(scene["camera_ring"]["target"][:2], dtype=np.float64)
    loc = np.array(camera["location"][:2], dtype=np.float64)
    fwd = target - loc
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0]])
    return {"front": -fwd, "behind": fwd, "left": -right, "right": right}


def relation_holds(name: str, i: int, j: int, coords: np.ndarray, rotations: Sequence[float],
                   camera_dirs: Dict[int, Dict[str, np.ndarray]],
                   conv: RelationConvention) -> bool:
    """Does ``name(i, j)`` hold: is object i <name> of object j?"""
    if name == DISTINCT:
        return i != j
    d = coords[i, :2] - coords[j, :2]
    if name.startswith("obj_"):
        direction = name[4:]
        theta = math.radians(float(rotations[j]) + conv.object_heading_offset_deg)
        heading = np.array([math.cos(theta), math.sin(theta)])
        left = np.array([-heading[1], heading[0]])
        if direction in ("front", "behind"):
            val = float(np.dot(d, heading)) * conv.object_fb_sign
            return val > 0 if direction == "front" else val < 0
        val = float(np.dot(d, left)) * conv.object_lr_sign
        return val > 0 if direction == "left" else val < 0
    if name in DIRECTIONS:
        direction, cam = name, 0
    else:
        direction, cam_s = name.rsplit("_", 1)
        cam = int(cam_s)
    if cam not in camera_dirs:
        return False  # relation refers to a view this question does not have
    val = float(np.dot(d, camera_dirs[cam][direction])) * conv.camera_sign
    return val > 0


def compute_relation_labels(scene: dict, cameras: Sequence[dict], relations: Sequence[str],
                            conv: RelationConvention) -> np.ndarray:
    """(n*n, len(relations)) float32 matrix, row i*n+j == relation(i, j)."""
    objs = scene["objects"]
    n = len(objs)
    coords = np.array([o["3d_coords"] for o in objs], dtype=np.float64)
    rotations = [o.get("rotation", 0.0) for o in objs]
    camera_dirs = {k: camera_direction_vectors(scene, cam)
                   for k, cam in enumerate(cameras)}
    out = np.zeros((n * n, len(relations)), dtype=np.float32)
    for r_idx, name in enumerate(relations):
        for i in range(n):
            for j in range(n):
                if i != j and relation_holds(name, i, j, coords, rotations, camera_dirs, conv):
                    out[i * n + j, r_idx] = 1.0
    return out


def _binary_terms(program_str: str) -> List[Tuple[str, str, str]]:
    """(relation, var_a, var_b) for every binary predicate in a puzzle program."""
    return [(m.group(1), m.group(2), m.group(3))
            for m in re.finditer(r"\b([a-z_]+\d*)\(x(\d+), x(\d+)\)", program_str)]


def validate_oracle_against_solutions(questions: Sequence[dict], scenes: Dict[str, dict],
                                      cameras_of: Dict[int, Sequence[dict]],
                                      conv: RelationConvention) -> Dict[str, Tuple[int, int]]:
    """Per relation family: (agreements, total) over True puzzles' solutions."""
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for qi, q in enumerate(questions):
        if not q.get("answer") or not q.get("solution"):
            continue
        scene = scenes[q["_scene_dir"]]
        objs = scene["objects"]
        coords = np.array([o["3d_coords"] for o in objs], dtype=np.float64)
        rotations = [o.get("rotation", 0.0) for o in objs]
        camera_dirs = {k: camera_direction_vectors(scene, cam)
                       for k, cam in enumerate(cameras_of[qi])}
        for rel, va, vb in _binary_terms(q["program"]):
            i, j = q["solution"][va], q["solution"][vb]
            family = "obj" if rel.startswith("obj_") else "cam"
            ok = relation_holds(rel, i, j, coords, rotations, camera_dirs, conv)
            stats[family][1] += 1
            stats[family][0] += int(ok)
    return {k: (v[0], v[1]) for k, v in stats.items()}


def calibrate_convention(questions, scenes, cameras_of, verbose=True) -> RelationConvention:
    """Search sign/offset conventions and keep the one agreeing with solutions."""
    best, best_score = None, -1.0
    results = []
    for cam_sign in (1, -1):
        for offset in (0.0, 90.0, 180.0, 270.0):
            for lr in (1, -1):
                for fb in (1, -1):
                    conv = RelationConvention(cam_sign, offset, lr, fb)
                    st = validate_oracle_against_solutions(questions, scenes, cameras_of, conv)
                    total = sum(t for _, t in st.values()) or 1
                    agree = sum(a for a, _ in st.values())
                    score = agree / total
                    results.append((score, conv, st))
                    if score > best_score:
                        best, best_score = conv, score
    if verbose:
        results.sort(key=lambda r: -r[0])
        print("[force3d] oracle convention calibration (top 5):")
        for score, conv, st in results[:5]:
            print(f"   {score:.3f}  {conv}  {dict(st)}")
    return best


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _scene_dir_from_path(shared_path: str) -> Tuple[str, str]:
    parts = shared_path.rstrip("/").split("/")
    return parts[-3], parts[-2]  # scene_XXXXXX, view hash


def _local_image_path(root: Path, scene_dir: str, view_hash: str) -> Path:
    jpg = root / "multiview" / scene_dir / view_hash / "image.jpg"
    if jpg.exists():
        return jpg
    png = jpg.with_suffix(".png")
    if png.exists():
        return png
    raise FileNotFoundError(f"no image for {scene_dir}/{view_hash} under {root}")


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _flatten_questions(json_path: Path) -> List[dict]:
    data = _load_json(json_path)
    questions = []
    for task in data["tasks"]:
        for q in task["data"]["questions"]:
            q = dict(q)
            q["_subset"] = task.get("subset")
            questions.append(q)
    return questions


def load_force3d(split: str = "puzzle", root: Path = FORCE3D_ROOT, image_transform=None,
                 view_policy: str = "first", limit: Optional[int] = None,
                 with_oracle: bool = True, convention: Optional[RelationConvention] = None,
                 verbose: bool = True) -> List[dict]:
    """Load a 3D-FORCE split into Clever-style samples.

    ``split`` is ``'puzzle'`` (yes/no) or ``'ref'`` (object index).  Only
    ``view_policy='first'`` (the question's camera 0) is implemented.
    """
    if split not in ("puzzle", "ref"):
        raise ValueError("split must be 'puzzle' or 'ref'")
    if view_policy != "first":
        raise NotImplementedError("multi-view policies are not implemented yet")
    if image_transform is None:
        from dataset import default_image_transform as image_transform  # noqa

    root = Path(root)
    json_name = "3DForcePuzzle.json" if split == "puzzle" else "3DForceRef.json"
    questions = _flatten_questions(root / json_name)
    if limit is not None:
        questions = questions[:limit]

    scenes: Dict[str, dict] = {}
    cameras_of: Dict[int, List[dict]] = {}
    for qi, q in enumerate(questions):
        scene_dir, _ = _scene_dir_from_path(q["image_filename"][0])
        q["_scene_dir"] = scene_dir
        if scene_dir not in scenes:
            scenes[scene_dir] = _load_json(root / "multiview" / scene_dir / "scene.json")
        cams = []
        for p in q["image_filename"]:
            sd, vh = _scene_dir_from_path(p)
            cams.append(_load_json(root / "multiview" / sd / vh / "camera.json"))
        cameras_of[qi] = cams

    if with_oracle and convention is None:
        convention = RelationConvention()
        if split == "puzzle" and verbose:
            # Verify (not search) the calibrated defaults against the solutions.
            st = validate_oracle_against_solutions(questions, scenes, cameras_of, convention)
            for fam, (a, t) in sorted(st.items()):
                flag = "" if t == 0 or a / t >= 0.95 else "  <-- LOW, run calibrate_convention()"
                print(f"[force3d] oracle check ({fam} relations): {a}/{t} agree{flag}")

    samples: List[dict] = []
    skipped = Counter()
    for qi, q in enumerate(questions):
        scene_dir = q["_scene_dir"]
        _, view_hash = _scene_dir_from_path(q["image_filename"][0])
        scene = scenes[scene_dir]
        objs = scene["objects"]
        n = len(objs)

        # Boxes: prefer the view's bboxes.json (slot order), fall back to the
        # question's own list (random views ship no bboxes.json).
        bfile = root / "multiview" / scene_dir / view_hash / "bboxes.json"
        boxes = None
        if bfile.exists():
            bobjs = sorted(_load_json(bfile)["objects"], key=lambda o: o["slot"])
            if len(bobjs) == n:
                boxes = [o["bbox_2d_pixels"] for o in bobjs]
        if boxes is None:
            boxes = q.get("bboxes")
        if boxes is None or len(boxes) != n:
            skipped["box_count_mismatch"] += 1
            continue
        boxes_np = np.array(boxes, dtype=np.float32).reshape(n, 4)

        try:
            logic_str = translate_force3d_program(q["program"], camera_alias=0)
        except ValueError as exc:
            skipped[f"translate:{exc}"[:60]] += 1
            continue

        image_path = _local_image_path(root, scene_dir, view_hash)
        pil_image = Image.open(image_path).convert("RGB")
        image_arr, boxes_t = image_transform(pil_image, boxes_np)

        all_objects = [
            {k: o.get(k) for k in ("color", "shape", "size", "material", "3d_coords", "rotation")}
            for o in objs
        ]
        sample = {
            "force3d_split": split,
            "scene_index": int(scene_dir.split("_")[1]),
            "scene_dir": scene_dir,
            "view_hash": view_hash,
            "image_index": f"{scene_dir}_{view_hash}",
            "image_filename": str(image_path),
            "pil_image": pil_image,
            "image": image_arr,
            "objects_raw": np.asarray(boxes_t, dtype=np.float32),
            "all_objects": all_objects,
            "question_raw": q["question"],
            "question": q["question"],
            "program_str": q["program"],
            "program": program_predicates(q["program"]),
            "logic_str": logic_str,
            "relation_perspective": q.get("relation_perspective"),
            "subset": q.get("_subset"),
            "n_views": len(q["image_filename"]),
        }
        if split == "puzzle":
            sample["answer"] = bool(q["answer"])
            sample["solution"] = q.get("solution")
        else:
            idx = int(q["answer"])
            if not 0 <= idx < n:
                skipped["ref_answer_out_of_range"] += 1
                continue
            sample["answer"] = idx
            sample["answer_index"] = idx
            # Shape [1, n]: Clever also attaches ``constraint['label']`` to the
            # raw key, and a 1-D vector there would create one constraint
            # DataNode per object.  LogicDataset keeps 2-D labels as-is.
            label = torch.zeros(1, n, dtype=torch.float32)
            label[0, idx] = 1.0
            sample["logic_label"] = label
            sample["anchor_indices"] = q.get("anchor_indices")
        if with_oracle:
            sample["relation_spatial_relation"] = compute_relation_labels(
                scene, cameras_of[qi], FORCE3D_RELATIONS, convention)
        distinct_logits = [[0, 100] if i != j else [100, 0] for i in range(n) for j in range(n)]
        sample[f"oracle_is_{DISTINCT}"] = distinct_logits
        sample[f"oracle_is_{DISTINCT}_rev"] = list(distinct_logits)
        samples.append(sample)

    if verbose:
        print(f"[force3d] split={split}: {len(samples)} samples from {len(questions)} questions, "
              f"{len(scenes)} scenes; skipped={dict(skipped)}")
    return samples


def split_by_scene(samples: Sequence[dict], n_test_scenes: int, seed: int = 0
                   ) -> Tuple[List[dict], List[dict]]:
    """Hold out whole scenes so no image appears in both train and test."""
    scene_ids = sorted({s["scene_dir"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    test_scenes = set(scene_ids[:n_test_scenes])
    train = [s for s in samples if s["scene_dir"] not in test_scenes]
    test = [s for s in samples if s["scene_dir"] in test_scenes]
    return train, test


# ---------------------------------------------------------------------------
# REF metric
# ---------------------------------------------------------------------------

def force3d_ref_top1(program, dataset, device="cpu") -> Tuple[float, int, int]:
    """Top-1 accuracy of the miotaL selection against ``answer_index``.

    Reads the same ``selectionDistribution`` the framework's evaluator uses,
    but scores argmax == answer instead of thresholded exact match.
    """
    from domiknows.graph.logicalConstrain import miotaL

    correct = total = 0
    program.model.eval()
    program.model.reset() if hasattr(program.model, "reset") else None
    with torch.no_grad():
        for datanode, sample in zip(program.populate(dataset, device=device), dataset):
            answer = sample.get("answer_index")
            if answer is None:
                continue
            active = datanode.getActiveExecutableConstraintNames()
            for lc_name in active:
                lc = program.graph.executableLCs.get(lc_name)
                if lc is None or not isinstance(getattr(lc, "innerLC", lc), miotaL):
                    continue
                cmodel = getattr(program, "cmodel", None)
                loss_dict = datanode.calculateSingleLcLoss(
                    lc_name,
                    tnorm=getattr(cmodel, "tnorm", "P"),
                    counting_tnorm=getattr(cmodel, "counting_tnorm", None),
                )
                dist = loss_dict.get("selectionDistribution")
                if dist is None:
                    continue
                pred = int(torch.argmax(dist.detach().reshape(-1)).item())
                total += 1
                correct += int(pred == int(answer))
    acc = 100.0 * correct / total if total else 0.0
    return acc, correct, total
