"""3D-FORCE adapter for the Clever/DomiKnowS pipeline.

Loads the 3D-FORCE Puzzle (yes/no ``exists`` programs) and REF (object
selection ``point``/``iota`` programs) splits into the sample format consumed
by ``main.py``'s sensors, translates the lambda-style programs into DomiKnowS
executable logic strings, and derives oracle spatial-relation labels from the
3D scene geometry.

Sample contract (what ``main.py`` reads):
    pil_image, image_filename, image_index, objects_raw (float32 n x 4, raw pixel
    coordinates of pil_image; main.py rescales them into the backbone frame),
    all_objects (list of dicts with color/shape/...),
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

# Relation vocabulary.
#
# Questions name camera relations by camera index (``left_k``: left of, seen
# from the question's k-th view).  The angle of camera k relative to camera 0
# depends on the view setup (camera 1 is 180/120/90 degrees away in 2/3/4-view
# scenes, arbitrary for random views), so one head per index name receives
# contradictory labels (50% agreement between 2- and 4-view renderings of the
# same scene).  Every camera relation is a half-plane test along a
# ground-plane direction, so the model vocabulary expresses it as a direction
# in camera 0's frame (the fed view), quantised to DIR_BIN_DEG degrees:
# ``dir0`` = right of (camera 0), ``dir90`` = in front of (toward camera 0),
# ``dir180`` = left of, ``dir270`` = behind.  ``camera_relation_bins`` maps a
# question's names onto these bins with that question's cameras; fixed
# 2/3/4-view setups map exactly, random views within DIR_BIN_DEG / 2.
# ``obj_*`` are anchor-object perspective relations (heading dependent).
# ``distinct`` is a fixed (never learned) identity relation: distinct(i, j)
# iff i != j.  The puzzle/REF semantics require every logical variable to
# bind a different object, which the existential reading of andL does not
# enforce on its own; the translator adds distinct('x', 'y') for every pair
# of variables.  main.py wires it through OracleDummyLearner in all modes.
DISTINCT = "distinct"
DIR_BIN_DEG = 30
CAMERA_DIRECTION_RELATIONS: List[str] = [f"dir{a}" for a in range(0, 360, DIR_BIN_DEG)]
OBJECT_RELATIONS: List[str] = [f"obj_{d}" for d in DIRECTIONS]
# Names that may appear in question programs (exact question semantics; used by
# the oracle check and the free generator).
QUESTION_RELATIONS: List[str] = (
    list(DIRECTIONS)
    + OBJECT_RELATIONS
    + [f"{d}_{k}" for k in range(N_CAMERAS) for d in DIRECTIONS]
    + [DISTINCT]
)
# Names the model learns, one head each.
FORCE3D_RELATIONS: List[str] = CAMERA_DIRECTION_RELATIONS + OBJECT_RELATIONS + [DISTINCT]
FORCE3D_RELATIONAL_CONCEPTS: Dict[str, List[str]] = {
    "spatial_relation": FORCE3D_RELATIONS,
}

_OPPOSITE = {"left": "right", "right": "left", "front": "behind", "behind": "front"}


def opposite_relation(name: str) -> str:
    """left_2 -> right_2, obj_front -> obj_behind, left -> right, dir30 -> dir210,
    distinct -> distinct."""
    if name == DISTINCT:
        return DISTINCT
    if name.startswith("dir") and name[3:].isdigit():
        return f"dir{(int(name[3:]) + 180) % 360}"
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
_BINARY = set(FORCE3D_RELATIONS) | set(QUESTION_RELATIONS)
_QUANTIFIERS = {"exists", "point", "iota"}


def _var_letter(index: int) -> str:
    if index >= 26:
        raise ValueError("more than 26 logical variables in one program")
    return chr(ord("a") + index)


class _Translator:
    """AST walker turning 3D-FORCE lambda programs into DomiKnowS logic."""

    def __init__(self, camera_bins: Optional[Dict[str, str]] = None):
        self.camera_bins = camera_bins
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
        is_camera_name = not (name.startswith("obj_") or name.startswith("dir") or name == DISTINCT)
        if self.camera_bins is not None and is_camera_name:
            if name not in self.camera_bins:
                raise ValueError(f"camera relation {name!r} refers to a view this question does not have")
            return self.camera_bins[name]
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


def translate_force3d_program(program_str: str, camera_bins: Optional[Dict[str, str]] = None) -> str:
    """Translate a 3D-FORCE program string into a DomiKnowS logic string.

    Puzzle (``exists``) programs become ``existsL(andL(...))``; REF (``point``)
    programs become ``miotaL(andL(...), threshold=0.5, hard=False)`` whose
    first variable ``'a'`` is the selected object.  Nested ``iota`` anchors are
    flattened into extra existential variables.  ``camera_bins`` (from
    ``camera_relation_bins``) maps camera-index relation names such as
    ``left_2`` onto the model's direction bins; without it names are kept.
    """
    return _Translator(camera_bins).translate(program_str)


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


def _direction_unit(camera0_dirs: Dict[str, np.ndarray], bin_name: str) -> np.ndarray:
    """Ground-plane unit vector of a ``dirA`` bin in camera 0's frame."""
    a = math.radians(int(bin_name[3:]))
    c, s = math.cos(a), math.sin(a)
    # Snap floating-point residue (cos 90 = 6e-17) so axis-aligned bins equal
    # the camera-0 relations exactly, including for exactly aligned objects.
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s
    return c * camera0_dirs["right"] + s * camera0_dirs["front"]


def camera_relation_bins(scene: dict, cameras: Sequence[dict]) -> Tuple[Dict[str, str], float]:
    """Map a question's camera-relation names to model direction bins.

    Returns ``({"left_1": "dir0", ..., "left": "dir180", ...}, worst_error_deg)``
    where the error is the largest angle between a relation's exact direction
    and its bin centre (0 for fixed 2/3/4-view setups).
    """
    base = camera_direction_vectors(scene, cameras[0])
    right0, front0 = base["right"], base["front"]
    mapping: Dict[str, str] = {}
    worst = 0.0
    n_bins = 360 // DIR_BIN_DEG
    for k, cam in enumerate(cameras):
        vecs = camera_direction_vectors(scene, cam)
        for d in DIRECTIONS:
            v = vecs[d]
            ang = math.degrees(math.atan2(float(v @ front0), float(v @ right0))) % 360.0
            b = (int(round(ang / DIR_BIN_DEG)) % n_bins) * DIR_BIN_DEG
            worst = max(worst, abs((ang - b + 180.0) % 360.0 - 180.0))
            mapping[f"{d}_{k}"] = f"dir{b}"
            if k == 0:
                mapping[d] = f"dir{b}"
    return mapping, worst


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
    if name.startswith("dir"):
        if 0 not in camera_dirs:
            return False
        return float(np.dot(d, _direction_unit(camera_dirs[0], name))) * conv.camera_sign > 0
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
    """(n*n, len(relations)) float32 matrix, row i*n+j == relation(i, j).

    Vectorised: one broadcasted dot product per relation instead of a Python
    loop over every pair (the loop cost ~0.5 s per question, which made a
    44k-question set take hours to load).  Semantics identical to
    ``relation_holds``.
    """
    objs = scene["objects"]
    n = len(objs)
    coords = np.array([o["3d_coords"][:2] for o in objs], dtype=np.float64)
    rot = np.radians(np.array([float(o.get("rotation", 0.0)) for o in objs]) + conv.object_heading_offset_deg)
    heading = np.stack([np.cos(rot), np.sin(rot)], axis=1)          # per anchor j
    left_vec = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
    d = coords[:, None, :] - coords[None, :, :]                       # d[i, j] = p_i - p_j
    camera_dirs = {k: camera_direction_vectors(scene, cam) for k, cam in enumerate(cameras)}
    not_diag = ~np.eye(n, dtype=bool)
    out = np.zeros((n * n, len(relations)), dtype=np.float32)
    for r_idx, name in enumerate(relations):
        if name == DISTINCT:
            mat = not_diag
        elif name.startswith("obj_"):
            direction = name[4:]
            if direction in ("front", "behind"):
                val = np.einsum("ijk,jk->ij", d, heading) * conv.object_fb_sign
                mat = val > 0 if direction == "front" else val < 0
            else:
                val = np.einsum("ijk,jk->ij", d, left_vec) * conv.object_lr_sign
                mat = val > 0 if direction == "left" else val < 0
        elif name.startswith("dir"):
            if 0 not in camera_dirs:
                continue
            mat = (d @ _direction_unit(camera_dirs[0], name)) * conv.camera_sign > 0
        else:
            if name in DIRECTIONS:
                direction, cam = name, 0
            else:
                direction, cam_s = name.rsplit("_", 1)
                cam = int(cam_s)
            if cam not in camera_dirs:
                continue
            val = d @ camera_dirs[cam][direction] * conv.camera_sign
            mat = val > 0
        out[:, r_idx] = (mat & not_diag).reshape(-1).astype(np.float32)
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


_JSON_MEMO: Dict[str, object] = {}


def _load_json(path: Path, memo: bool = False):
    """Read JSON; ``memo=True`` caches small per-scene files (scene/camera/bboxes),
    which are re-read for every question and live on a network share."""
    key = str(path)
    if memo and key in _JSON_MEMO:
        return _JSON_MEMO[key]
    with open(path) as f:
        data = json.load(f)
    if memo:
        _JSON_MEMO[key] = data
    return data


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
                 verbose: bool = True, json_name: Optional[str] = None,
                 with_images: bool = True) -> List[dict]:
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
    if json_name is None:
        json_name = "3DForcePuzzle.json" if split == "puzzle" else "3DForceRef.json"
    json_path = Path(json_name) if os.path.isabs(json_name) else root / json_name
    questions = _flatten_questions(json_path)
    if limit is not None:
        questions = questions[:limit]

    scenes: Dict[str, dict] = {}
    cameras_of: Dict[int, List[dict]] = {}
    for qi, q in enumerate(questions):
        scene_dir, _ = _scene_dir_from_path(q["image_filename"][0])
        q["_scene_dir"] = scene_dir
        if scene_dir not in scenes:
            scenes[scene_dir] = _load_json(root / "multiview" / scene_dir / "scene.json", memo=True)
        cams = []
        for p in q["image_filename"]:
            sd, vh = _scene_dir_from_path(p)
            cams.append(_load_json(root / "multiview" / sd / vh / "camera.json", memo=True))
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
                # Clipped to the image: objects cut off by the frame have
                # unclipped boxes that extend past the image edge.
                boxes = [o.get("bbox_2d_clipped") or o["bbox_2d_pixels"] for o in bobjs]
        if boxes is None:
            boxes = q.get("bboxes")
        if boxes is None or len(boxes) != n:
            skipped["box_count_mismatch"] += 1
            continue
        boxes_np = np.array(boxes, dtype=np.float32).reshape(n, 4)
        width, height = float(scene.get("width", 1024)), float(scene.get("height", 768))
        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0.0, width)
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0.0, height)

        try:
            camera_bins, bin_error = camera_relation_bins(scene, cameras_of[qi])
            logic_str = translate_force3d_program(q["program"], camera_bins=camera_bins)
        except ValueError as exc:
            skipped[f"translate:{exc}"[:60]] += 1
            continue

        image_path = _local_image_path(root, scene_dir, view_hash)
        if with_images:
            pil_image = Image.open(image_path).convert("RGB")
            image_arr, boxes_t = image_transform(pil_image, boxes_np)
        else:
            # Light sample: images attached later by ``attach_images`` for the
            # subset actually used (a 44k-question set with images does not fit
            # in a cache file or comfortably in memory).
            pil_image, image_arr, boxes_t = None, None, boxes_np

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
            # Raw pixel boxes of pil_image (same convention as the CLEVR loader);
            # main.py rescales them into the backbone's input frame.
            "objects_raw": boxes_np,
            "_raw_boxes": boxes_np,
            "all_objects": all_objects,
            "question_raw": q["question"],
            "question": q["question"],
            "program_str": q["program"],
            "program": program_predicates(q["program"]),
            "logic_str": logic_str,
            "relation_perspective": q.get("relation_perspective"),
            "subset": q.get("_subset"),
            "n_views": len(q["image_filename"]),
            "camera_bin_error_deg": bin_error,
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
        coarse = sum(1 for s in samples if s["camera_bin_error_deg"] > 1.0)
        print(f"[force3d] split={split}: {len(samples)} samples from {len(questions)} questions, "
              f"{len(scenes)} scenes; skipped={dict(skipped)}; camera relations binned at "
              f"{DIR_BIN_DEG} deg ({coarse} questions with a >1 deg quantisation error)")
    return samples


def attach_images(samples: Sequence[dict], image_transform=None) -> None:
    """Load and transform images for light samples in place."""
    if image_transform is None:
        from dataset import default_image_transform as image_transform  # noqa
    for s in samples:
        if s.get("pil_image") is not None:
            continue
        pil_image = Image.open(s["image_filename"]).convert("RGB")
        image_arr, _ = image_transform(pil_image, s["_raw_boxes"])
        s["pil_image"] = pil_image
        s["image"] = image_arr
        s["objects_raw"] = np.asarray(s["_raw_boxes"], dtype=np.float32)  # raw pixels


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


def soft_constraint_accuracy(program, dataset, device="cpu", tnorm="P"):
    """Accuracy of P(formula satisfied) > 0.5 against the yes/no label.

    The exact evaluator thresholds every predicate at 0.5 before evaluating
    the formula, so a model whose predicates are still below 0.5 everywhere
    scores the constant class rate even while its satisfaction probabilities
    already separate the classes.  This soft score shows that progress.
    Returns (accuracy_percent, correct, total, mean_P_positives, mean_P_negatives).
    """
    correct = total = 0
    p_pos, p_neg = [], []
    with torch.no_grad():
        for datanode, sample in zip(program.populate(dataset, device=device), dataset):
            label = sample.get("logic_label")
            if label is None or (torch.is_tensor(label) and label.numel() != 1):
                continue
            label = int(label.reshape(-1)[0]) if torch.is_tensor(label) else int(bool(label))
            for lc_name in datanode.getActiveExecutableConstraintNames():
                out = datanode.calculateSingleLcLoss(lc_name, tnorm=tnorm)
                prob = out.get("conversionSigmoid") if isinstance(out, dict) else None
                if prob is None:
                    continue
                prob = float(torch.as_tensor(prob).reshape(-1)[0])
                total += 1
                correct += int((prob > 0.5) == bool(label))
                (p_pos if label else p_neg).append(prob)
    acc = 100.0 * correct / total if total else 0.0
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return acc, correct, total, mean(p_pos), mean(p_neg)


# ---------------------------------------------------------------------------
# Structure-only baseline
# ---------------------------------------------------------------------------

def structure_signature(program_str: str) -> Tuple[int, int, int, int, int]:
    """Question structure without scene content.

    (variables, descriptor tokens, bare variables, object-perspective
    relations, camera relations).  Enough to reproduce the length shortcut
    that lets a model with chance-level predicates beat 50%.
    """
    variables = re.findall(r"lambda\s+(\w+)\s*:", program_str)
    unary = re.findall(r"\b([a-z]+)\((\w+)\)", program_str)
    described = {arg for _, arg in unary}
    binary = re.findall(r"\b([a-z_]+\d*)\(\w+\s*,", program_str)
    binary = [b for b in binary if b not in _QUANTIFIERS]
    n_obj = sum(1 for b in binary if b.startswith("obj_"))
    n_cam = sum(1 for b in binary if not b.startswith("obj_") and b != DISTINCT)
    return (len(variables), len(unary), len(set(variables) - described), n_obj, n_cam)


def structure_only_baseline(train: Sequence[dict], test: Sequence[dict]) -> Tuple[float, float, int]:
    """Held-out accuracy of answering from question structure alone.

    Fits the majority answer per ``structure_signature`` on ``train`` and
    scores ``test``.  Returns (structure_accuracy_pct, majority_class_pct, n).
    A trained model has to beat this number, not 50%.
    """
    table: Dict[Tuple, Counter] = defaultdict(Counter)
    for s in train:
        table[structure_signature(s["program_str"])][bool(s["answer"])] += 1
    overall = Counter(bool(s["answer"]) for s in train)
    majority = overall.most_common(1)[0][0] if overall else True
    correct = majority_correct = 0
    for s in test:
        counts = table.get(structure_signature(s["program_str"]))
        pred = majority
        if counts:
            (top, top_n), *rest = counts.most_common()
            pred = top if not rest or rest[0][1] < top_n else majority
        correct += int(pred == bool(s["answer"]))
        majority_correct += int(majority == bool(s["answer"]))
    n = len(test)
    if not n:
        return float("nan"), float("nan"), 0
    return 100.0 * correct / n, 100.0 * majority_correct / n, n
