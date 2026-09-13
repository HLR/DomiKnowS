"""CPU-only tests for the 3D-FORCE adapter (loader, translator, oracle)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import force3d_dataset as F  # noqa: E402

HAS_DATA = (F.FORCE3D_ROOT / "3DForcePuzzle.json").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="3D-FORCE dataset not present")


def test_translate_puzzle_golden():
    prog = ("exists(Object, lambda x1: exists(Object, lambda x2: exists(Object, lambda x3: "
            "exists(Object, lambda x4: brown(x1) and brown(x2) and school(x3) and school(x4) "
            "and obj_left(x2, x4) ))))")
    assert F.translate_force3d_program(prog) == (
        "existsL(andL(brown('a'), brown('b'), school('c'), school('d'), obj_left('b', 'd'), "
        "distinct('a', 'b'), distinct('a', 'c'), distinct('a', 'd'), distinct('b', 'c'), "
        "distinct('b', 'd'), distinct('c', 'd')))")


def test_translate_camera_alias_and_keep_other_cameras():
    prog = "exists(Object, lambda x1: exists(Object, lambda x2: red(x1) and suv(x2) and left_0(x1, x2) and front_2(x2, x1)))"
    out = F.translate_force3d_program(prog, camera_alias=0)
    assert out == ("existsL(andL(red('a'), suv('b'), left('a', 'b'), front_2('b', 'a'), "
                   "distinct('a', 'b')))")
    out_none = F.translate_force3d_program(prog, camera_alias=None)
    assert "left_0('a', 'b')" in out_none


def test_translate_ref_golden_flattens_iota_chain():
    prog = ("point(Object, lambda x: yellow(x) and suv(x) and obj_left(x, iota(Object, lambda y: "
            "yellow(y) and obj_front(y, iota(Object, lambda z: scooter(z))))))")
    out = F.translate_force3d_program(prog)
    assert out == ("miotaL(andL(yellow('a'), suv('a'), yellow('b'), scooter('c'), "
                   "obj_front('b', 'c'), obj_left('a', 'b'), distinct('a', 'b'), "
                   "distinct('a', 'c'), distinct('b', 'c')), threshold=0.5, hard=False)")


def test_translate_single_predicate_and_unknown_predicate():
    assert F.translate_force3d_program("point(Object, lambda x: tank(x))") == \
        "miotaL(tank('a'), threshold=0.5, hard=False)"
    with pytest.raises(ValueError):
        F.translate_force3d_program("exists(Object, lambda x: pink(x))")


def test_opposite_relation():
    assert F.opposite_relation("left") == "right"
    assert F.opposite_relation("obj_front") == "obj_behind"
    assert F.opposite_relation("behind_3") == "front_3"
    assert F.opposite_relation("distinct") == "distinct"


def test_relation_holds_camera_geometry():
    scene = {"camera_ring": {"target": [0.0, 0.0, 0.7]}}
    cam = {"location": [0.0, -10.0, 7.0]}  # camera south of origin looking north
    dirs = {0: F.camera_direction_vectors(scene, cam)}
    coords = np.array([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    conv = F.RelationConvention()
    # object 0 is closer to the camera than object 1 -> in front of it
    assert F.relation_holds("front", 0, 1, coords, [0, 0, 0], dirs, conv)
    assert F.relation_holds("behind_0", 1, 0, coords, [0, 0, 0], dirs, conv)
    # object 2 is to the camera's right of object 0
    assert F.relation_holds("right", 2, 0, coords, [0, 0, 0], dirs, conv)
    assert F.relation_holds("left", 0, 2, coords, [0, 0, 0], dirs, conv)
    # camera index not present in this question -> False, never an exception
    assert not F.relation_holds("left_3", 0, 2, coords, [0, 0, 0], dirs, conv)


@needs_data
def test_oracle_agrees_with_puzzle_solutions():
    root = F.FORCE3D_ROOT
    qs = F._flatten_questions(root / "3DForcePuzzle.json")[:400]
    scenes, cams = {}, {}
    for qi, q in enumerate(qs):
        sd, _ = F._scene_dir_from_path(q["image_filename"][0])
        q["_scene_dir"] = sd
        scenes.setdefault(sd, F._load_json(root / "multiview" / sd / "scene.json"))
        cams[qi] = [F._load_json(root / "multiview" / F._scene_dir_from_path(p)[0]
                                 / F._scene_dir_from_path(p)[1] / "camera.json")
                    for p in q["image_filename"]]
    st = F.validate_oracle_against_solutions(qs, scenes, cams, F.RelationConvention())
    for fam, (a, t) in st.items():
        assert t > 0 and a / t >= 0.95, (fam, a, t)


@needs_data
@pytest.mark.parametrize("split", ["puzzle", "ref"])
def test_loader_contract(split):
    samples = F.load_force3d(split=split, limit=6, verbose=False)
    assert len(samples) >= 1
    for s in samples:
        n = len(s["all_objects"])
        assert s["objects_raw"].shape == (n, 4) and s["objects_raw"].dtype == np.float32
        assert s["pil_image"].size[0] > 0 and os.path.exists(s["image_filename"])
        assert isinstance(s["logic_str"], str) and s["logic_str"].endswith(")")
        assert isinstance(s["program"], list) and len(s["program"]) >= 1
        assert s["relation_spatial_relation"].shape == (n * n, len(F.FORCE3D_RELATIONS))
        assert set(s["all_objects"][0]) >= {"color", "shape"}
        if split == "puzzle":
            assert isinstance(s["answer"], bool)
            assert s["logic_str"].startswith("existsL(")
        else:
            assert s["logic_str"].startswith("miotaL(")
            assert torch.is_tensor(s["logic_label"]) and s["logic_label"].shape == (1, n)
            assert int(s["logic_label"].argmax()) == s["answer_index"]


@needs_data
def test_split_by_scene_is_disjoint():
    samples = F.load_force3d(split="puzzle", limit=60, with_oracle=False, verbose=False)
    train, test = F.split_by_scene(samples, n_test_scenes=2, seed=0)
    assert len(train) + len(test) == len(samples)
    assert not ({s["scene_dir"] for s in train} & {s["scene_dir"] for s in test})
