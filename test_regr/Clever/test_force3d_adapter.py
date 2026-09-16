"""CPU-only tests for the 3D-FORCE adapter (loader, translator, oracle, baselines)."""
import os
import re
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import force3d_dataset as F  # noqa: E402

HAS_DATA = (F.FORCE3D_ROOT / "3DForcePuzzle.json").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="3D-FORCE dataset not present")


def test_translate_puzzle_golden():
    prog = ("exists(Object, lambda x1: exists(Object, lambda x2: exists(Object, lambda x3: "
            "exists(Object, lambda x4: brown(x1) and brown(x2) and school(x3) and school(x4) "
            "and obj_left(x2, x4) ))))")
    assert F.translate_force3d_program(prog) == (
        "existsL(andL(brown('a'), brown('b'), school('c'), school('d'), "
        + F.object_relation_formula("obj_left", "b", "d") + ", "
        "distinct('a', 'b'), distinct('a', 'c'), distinct('a', 'd'), distinct('b', 'c'), "
        "distinct('b', 'd'), distinct('c', 'd')))")


def test_translate_maps_camera_relations_to_direction_bins():
    prog = ("exists(Object, lambda x1: exists(Object, lambda x2: red(x1) and suv(x2) "
            "and left_0(x1, x2) and front_2(x2, x1)))")
    bins = {"left_0": "dir180", "front_2": "dir270"}
    out = F.translate_force3d_program(prog, camera_bins=bins)
    assert out == ("existsL(andL(red('a'), suv('b'), dir180('a', 'b'), dir270('b', 'a'), "
                   "distinct('a', 'b')))")
    assert "left_0('a', 'b')" in F.translate_force3d_program(prog)  # no bins: names kept
    with pytest.raises(ValueError):  # camera 2 missing from this question's bins
        F.translate_force3d_program(prog, camera_bins={"left_0": "dir180"})


def test_translate_ref_golden_flattens_iota_chain():
    prog = ("point(Object, lambda x: yellow(x) and suv(x) and obj_left(x, iota(Object, lambda y: "
            "yellow(y) and obj_front(y, iota(Object, lambda z: scooter(z))))))")
    out = F.translate_force3d_program(prog)
    assert out == ("miotaL(andL(yellow('a'), suv('a'), yellow('b'), scooter('c'), "
                   + F.object_relation_formula("obj_front", "b", "c") + ", "
                   + F.object_relation_formula("obj_left", "a", "b") + ", distinct('a', 'b'), "
                   "distinct('a', 'c'), distinct('b', 'c')), threshold=0.5, hard=False)")


def test_object_relation_formula_uses_heading_and_direction_bins():
    out = F.object_relation_formula("obj_left", "a", "b")
    n = 360 // F.HEADING_BIN_DEG
    assert out.startswith("orL(andL(hd0('b'), dir270('a', 'b')), ")
    assert out.count("andL(") == n and "obj_" not in out
    assert "andL(hd90('b'), dir0('a', 'b'))" in out  # heading 90 (toward camera): its left is camera-right
    assert "andL(hd0('b'), dir90('a', 'b'))" in F.object_relation_formula("obj_right", "a", "b")
    assert set(F.HEADING_CONCEPTS) == set(F.FORCE3D_ATTRIBUTE_CONCEPTS["heading"])
    assert not any(r.startswith("obj_") for r in F.FORCE3D_RELATIONS)


@needs_data
def test_composed_object_relations_match_exact_relations():
    """hd(b) AND dir-half-plane(a, b), with true bins, vs the exact obj_* labels."""
    samples = F.load_force3d(split="puzzle", limit=300, with_images=False, verbose=False)
    conv = F.RelationConvention()
    agree = total = 0
    seen = set()
    for s in samples:
        if s["image_index"] in seen:
            continue
        seen.add(s["image_index"])
        objs = s["all_objects"]
        n = len(objs)
        coords = np.array([o["3d_coords"][:2] for o in objs], dtype=np.float64)
        rots = [o["rotation"] for o in objs]
        dirs = s["relation_spatial_relation"]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                h = int(objs[j]["heading"][2:])
                for d, off in F.OBJECT_RELATION_OFFSET_DEG.items():
                    b = F.FORCE3D_RELATIONS.index(f"dir{(h + off) % 360}")
                    pred = dirs[i * n + j, b] > 0.5
                    truth = F.relation_holds(f"obj_{d}", i, j, coords, rots, {}, conv)
                    agree += int(pred == truth)
                    total += 1
    assert total > 1000 and agree / total >= 0.96, (agree, total)


def test_translate_single_predicate_and_unknown_predicate():
    assert F.translate_force3d_program("point(Object, lambda x: tank(x))") == \
        "miotaL(tank('a'), threshold=0.5, hard=False)"
    with pytest.raises(ValueError):
        F.translate_force3d_program("exists(Object, lambda x: pink(x))")


def test_opposite_relation():
    assert F.opposite_relation("left") == "right"
    assert F.opposite_relation("obj_front") == "obj_behind"
    assert F.opposite_relation("behind_3") == "front_3"
    assert F.opposite_relation("dir30") == "dir210"
    assert F.opposite_relation("dir270") == "dir90"
    assert F.opposite_relation("distinct") == "distinct"


def test_model_vocabulary_has_no_camera_index_names():
    assert not any(re.fullmatch(r"(left|right|front|behind)(_\d)?", r) for r in F.FORCE3D_RELATIONS)
    assert len(F.CAMERA_DIRECTION_RELATIONS) == 360 // F.DIR_BIN_DEG


def test_relation_holds_camera_geometry_and_bins():
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
    # direction bins in camera 0's frame agree with camera-0 relations
    for name, bin_name in (("right", "dir0"), ("front", "dir90"), ("left", "dir180"), ("behind", "dir270")):
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert F.relation_holds(name, i, j, coords, [0, 0, 0], dirs, conv) == \
                        F.relation_holds(bin_name, i, j, coords, [0, 0, 0], dirs, conv)
    # camera index not present in this question -> False, never an exception
    assert not F.relation_holds("left_3", 0, 2, coords, [0, 0, 0], dirs, conv)


def test_camera_bins_exact_for_rotated_cameras():
    scene = {"camera_ring": {"target": [0.0, 0.0, 0.7]}}
    cams = [{"location": [10.0 * np.cos(a), 10.0 * np.sin(a), 7.0]}
            for a in np.radians([30.0, 30.0 + 90.0, 30.0 + 180.0, 30.0 + 270.0])]
    bins, worst = F.camera_relation_bins(scene, cams)
    assert worst < 1e-6
    assert bins["left_0"] == "dir180" and bins["right_2"] == "dir180"  # camera 2 faces camera 0
    rng = np.random.default_rng(0)
    objs = [{"3d_coords": [float(x), float(y), 0.0], "rotation": 0.0} for x, y in rng.uniform(-5, 5, (8, 2))]
    scene = {**scene, "objects": objs}
    conv = F.RelationConvention()
    by_name = F.compute_relation_labels(scene, cams, F.QUESTION_RELATIONS, conv)
    by_bin = F.compute_relation_labels(scene, cams, F.FORCE3D_RELATIONS, conv)
    for name, bin_name in bins.items():
        np.testing.assert_array_equal(by_name[:, F.QUESTION_RELATIONS.index(name)],
                                      by_bin[:, F.FORCE3D_RELATIONS.index(bin_name)])


def test_structure_baseline_is_chance_on_contrastive_pairs_and_catches_length_bias():
    def q(prog, answer, scene):
        return {"program_str": prog, "answer": answer, "scene_dir": scene}
    short = "exists(Object, lambda x1: red(x1) )"
    long_ = ("exists(Object, lambda x1: exists(Object, lambda x2: red(x1) and suv(x2) "
             "and left_0(x1, x2) ))")
    # paired: each structure appears once true, once false
    paired = [q(short, True, "s1"), q(short, False, "s1"), q(long_, True, "s2"), q(long_, False, "s2")]
    acc, _, n = F.structure_only_baseline(paired, paired)
    assert n == 4 and acc == 50.0
    # biased: short -> true, long -> false, structure alone is perfect
    biased = [q(short, True, "s1"), q(long_, False, "s2")] * 3
    acc, _, _ = F.structure_only_baseline(biased, biased)
    assert acc == 100.0


def test_boxes_in_backbone_frame():
    from modules import BACKBONE_INPUT_SIZE, boxes_in_backbone_frame
    img = Image.new("RGB", (1024, 768))
    boxes = np.array([[512.0, 384.0, 1024.0, 768.0]], dtype=np.float32)
    out = boxes_in_backbone_frame(boxes, img)
    s = BACKBONE_INPUT_SIZE
    np.testing.assert_allclose(out, [[s / 2, s / 2, s, s]])
    np.testing.assert_allclose(boxes_in_backbone_frame(boxes, None), boxes)  # unknown image: unchanged


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
        width, height = s["pil_image"].size
        assert (s["objects_raw"][:, 2] <= width + 1).all() and (s["objects_raw"][:, 3] <= height + 1).all()
        assert os.path.exists(s["image_filename"])
        assert isinstance(s["logic_str"], str) and s["logic_str"].endswith(")")
        assert not re.search(r"\b(?:left|right|front|behind)(?:_\d)?\(", s["logic_str"])
        assert isinstance(s["program"], list) and len(s["program"]) >= 1
        assert s["relation_spatial_relation"].shape == (n * n, len(F.FORCE3D_RELATIONS))
        assert set(s["all_objects"][0]) >= {"color", "shape", "heading"}
        assert all(o["heading"] in F.HEADING_CONCEPTS for o in s["all_objects"])
        assert "obj_" not in s["logic_str"]
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
