"""
test_main.py

Pytest suite for the visual-constraint example.
Validates ILP inference, constraint verification, and loss calculation
for the generic constraint library.
"""

import sys
import pytest

# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def program():
    import torch
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import (
        EdgeSensor, CompositionCandidateSensor,
    )
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor

    from graph import (
        graph, image, object_node, image_contains_object,
        pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder,
        right_of, left_of, material,
    )
    from sensor import (
        SmallLearner, LargeLearner,
        RedLearner, GreenLearner, BlueLearner,
        CubeLearner, SphereLearner, CylinderLearner,
        RightOfLearner, LeftOfLearner,
        MaterialEnumLearner,
    )

    graph.detach()

    # --- image / object scaffolding ---
    image["index"] = ReaderSensor(keyword="image")
    object_node["index"] = ReaderSensor(keyword="objects")
    object_node[image_contains_object] = EdgeSensor(
        object_node["index"], image["index"],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1),
    )

    # --- pair scaffolding ---
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node["index"],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True,
    )

    # --- attribute learners ---
    object_node[small]    = SmallLearner("index")
    object_node[large]    = LargeLearner("index")
    object_node[red]      = RedLearner("index")
    object_node[green]    = GreenLearner("index")
    object_node[blue]     = BlueLearner("index")
    object_node[cube]     = CubeLearner("index")
    object_node[sphere]   = SphereLearner("index")
    object_node[cylinder] = CylinderLearner("index")

    # --- material enum learner ---
    object_node[material] = MaterialEnumLearner("index")

    # --- spatial DataNodeReaderSensors ---
    def _read_spatial(rel_arg1_ref, rel_arg2_ref, keyword):
        def _fn(*_, data, datanode):
            a1 = datanode.relationLinks[rel_arg1_ref.name][0].getAttribute("index").item()
            a2 = datanode.relationLinks[rel_arg2_ref.name][0].getAttribute("index").item()
            dev = getattr(datanode, "current_device", "cpu")
            if (a1, a2) in data:
                return torch.tensor([0, 1], device=dev)
            return torch.tensor([1, 0], device=dev)
        return _fn

    pair[right_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="right_of",
        forward=_read_spatial(rel_arg1, rel_arg2, "right_of"),
    )
    pair[left_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="left_of",
        forward=_read_spatial(rel_arg1, rel_arg2, "left_of"),
    )

    return LearningBasedProgram(graph, PoiModel, poi=[image, object_node, pair])


@pytest.fixture(scope="module")
def dataset():
    from .reader import SceneReader
    return SceneReader().run()


# Expected constants
EXPECTED_SMALL_BLUE_CUBE_ID = 1
EXPECTED_TARGET_ID = 2            # large red sphere
EXPECTED_TARGET_MATERIAL = "metal"
EXPECTED_DISTRACTOR_MATERIAL = "rubber"


# =====================================================================
# ILP inference
# =====================================================================

@pytest.mark.gurobi
def test_iotaL_target_object_selection(program, dataset):
    """ILP should select object 2 as THE large red sphere right of the small blue cube."""
    from graph import (
        object_node, small, large, red, green, blue,
        cube, sphere, cylinder, right_of, left_of, material,
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        obj_nodes = [n for n in datanode.getChildDataNodes()
                     if n.ontologyNode == object_node]
        assert len(obj_nodes) == 3

        concepts = (small, large, red, green, blue,
                    cube, sphere, cylinder, right_of, left_of, material)
        datanode.inferILPResults(*concepts, fun=None, minimizeObjective=False)

        results = {}
        for on in obj_nodes:
            oid = on.getAttribute("index").item()
            results[oid] = {
                "small":    on.getAttribute(small,    "ILP").item() > 0,
                "large":    on.getAttribute(large,    "ILP").item() > 0,
                "red":      on.getAttribute(red,      "ILP").item() > 0,
                "blue":     on.getAttribute(blue,     "ILP").item() > 0,
                "cube":     on.getAttribute(cube,     "ILP").item() > 0,
                "sphere":   on.getAttribute(sphere,   "ILP").item() > 0,
            }

        print("\n=== ILP Results ===")
        for oid, a in results.items():
            print(f"  Object {oid}: {a}")

        # THE small blue cube
        sbc = [o for o, a in results.items()
               if a["small"] and a["blue"] and a["cube"]]
        assert EXPECTED_SMALL_BLUE_CUBE_ID in sbc

        # THE large red sphere (target)
        lrs = [o for o, a in results.items()
               if a["large"] and a["red"] and a["sphere"]]
        assert EXPECTED_TARGET_ID in lrs


@pytest.mark.gurobi
def test_queryL_material_selection(program, dataset):
    """queryL should identify object 2 as metal."""
    from graph import (
        object_node, small, large, red, green, blue,
        cube, sphere, cylinder, right_of, left_of, material,
    )

    for datanode in program.populate(dataset=dataset):
        obj_nodes = [n for n in datanode.getChildDataNodes()
                     if n.ontologyNode == object_node]

        concepts = (small, large, red, green, blue,
                    cube, sphere, cylinder, right_of, left_of, material)
        datanode.inferILPResults(*concepts, fun=None, minimizeObjective=False)

        mat_results = {}
        for on in obj_nodes:
            oid = on.getAttribute("index").item()
            mt = on.getAttribute(material, "ILP")
            if mt is not None and len(mt) == 2:
                idx = mt.argmax().item()
                mat_results[oid] = material.get_value(idx)
            else:
                mat_results[oid] = None

        print("\n=== Material results ===")
        for oid, m in mat_results.items():
            print(f"  Object {oid}: {m}")

        assert mat_results[EXPECTED_TARGET_ID] == EXPECTED_TARGET_MATERIAL
        assert mat_results[3] == EXPECTED_DISTRACTOR_MATERIAL


@pytest.mark.gurobi
def test_spatial_relations(program, dataset):
    """Verify right_of(2,1) is asserted by ILP."""
    from graph import (
        object_node, pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder, right_of, left_of, material,
    )

    for datanode in program.populate(dataset=dataset):
        concepts = (small, large, red, green, blue,
                    cube, sphere, cylinder, right_of, left_of, material)
        datanode.inferILPResults(*concepts, fun=None, minimizeObjective=False)

        pair_nodes = [n for n in datanode.getChildDataNodes()
                      if n.ontologyNode == pair]
        ro_pairs = []
        for pn in pair_nodes:
            a1 = pn.relationLinks.get(rel_arg1.name, [])
            a2 = pn.relationLinks.get(rel_arg2.name, [])
            if a1 and a2:
                a1id = a1[0].getAttribute("index").item()
                a2id = a2[0].getAttribute("index").item()
                if pn.getAttribute(right_of, "ILP").item() > 0:
                    ro_pairs.append((a1id, a2id))

        print(f"\n  right_of pairs: {ro_pairs}")
        assert (EXPECTED_TARGET_ID, EXPECTED_SMALL_BLUE_CUBE_ID) in ro_pairs


# =====================================================================
# Verification
# =====================================================================

def test_verifyResultsLC(program, dataset):
    """verifyResultsLC should return results for all constraints."""
    for datanode in program.populate(dataset=dataset):
        results = datanode.verifyResultsLC(key="/local/argmax")
        print("\n=== verifyResultsLC ===")
        for lc_name, (sat, total, rate) in results.items():
            print(f"  {lc_name}: {sat}/{total} = {rate:.2%}")
        assert len(results) > 0


def test_verifySingleConstraint_iotaL(program, dataset):
    """Verify each iotaL constraint individually."""
    from graph import the_small_blue_cube, the_target_object

    for datanode in program.populate(dataset=dataset):
        for lc in [the_small_blue_cube, the_target_object]:
            sat, total, rate = datanode.verifySingleConstraint(
                lc.name, key="/local/argmax"
            )
            print(f"  {lc.name}: {sat}/{total} = {rate:.2%}")


def test_verifySingleConstraint_queryL(program, dataset):
    """Verify queryL constraint individually."""
    from graph import the_material_answer

    for datanode in program.populate(dataset=dataset):
        sat, total, rate = datanode.verifySingleConstraint(
            the_material_answer.name, key="/local/argmax"
        )
        print(f"  {the_material_answer.name}: {sat}/{total} = {rate:.2%}")


# =====================================================================
# Loss calculation
# =====================================================================

@pytest.fixture(scope="module")
def program_with_labels():
    """Program with constraint-label sensors for loss tests."""
    import torch
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import (
        EdgeSensor, CompositionCandidateSensor,
    )
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor

    from graph import (
        graph, image, object_node, image_contains_object,
        pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder,
        right_of, left_of, material,
        the_small_blue_cube, the_target_object, the_material_answer,
    )
    from .sensor import (
        SmallLearner, LargeLearner,
        RedLearner, GreenLearner, BlueLearner,
        CubeLearner, SphereLearner, CylinderLearner,
        MaterialEnumLearner,
    )

    graph.detach()

    image["index"] = ReaderSensor(keyword="image")
    object_node["index"] = ReaderSensor(keyword="objects")
    object_node[image_contains_object] = EdgeSensor(
        object_node["index"], image["index"],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1),
    )
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node["index"],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True,
    )

    object_node[small]    = SmallLearner("index")
    object_node[large]    = LargeLearner("index")
    object_node[red]      = RedLearner("index")
    object_node[green]    = GreenLearner("index")
    object_node[blue]     = BlueLearner("index")
    object_node[cube]     = CubeLearner("index")
    object_node[sphere]   = SphereLearner("index")
    object_node[cylinder] = CylinderLearner("index")
    object_node[material] = MaterialEnumLearner("index")

    def _read_spatial(rel_arg1_ref, rel_arg2_ref, keyword):
        def _fn(*_, data, datanode):
            a1 = datanode.relationLinks[rel_arg1_ref.name][0].getAttribute("index").item()
            a2 = datanode.relationLinks[rel_arg2_ref.name][0].getAttribute("index").item()
            if (a1, a2) in data:
                return torch.tensor([0, 1])
            return torch.tensor([1, 0])
        return _fn

    pair[right_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="right_of",
        forward=_read_spatial(rel_arg1, rel_arg2, "right_of"),
    )
    pair[left_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword="left_of",
        forward=_read_spatial(rel_arg1, rel_arg2, "left_of"),
    )

    # --- constraint label sensors ---
    graph.constraint[the_small_blue_cube] = ReaderSensor(
        keyword="the_small_blue_cube_label", is_constraint=True, label=True,
    )
    graph.constraint[the_target_object] = ReaderSensor(
        keyword="the_target_object_label", is_constraint=True, label=True,
    )
    graph.constraint[the_material_answer] = ReaderSensor(
        keyword="the_material_answer_label", is_constraint=True, label=True,
    )

    return LearningBasedProgram(
        graph, PoiModel, poi=[image, object_node, pair], device="cpu",
    )


@pytest.fixture(scope="module")
def dataset_with_labels():
    from .reader import SceneReader

    class ReaderWithLabels(SceneReader):
        def run(self):
            for item in super().run():
                item["the_small_blue_cube_label"] = 1.0
                item["the_target_object_label"]   = 1.0
                item["the_material_answer_label"] = 1.0
                yield item

    return ReaderWithLabels().run()


def test_calculateLcLoss(program_with_labels, dataset_with_labels):
    """Differentiable loss for iotaL + queryL constraints."""
    for datanode in program_with_labels.populate(
        dataset=dataset_with_labels, device="cpu"
    ):
        loss_dict = datanode.calculateLcLoss(tnorm="P", sample=False)
        print("\n=== calculateLcLoss ===")
        for name, ld in loss_dict.items():
            v = ld.get("loss")
            val = v.item() if hasattr(v, "item") else v
            print(f"  {name}: {val}")
        assert len(loss_dict) > 0


def test_calculateLcLoss_sampling(program_with_labels, dataset_with_labels):
    """Sample-based loss for iotaL + queryL constraints."""
    for datanode in program_with_labels.populate(
        dataset=dataset_with_labels, device="cpu"
    ):
        loss_dict = datanode.calculateLcLoss(tnorm="P", sample=True, sampleSize=10)
        print("\n=== calculateLcLoss (sampling) ===")
        for name, ld in loss_dict.items():
            v = ld.get("loss")
            if isinstance(v, list):
                print(f"  {name}: {v}")
            else:
                val = v.item() if hasattr(v, "item") else v
                print(f"  {name}: {val}")
        assert len(loss_dict) > 0