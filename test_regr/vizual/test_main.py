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

    from .graph import (
        graph, image, object_node, image_contains_object,
        pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder,
        right_of, material,
    )
    from .sensor import (
        SmallLearner, LargeLearner,
        RedLearner, GreenLearner, BlueLearner,
        CubeLearner, SphereLearner, CylinderLearner,
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
# Single comprehensive debug test
# =====================================================================

@pytest.mark.gurobi
def test_iotaL_target_object_selection(program, dataset):
    """ILP should select object 2 as THE large red sphere right of the small blue cube."""
    from .graph import (
        object_node, pair, rel_arg1, rel_arg2,
        small, large, red, green, blue,
        cube, sphere, cylinder, right_of, material,
    )

    concepts = (small, large, red, green, blue,
                cube, sphere, cylinder, right_of, material)

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        # Debug: show all child datanodes and their types
        all_children = list(datanode.getChildDataNodes())
        print(f"\n=== All child datanodes ({len(all_children)}) ===")
        for c in all_children:
            print(f"  {c.ontologyNode.name}: {c}")

        obj_nodes = [n for n in all_children if n.ontologyNode == object_node]
        pair_nodes = [n for n in all_children if n.ontologyNode == pair]
        print(f"\n  Object nodes: {len(obj_nodes)}")
        print(f"  Pair nodes: {len(pair_nodes)}")
        assert len(obj_nodes) == 3

        # Debug: show pair node details before ILP
        for pn in pair_nodes[:3]:
            rl = {k: len(v) for k, v in pn.relationLinks.items()}
            print(f"  Pair {pn}: relationLinks keys={list(pn.relationLinks.keys())}")
            # Show right_of local predictions
            ro_local = pn.getAttribute(right_of)
            print(f"    right_of local: {ro_local}")

        datanode.inferILPResults(*concepts, fun=None, minimizeObjective=False)

        # Check object attributes
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

        print("\n=== ILP Results (objects) ===")
        for oid, a in results.items():
            print(f"  Object {oid}: {a}")

        # Check material
        print("\n=== Material Results ===")
        for on in obj_nodes:
            oid = on.getAttribute("index").item()
            mt = on.getAttribute(material, "ILP")
            mt_local = on.getAttribute(material)
            print(f"  Object {oid}: ILP={mt}, local={mt_local}")

        # Check spatial relations
        print("\n=== Spatial Relations ===")
        for pn in pair_nodes:
            a1 = pn.relationLinks.get(rel_arg1.name, [])
            a2 = pn.relationLinks.get(rel_arg2.name, [])
            if a1 and a2:
                a1id = a1[0].getAttribute("index").item()
                a2id = a2[0].getAttribute("index").item()
                ro_ilp = pn.getAttribute(right_of, "ILP")
                ro_local = pn.getAttribute(right_of)
                print(f"  Pair ({a1id},{a2id}): right_of ILP={ro_ilp}, local={ro_local}")

        # THE small blue cube
        sbc = [o for o, a in results.items()
               if a["small"] and a["blue"] and a["cube"]]
        assert EXPECTED_SMALL_BLUE_CUBE_ID in sbc

        # THE large red sphere (target)
        lrs = [o for o, a in results.items()
               if a["large"] and a["red"] and a["sphere"]]
        assert EXPECTED_TARGET_ID in lrs
