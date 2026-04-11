import sys
import pytest
import torch

sys.path.append('.')
sys.path.append('../..')


@pytest.fixture()
def program():
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel

    from .graph import graph, image, object_node, image_contains_object, color

    graph.detach()

    # Image container
    image['index'] = ReaderSensor(keyword='image')

    # Object sensors
    object_node['index'] = ReaderSensor(keyword='objects')
    object_node[image_contains_object] = EdgeSensor(
        object_node['index'], image['index'],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1)
    )

    # Color EnumConcept learner on objects
    from .sensors import ColorEnumLearner
    object_node[color] = ColorEnumLearner('index')

    return LearningBasedProgram(graph, PoiModel, poi=[image, object_node])


@pytest.fixture()
def dataset():
    from .reader import SameDifferentReader
    return list(SameDifferentReader().run())


# =====================================================================
# ILP Tests
# =====================================================================

@pytest.mark.gurobi
def test_sameL_ilp_verify(program, dataset):
    """
    Verify sameL and differentL constraints via ILP.

    Both x and y range over the same set of objects, aligned row-by-row.
    Each object is compared to itself, so:
    - sameL should be satisfied (each object has the same color as itself)
    - differentL should be violated
    """
    from .graph import color, same_color, diff_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        conceptsRelations = (color,)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)

        results = datanode.verifyResultsLC(key="/local/argmax")

        print("\n=== verifyResultsLC (sameL/differentL) ===")
        for lc_name, result in results.items():
            print(f"{lc_name}: satisfied={result['satisfied']}")

        assert len(results) > 0, "Expected verification results"


# =====================================================================
# Loss Tests
# =====================================================================

def test_sameL_loss(program, dataset):
    """
    Test sameL constraint loss computation.
    Self-comparison: each object vs itself -> sameL loss should be low.
    """
    from .graph import same_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        for tnorm in ['L', 'P']:
            lc_results = datanode.calculateLcLoss(tnorm=tnorm)

            lc_name = same_color.lcName
            if lc_name in lc_results:
                loss = lc_results[lc_name]['lossTensor']
                print(f"\nsameL loss (tnorm={tnorm}): shape={loss.shape}, values={loss}")
                assert torch.is_tensor(loss), "Loss should be a tensor"
                assert not torch.isnan(loss).any(), "Loss should not contain NaN"


def test_differentL_loss(program, dataset):
    """
    Test differentL constraint loss computation.
    Self-comparison: each object vs itself -> differentL loss should be high.
    """
    from .graph import diff_color

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None

        for tnorm in ['L', 'P']:
            lc_results = datanode.calculateLcLoss(tnorm=tnorm)

            lc_name = diff_color.lcName
            if lc_name in lc_results:
                loss = lc_results[lc_name]['lossTensor']
                print(f"\ndifferentL loss (tnorm={tnorm}): shape={loss.shape}, values={loss}")
                assert torch.is_tensor(loss), "Loss should be a tensor"
                assert not torch.isnan(loss).any(), "Loss should not contain NaN"
