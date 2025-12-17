import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


@pytest.fixture(scope="module")
def program():
    import torch

    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel

    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
    
    from .graph import (
        graph, image, object_node, image_contains_object, pair, rel_arg1, rel_arg2,
        big, large, brown, cylinder, sphere, right_of, left_of, material
    )
    from .sensor import (
        BigLearner, LargeLearner, BrownLearner, CylinderLearner, SphereLearner,
        MaterialLearner
    )

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
    
    # Pair sensors - composition of two objects
    # forward returns True/False for candidate validity check
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node['index'],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True  # All pairs are valid candidates
    )

    # Object property learners
    object_node[big] = BigLearner('index')
    object_node[large] = LargeLearner('index')
    object_node[brown] = BrownLearner('index')
    object_node[cylinder] = CylinderLearner('index')
    object_node[sphere] = SphereLearner('index')
    object_node[material] = MaterialLearner('index')
    
    # Spatial relation sensors using DataNodeReaderSensor
    def read_right_of(*_, data, datanode):
        """Check if pair (arg1, arg2) is in the right_of set"""
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
        # Return [0, 1] if in right_of, else [1, 0]
        if (arg1_id, arg2_id) in data:
            return torch.tensor([0, 1])
        else:
            return torch.tensor([1, 0])
    
    def read_left_of(*_, data, datanode):
        """Check if pair (arg1, arg2) is in the left_of set"""
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
        # Return [0, 1] if in left_of, else [1, 0]
        if (arg1_id, arg2_id) in data:
            return torch.tensor([0, 1])
        else:
            return torch.tensor([1, 0])

    pair[right_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword='right_of',
        forward=read_right_of
    )
    pair[left_of] = DataNodeReaderSensor(
        rel_arg1.reversed, rel_arg2.reversed,
        keyword='left_of',
        forward=read_left_of
    )

    return LearningBasedProgram(graph, PoiModel, poi=[image, object_node, pair])


@pytest.fixture(scope="module")
def dataset():
    from .reader import VisualQAReader
    return VisualQAReader().run()


# Expected values based on test data:
# Object 1 = brown cylinder (THE brown cylinder)
# Object 2 = large brown sphere (THE large brown sphere)
# Object 3 = big target object, right of 1, left of 2 (THE target object)
# Object 4 = other object

EXPECTED_BROWN_CYLINDER_ID = 1
EXPECTED_LARGE_BROWN_SPHERE_ID = 2
EXPECTED_TARGET_OBJECT_ID = 3


@pytest.mark.gurobi  
def test_iotaL_target_object_selection(program, dataset):
    """
    Test that the_target_object iotaL constraint correctly selects object 3.
    
    Uses ILP inference and then verifies the selection matches expected.
    """
    from .graph import (
        graph, object_node, pair, 
        big, large, brown, cylinder, sphere, right_of, left_of
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None
        
        object_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == object_node]
        assert len(object_nodes) == 4
        
        # Run ILP inference
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)
        
        # Collect ILP results
        results = {}
        for obj_node in object_nodes:
            obj_id = obj_node.getAttribute('index').item()
            results[obj_id] = {
                'big': obj_node.getAttribute(big, 'ILP').item() > 0,
                'large': obj_node.getAttribute(large, 'ILP').item() > 0,
                'brown': obj_node.getAttribute(brown, 'ILP').item() > 0,
                'cylinder': obj_node.getAttribute(cylinder, 'ILP').item() > 0,
                'sphere': obj_node.getAttribute(sphere, 'ILP').item() > 0,
            }
        
        print("\n=== ILP Results per Object ===")
        for obj_id, attrs in results.items():
            print(f"Object {obj_id}: {attrs}")
        
        # Find THE brown cylinder (brown AND cylinder)
        brown_cylinders = [oid for oid, attrs in results.items() 
                          if attrs['brown'] and attrs['cylinder']]
        print(f"\nBrown cylinders found: {brown_cylinders}")
        print(f"Expected: {EXPECTED_BROWN_CYLINDER_ID}")
        
        # Find THE large brown sphere (large AND brown AND sphere)
        large_brown_spheres = [oid for oid, attrs in results.items()
                               if attrs['large'] and attrs['brown'] and attrs['sphere']]
        print(f"Large brown spheres found: {large_brown_spheres}")
        print(f"Expected: {EXPECTED_LARGE_BROWN_SPHERE_ID}")
        
        # Find THE big object (target)
        big_objects = [oid for oid, attrs in results.items() if attrs['big']]
        print(f"Big objects found: {big_objects}")
        print(f"Expected target: {EXPECTED_TARGET_OBJECT_ID}")
        
        # Verify expectations
        assert EXPECTED_BROWN_CYLINDER_ID in brown_cylinders, \
            f"Expected object {EXPECTED_BROWN_CYLINDER_ID} to be brown cylinder"
        assert EXPECTED_LARGE_BROWN_SPHERE_ID in large_brown_spheres, \
            f"Expected object {EXPECTED_LARGE_BROWN_SPHERE_ID} to be large brown sphere"
        assert EXPECTED_TARGET_OBJECT_ID in big_objects, \
            f"Expected object {EXPECTED_TARGET_OBJECT_ID} to be big"


@pytest.mark.gurobi
def test_iotaL_spatial_relations(program, dataset):
    """
    Test that spatial relations in iotaL constraints are correctly evaluated.
    
    Verifies:
    - Object 3 is right_of object 1 (the brown cylinder)
    - Object 3 is left_of object 2 (the large brown sphere)
    """
    from .graph import (
        graph, object_node, pair, rel_arg1, rel_arg2,
        big, large, brown, cylinder, sphere, right_of, left_of
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None
        
        pair_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == pair]
        
        # Run ILP inference
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)
        
        # Find right_of and left_of relations
        right_of_pairs = []
        left_of_pairs = []
        
        for pair_node in pair_nodes:
            arg1_links = pair_node.relationLinks.get(rel_arg1.name, [])
            arg2_links = pair_node.relationLinks.get(rel_arg2.name, [])
            
            if arg1_links and arg2_links:
                arg1_id = arg1_links[0].getAttribute('index').item()
                arg2_id = arg2_links[0].getAttribute('index').item()
                
                is_right_of = pair_node.getAttribute(right_of, 'ILP').item() > 0
                is_left_of = pair_node.getAttribute(left_of, 'ILP').item() > 0
                
                if is_right_of:
                    right_of_pairs.append((arg1_id, arg2_id))
                if is_left_of:
                    left_of_pairs.append((arg1_id, arg2_id))
        
        print("\n=== Spatial Relations ===")
        print(f"right_of pairs: {right_of_pairs}")
        print(f"left_of pairs: {left_of_pairs}")
        
        # Expected: (3, 1) in right_of - object 3 is right of object 1 (brown cylinder)
        expected_right_of = (EXPECTED_TARGET_OBJECT_ID, EXPECTED_BROWN_CYLINDER_ID)
        print(f"Expected right_of: {expected_right_of}")
        
        # Expected: (3, 2) in left_of - object 3 is left of object 2 (large brown sphere)
        expected_left_of = (EXPECTED_TARGET_OBJECT_ID, EXPECTED_LARGE_BROWN_SPHERE_ID)
        print(f"Expected left_of: {expected_left_of}")
        
        assert expected_right_of in right_of_pairs, \
            f"Expected {expected_right_of} in right_of relations"
        assert expected_left_of in left_of_pairs, \
            f"Expected {expected_left_of} in left_of relations"
        
        print("\nAll spatial relation constraints verified!")