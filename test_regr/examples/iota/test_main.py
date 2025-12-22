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
        big, large, brown, cylinder, sphere, right_of, left_of, material, metal, rubber
    )
    from .sensor import (
        BigLearner, LargeLearner, BrownLearner, CylinderLearner, SphereLearner,
        MetalLearner, RubberLearner
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
    
    # Pair sensors
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node['index'],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True
    )

    # Object property learners
    object_node[big] = BigLearner('index')
    object_node[large] = LargeLearner('index')
    object_node[brown] = BrownLearner('index')
    object_node[cylinder] = CylinderLearner('index')
    object_node[sphere] = SphereLearner('index')
    
    # Material subclass learners
    object_node[metal] = MetalLearner('index')
    object_node[rubber] = RubberLearner('index')
    
    # Spatial relation sensors
    def read_right_of(*_, data, datanode):
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
        if (arg1_id, arg2_id) in data:
            return torch.tensor([0, 1])
        else:
            return torch.tensor([1, 0])
    
    def read_left_of(*_, data, datanode):
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
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


# Expected values based on test data
EXPECTED_BROWN_CYLINDER_ID = 1
EXPECTED_LARGE_BROWN_SPHERE_ID = 2
EXPECTED_TARGET_OBJECT_ID = 3
EXPECTED_TARGET_MATERIAL = 'metal'


@pytest.mark.gurobi  
def test_iotaL_target_object_selection(program, dataset):
    """Test that the_target_object iotaL constraint correctly selects object 3."""
    from .graph import (
        object_node, big, large, brown, cylinder, sphere, right_of, left_of,
        metal, rubber
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None
        
        object_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == object_node]
        assert len(object_nodes) == 4
        
        # Run ILP inference including material subclasses
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of, metal, rubber)
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
        
        # Find THE brown cylinder
        brown_cylinders = [oid for oid, attrs in results.items() 
                          if attrs['brown'] and attrs['cylinder']]
        print(f"\nBrown cylinders found: {brown_cylinders}")
        
        # Find THE large brown sphere
        large_brown_spheres = [oid for oid, attrs in results.items()
                               if attrs['large'] and attrs['brown'] and attrs['sphere']]
        print(f"Large brown spheres found: {large_brown_spheres}")
        
        # Find THE big object (target)
        big_objects = [oid for oid, attrs in results.items() if attrs['big']]
        print(f"Big objects found: {big_objects}")
        
        # Verify expectations
        assert EXPECTED_BROWN_CYLINDER_ID in brown_cylinders
        assert EXPECTED_LARGE_BROWN_SPHERE_ID in large_brown_spheres
        assert EXPECTED_TARGET_OBJECT_ID in big_objects


@pytest.mark.gurobi
def test_iotaL_spatial_relations(program, dataset):
    """Test that spatial relations in iotaL constraints are correctly evaluated."""
    from .graph import (
        object_node, pair, rel_arg1, rel_arg2,
        big, large, brown, cylinder, sphere, right_of, left_of, metal, rubber
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None
        
        pair_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == pair]
        
        # Run ILP inference
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of, metal, rubber)
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
        
        expected_right_of = (EXPECTED_TARGET_OBJECT_ID, EXPECTED_BROWN_CYLINDER_ID)
        expected_left_of = (EXPECTED_TARGET_OBJECT_ID, EXPECTED_LARGE_BROWN_SPHERE_ID)
        
        assert expected_right_of in right_of_pairs
        assert expected_left_of in left_of_pairs
        
        print("\nAll spatial relation constraints verified!")


@pytest.mark.gurobi
def test_queryL_material_selection(program, dataset):
    """
    Test that queryL correctly identifies the material of the target object.
    
    The target object (object 3) should have material 'metal'.
    queryL should return metal as the selected subclass.
    """
    from .graph import (
        object_node, big, large, brown, cylinder, sphere, right_of, left_of,
        material, metal, rubber
    )

    for datanode in program.populate(dataset=dataset):
        assert datanode is not None
        
        object_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == object_node]
        
        # Run ILP inference with all concepts including material subclasses
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of, metal, rubber)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)
        
        # Collect material results per object
        material_results = {}
        for obj_node in object_nodes:
            obj_id = obj_node.getAttribute('index').item()
            
            is_metal = obj_node.getAttribute(metal, 'ILP').item() > 0
            is_rubber = obj_node.getAttribute(rubber, 'ILP').item() > 0
            
            material_results[obj_id] = {
                'metal': is_metal,
                'rubber': is_rubber
            }
        
        print("\n=== Material Results per Object ===")
        for obj_id, mats in material_results.items():
            mat_str = 'metal' if mats['metal'] else ('rubber' if mats['rubber'] else 'none')
            print(f"Object {obj_id}: {mat_str} (metal={mats['metal']}, rubber={mats['rubber']})")
        
        # Verify target object (3) has material 'metal'
        target_material = material_results[EXPECTED_TARGET_OBJECT_ID]
        
        print(f"\n=== queryL Material Test ===")
        print(f"Target object ID: {EXPECTED_TARGET_OBJECT_ID}")
        print(f"Expected material: {EXPECTED_TARGET_MATERIAL}")
        print(f"Target is metal: {target_material['metal']}")
        print(f"Target is rubber: {target_material['rubber']}")
        
        # Assert target object is metal
        assert target_material['metal'] == True, \
            f"Expected object {EXPECTED_TARGET_OBJECT_ID} to be metal"
        assert target_material['rubber'] == False, \
            f"Expected object {EXPECTED_TARGET_OBJECT_ID} to NOT be rubber"
        
        # Verify object 4 is rubber (distractor)
        distractor_material = material_results[4]
        assert distractor_material['rubber'] == True, \
            "Expected object 4 to be rubber"
        assert distractor_material['metal'] == False, \
            "Expected object 4 to NOT be metal"
        
        print("\nqueryL material selection test PASSED!")
        print(f"Correctly identified object {EXPECTED_TARGET_OBJECT_ID} as '{EXPECTED_TARGET_MATERIAL}'")