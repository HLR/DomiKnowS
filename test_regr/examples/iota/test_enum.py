import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


@pytest.fixture(scope="module")
def program_enum():
    import torch

    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
    
    from .graph_enum import (
        graph, image, object_node, image_contains_object, pair, rel_arg1, rel_arg2,
        big, large, brown, cylinder, sphere, right_of, left_of, material
    )
    from .sensor_enum import (
        BigLearner, LargeLearner, BrownLearner, CylinderLearner, SphereLearner,
        MaterialEnumLearner
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
    
    # Material EnumConcept learner - outputs [metal_prob, rubber_prob]
    object_node[material] = MaterialEnumLearner('index')
    
    # Spatial relation sensors
    def read_right_of(*_, data, datanode):
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
        device = getattr(datanode, 'current_device', 'cpu')
        if (arg1_id, arg2_id) in data:
            return torch.tensor([0, 1], device=device)
        else:
            return torch.tensor([1, 0], device=device)
    
    def read_left_of(*_, data, datanode):
        arg1_node = datanode.relationLinks[rel_arg1.name][0]
        arg2_node = datanode.relationLinks[rel_arg2.name][0]
        arg1_id = arg1_node.getAttribute('index').item()
        arg2_id = arg2_node.getAttribute('index').item()
        device = getattr(datanode, 'current_device', 'cpu')
        if (arg1_id, arg2_id) in data:
            return torch.tensor([0, 1], device=device)
        else:
            return torch.tensor([1, 0], device=device)

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
def dataset_enum():
    from .reader import VisualQAReader
    return VisualQAReader().run()


EXPECTED_BROWN_CYLINDER_ID = 1
EXPECTED_LARGE_BROWN_SPHERE_ID = 2
EXPECTED_TARGET_OBJECT_ID = 3
EXPECTED_TARGET_MATERIAL = 'metal'


@pytest.mark.gurobi  
def test_enum_iotaL_target_object_selection(program_enum, dataset_enum):
    """Test iotaL with EnumConcept material."""
    from .graph_enum import object_node, big, large, brown, cylinder, sphere, right_of, left_of, material

    for datanode in program_enum.populate(dataset=dataset_enum):
        assert datanode is not None
        
        object_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == object_node]
        assert len(object_nodes) == 4
        
        # Run ILP inference with EnumConcept material
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of, material)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)
        
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
        
        print("\n=== ILP Results per Object (EnumConcept) ===")
        for obj_id, attrs in results.items():
            print(f"Object {obj_id}: {attrs}")
        
        brown_cylinders = [oid for oid, attrs in results.items() 
                          if attrs['brown'] and attrs['cylinder']]
        large_brown_spheres = [oid for oid, attrs in results.items()
                               if attrs['large'] and attrs['brown'] and attrs['sphere']]
        big_objects = [oid for oid, attrs in results.items() if attrs['big']]
        
        assert EXPECTED_BROWN_CYLINDER_ID in brown_cylinders
        assert EXPECTED_LARGE_BROWN_SPHERE_ID in large_brown_spheres
        assert EXPECTED_TARGET_OBJECT_ID in big_objects


@pytest.mark.gurobi
def test_enum_queryL_material_selection(program_enum, dataset_enum):
    """
    Test queryL with EnumConcept material.
    
    The target object (object 3) should have material 'metal' (index 0).
    queryL should return metal as the selected enum value.
    """
    from .graph_enum import object_node, big, large, brown, cylinder, sphere, right_of, left_of, material

    for datanode in program_enum.populate(dataset=dataset_enum):
        assert datanode is not None
        
        object_nodes = [n for n in datanode.getChildDataNodes() if n.ontologyNode == object_node]
        
        conceptsRelations = (big, large, brown, cylinder, sphere, right_of, left_of, material)
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=False)
        
        # Collect material results per object using EnumConcept
        material_results = {}
        for obj_node in object_nodes:
            obj_id = obj_node.getAttribute('index').item()
            
            # For EnumConcept, get the full material tensor
            mat_tensor = obj_node.getAttribute(material, 'ILP')
            
            # Determine which enum value is selected (argmax)
            if mat_tensor is not None and len(mat_tensor) == 2:
                selected_idx = mat_tensor.argmax().item()
                selected_material = material.get_value(selected_idx)
                material_results[obj_id] = {
                    'tensor': mat_tensor.tolist(),
                    'selected': selected_material,
                    'metal_prob': mat_tensor[0].item(),
                    'rubber_prob': mat_tensor[1].item()
                }
            else:
                material_results[obj_id] = {'tensor': None, 'selected': None}
        
        print("\n=== Material EnumConcept Results per Object ===")
        for obj_id, mats in material_results.items():
            print(f"Object {obj_id}: selected={mats['selected']}, tensor={mats.get('tensor')}")
        
        # Verify target object (3) has material 'metal'
        target_material = material_results[EXPECTED_TARGET_OBJECT_ID]
        
        print(f"\n=== queryL Material Test (EnumConcept) ===")
        print(f"Target object ID: {EXPECTED_TARGET_OBJECT_ID}")
        print(f"Expected material: {EXPECTED_TARGET_MATERIAL}")
        print(f"Selected material: {target_material['selected']}")
        
        assert target_material['selected'] == EXPECTED_TARGET_MATERIAL, \
            f"Expected object {EXPECTED_TARGET_OBJECT_ID} to be {EXPECTED_TARGET_MATERIAL}"
        
        # Verify object 4 is rubber
        distractor_material = material_results[4]
        assert distractor_material['selected'] == 'rubber', \
            "Expected object 4 to be rubber"
        
        print("\nqueryL EnumConcept material selection test PASSED!")


def test_enum_iotaL_queryL_verifyResultsLC(program_enum, dataset_enum):
    """Test verifyResultsLC for all constraints with EnumConcept material."""
    for datanode in program_enum.populate(dataset=dataset_enum):
        results = datanode.verifyResultsLC(key="/local/argmax")
        
        print("\n=== verifyResultsLC (EnumConcept) ===")
        for lc_name, (sat_count, total, sat_rate) in results.items():
            print(f"{lc_name}: {sat_count}/{total} = {sat_rate:.2%}")
        
        assert len(results) > 0, "Expected verification results"


def test_enum_queryL_verifySingleConstraint(program_enum, dataset_enum):
    """Test verifySingleConstraint for queryL with EnumConcept."""
    from .graph_enum import the_material_answer

    for datanode in program_enum.populate(dataset=dataset_enum):
        print("\n=== verifySingleConstraint for queryL (EnumConcept) ===")
        
        sat_count, total, sat_rate = datanode.verifySingleConstraint(
            the_material_answer.name, key="/local/argmax"
        )
        print(f"{the_material_answer.name}: {sat_count}/{total} = {sat_rate:.2%}")


@pytest.fixture(scope="module")
def program_enum_with_labels():
    """Program fixture with constraint label sensors for loss calculation."""
    import torch
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
    
    from .graph_enum import (
        graph, image, object_node, image_contains_object, pair, rel_arg1, rel_arg2,
        big, large, brown, cylinder, sphere, right_of, left_of, material,
        the_brown_cylinder, the_large_brown_sphere, the_target_object, the_material_answer
    )
    from .sensor_enum import (
        BigLearner, LargeLearner, BrownLearner, CylinderLearner, SphereLearner,
        MaterialEnumLearner
    )

    graph.detach()

    image['index'] = ReaderSensor(keyword='image')
    object_node['index'] = ReaderSensor(keyword='objects')
    object_node[image_contains_object] = EdgeSensor(
        object_node['index'], image['index'],
        relation=image_contains_object,
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1)
    )
    
    pair[rel_arg1.reversed, rel_arg2.reversed] = CompositionCandidateSensor(
        object_node['index'],
        relations=(rel_arg1.reversed, rel_arg2.reversed),
        forward=lambda *_, **__: True
    )

    object_node[big] = BigLearner('index')
    object_node[large] = LargeLearner('index')
    object_node[brown] = BrownLearner('index')
    object_node[cylinder] = CylinderLearner('index')
    object_node[sphere] = SphereLearner('index')
    object_node[material] = MaterialEnumLearner('index')
    
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

    graph.constraint[the_brown_cylinder] = ReaderSensor(
        keyword='the_brown_cylinder_label', is_constraint=True, label=True
    )
    graph.constraint[the_large_brown_sphere] = ReaderSensor(
        keyword='the_large_brown_sphere_label', is_constraint=True, label=True
    )
    graph.constraint[the_target_object] = ReaderSensor(
        keyword='the_target_object_label', is_constraint=True, label=True
    )
    graph.constraint[the_material_answer] = ReaderSensor(
        keyword='the_material_answer_label', is_constraint=True, label=True
    )

    return LearningBasedProgram(graph, PoiModel, poi=[image, object_node, pair], device='cpu')


@pytest.fixture(scope="module")
def dataset_enum_with_labels():
    """Dataset with constraint labels for loss calculation."""
    from .reader import VisualQAReader
    
    class VisualQAReaderWithLabels(VisualQAReader):
        def run(self):
            for item in super().run():
                item['the_brown_cylinder_label'] = 1.0
                item['the_large_brown_sphere_label'] = 1.0
                item['the_target_object_label'] = 1.0
                item['the_material_answer_label'] = 1.0
                yield item
    
    return VisualQAReaderWithLabels().run()


def test_enum_calculateLcLoss(program_enum_with_labels, dataset_enum_with_labels):
    """Test calculateLcLoss (differentiable) with EnumConcept material."""
    for datanode in program_enum_with_labels.populate(dataset=dataset_enum_with_labels, device='cpu'):
        loss_dict = datanode.calculateLcLoss(tnorm='P', sample=False)
        
        print("\n=== calculateLcLoss (EnumConcept) ===")
        for lc_name, lc_data in loss_dict.items():
            loss_val = lc_data.get('loss')
            if loss_val is not None:
                loss_num = loss_val.item() if hasattr(loss_val, 'item') else loss_val
                print(f"{lc_name}: loss = {loss_num:.4f}")
            else:
                print(f"{lc_name}: loss = None")
        
        assert len(loss_dict) > 0, "Expected loss values"


def test_enum_calculateLcLoss_sampling(program_enum_with_labels, dataset_enum_with_labels):
    """Test calculateLcLoss with sampling for EnumConcept material."""
    for datanode in program_enum_with_labels.populate(dataset=dataset_enum_with_labels, device='cpu'):
        loss_dict = datanode.calculateLcLoss(tnorm='P', sample=True, sampleSize=10)
        
        print("\n=== calculateLcLoss sampling (EnumConcept) ===")
        for lc_name, lc_data in loss_dict.items():
            loss_val = lc_data.get('loss')
            if loss_val is not None:
                if isinstance(loss_val, list):
                    print(f"{lc_name}: loss = {loss_val}")
                else:
                    loss_num = loss_val.item() if hasattr(loss_val, 'item') else loss_val
                    print(f"{lc_name}: loss = {loss_num:.4f}")
            else:
                print(f"{lc_name}: loss = None")
        
        assert len(loss_dict) > 0, "Expected loss values with sampling"