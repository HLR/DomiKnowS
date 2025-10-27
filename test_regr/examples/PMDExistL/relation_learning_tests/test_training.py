import pytest
import torch
import sys
from pathlib import Path

# Add the test directory to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent))

import tempfile
import shutil


class TestTraining:
    """Test suite for relation learning training pipeline"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test directory and temporary workspace"""
        self.test_dir = Path(__file__).parent
        self.original_dir = Path.cwd()
        
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_import_modules(self):
        """Test that required modules can be imported"""
        try:
            import domiknows
            from domiknows.program.model.base import Mode
            from domiknows.graph import Graph, Concept, Relation
            from utils import create_dataset_relation
            from graph_rel import get_graph
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

    def test_cuda_availability(self):
        """Verify CUDA is available (skip if not available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in this environment")
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_dataset_creation(self):
        """Test dataset creation functionality"""
        from utils import create_dataset_relation
        import argparse
        
        # Create minimal args
        args = argparse.Namespace(
            max_property=3,
            max_relation=1,
            constraint_2_existL=False,
            use_andL=False
        )
        
        # Create small dataset
        train, test, all_label_test = create_dataset_relation(
            args, N=10, M=2, K=8, read_data=False
        )
        
        assert len(train) > 0, "Training set is empty"
        assert len(test) > 0, "Test set is empty"
        assert len(all_label_test) == len(test), "Label count mismatch"
        
        # Check dataset structure
        sample = train[0]
        assert "all_obj" in sample
        assert "obj_index" in sample
        assert "obj_emb" in sample
        assert "condition_label" in sample
        assert "logic_str" in sample

    def test_graph_creation(self):
        """Test graph construction"""
        from graph_rel import get_graph
        import argparse
        
        args = argparse.Namespace()
        
        result = get_graph(args)
        
        # Unpack the result tuple
        assert len(result) == 15, "Graph should return 15 components"
        
        graph, scene, obj, scene_contain_obj, relation_obj1_obj2, obj1, obj2, \
            is_cond1, is_cond2, is_cond3, is_cond4, \
            is_relation1, is_relation2, is_relation3, is_relation4 = result
        
        assert graph is not None
        assert scene is not None
        assert obj is not None

    def test_minimal_training_run(self, mock_tqdm):
        """Test minimal training without subprocess"""
        import argparse
        import numpy as np
        import random
        from utils import create_dataset_relation
        from graph_rel import get_graph
        from domiknows.program.lossprogram import InferenceProgram
        from domiknows.program.model.pytorch import SolverModel
        from domiknows.sensor.pytorch import ModuleLearner
        from domiknows.sensor.pytorch.sensors import ReaderSensor
        from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
        
        # Set seeds
        random.seed(380)
        np.random.seed(380)
        torch.manual_seed(380)
        
        # Create minimal args
        args = argparse.Namespace(
            N=20,
            lr=1e-4,
            epoch=1,
            constraint_2_existL=False,
            evaluate=False,
            use_andL=False,
            max_property=3,
            max_relation=1,
            load_save="",
            save_file=""
        )
        
        # Create dataset
        train, test, all_label_test = create_dataset_relation(
            args, N=20, M=2, K=8, read_data=False
        )
        
        # Get graph
        (graph, scene, objects, scene_contain_obj, relation_obj1_obj2, obj1, obj2,
         is_cond1, is_cond2, is_cond3, is_cond4,
         is_relation1, is_relation2, is_relation3, is_relation4) = get_graph(args)
        
        # Setup sensors (simplified version)
        scene["all_obj"] = ReaderSensor(keyword="all_obj")
        objects["obj_index"] = ReaderSensor(keyword="obj_index")
        objects["obj_emb"] = ReaderSensor(keyword="obj_emb")
        objects[scene_contain_obj] = EdgeSensor(
            objects["obj_index"], 
            scene["all_obj"],
            relation=scene_contain_obj,
            forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1)
        )
        
        # Simple module for testing
        class SimpleModule(torch.nn.Module):
            def __init__(self, size):
                super().__init__()
                self.layer = torch.nn.Sequential(
                    torch.nn.Linear(size, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 2)
                )
                self.softmax = torch.nn.Softmax(dim=1)
            
            def forward(self, p):
                return self.softmax(self.layer(p))
        
        objects[is_cond1] = ModuleLearner("obj_emb", module=SimpleModule(size=8))
        
        # Add logic labels
        for i in range(len(train)):
            train[i]["logic_label"] = torch.LongTensor([bool(train[i]['condition_label'][0])])
        
        # Compile logic
        train = graph.compile_logic(train, logic_keyword='logic_str', logic_label_keyword='logic_label')
        
        # Create program
        program = InferenceProgram(
            graph, 
            SolverModel,
            poi=[scene, objects, is_cond1, graph.constraint],
            tnorm="G", 
            inferTypes=['local/argmax']
        )
        
        # Just verify program can be created and evaluated
        # Don't actually train to keep test fast
        assert program is not None
        
        # Test evaluation works
        acc = program.evaluate_condition(train)
        assert isinstance(acc, (int, float))
        assert 0 <= acc <= 100

    def test_model_persistence_structure(self):
        """Test that model saving structure works"""
        from pathlib import Path
        
        test_save_path = Path(self.temp_dir) / "test_model.pth"
        
        # Create a simple tensor to save
        test_data = {"test": torch.randn(10, 10)}
        torch.save(test_data, test_save_path)
        
        assert test_save_path.exists()
        
        # Load it back
        loaded_data = torch.load(test_save_path)
        assert "test" in loaded_data
        assert torch.allclose(test_data["test"], loaded_data["test"])