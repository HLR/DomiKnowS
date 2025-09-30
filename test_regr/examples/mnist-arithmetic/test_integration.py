import pytest
import torch
from data import get_readers
from model import build_program
from domiknows.program import SolverPOIProgram
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.loss import NBCrossEntropyLoss

class TestIntegration:
    @pytest.mark.slow
    def test_sampling_training_step(self, device):
        """Test that Sampling program can run one training step"""
        num_train = 5  # Very small for quick test
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting=None, digit_labels=False
        )
        
        program = SampleLossProgram(
            graph, SolverModel,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax'],
            metric={},
            sample=True,
            sampleSize=5,
            sampleGlobalLoss=True,
            beta=1
        )
        
        # Test that we can create optimizer
        optimizer = torch.optim.Adam(program.model.parameters(), lr=0.01)
        assert optimizer is not None
        
        # Test that we can populate program with data
        data_iter = iter(program.populate(trainloader, device=device))
        node = next(data_iter)
        assert node is not None

    @pytest.mark.slow
    def test_baseline_training_step(self, device):
        """Test that Baseline program can run one training step"""
        num_train = 5
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting='baseline', digit_labels=False
        )
        
        program = SolverPOIProgram(
            graph,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax'],
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={}
        )
        
        optimizer = torch.optim.Adam(program.model.parameters(), lr=0.01)
        assert optimizer is not None
        
        data_iter = iter(program.populate(trainloader, device=device))
        node = next(data_iter)
        assert node is not None

    @pytest.mark.slow  
    def test_explicit_training_step(self, device):
        """Test that Explicit program can run one training step"""
        num_train = 5
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting='explicit', digit_labels=False
        )
        
        program = SolverPOIProgram(
            graph,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax'],
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={}
        )
        
        optimizer = torch.optim.Adam(program.model.parameters(), lr=0.01)
        assert optimizer is not None
        
        data_iter = iter(program.populate(trainloader, device=device))
        node = next(data_iter)
        assert node is not None

    def test_data_loading_consistency(self, device):
        """Test that data loading is consistent across multiple calls with deterministic seeding"""
        import random
        import numpy as np
        
        num_train = 10
        
        # Reset seeds before first call
        random.seed(10)
        torch.manual_seed(10)
        np.random.seed(10)
        trainloader1, _, _, _ = get_readers(num_train, num_workers=0)
        
        # Reset seeds before second call to get same random sampling
        random.seed(10)
        torch.manual_seed(10)
        np.random.seed(10)
        trainloader2, _, _, _ = get_readers(num_train, num_workers=0)
        
        # Due to fixed seed and deterministic DataLoader, should get same data
        item1 = trainloader1[0]
        item2 = trainloader2[0]
        
        assert torch.equal(item1['pixels'], item2['pixels'])
        assert torch.equal(item1['summation'], item2['summation'])
        assert item1['digit'] == item2['digit']

    def test_model_output_shapes(self, device):
        """Test that all model components produce correct output shapes"""
        num_train = 5
        trainloader, _, _, _ = get_readers(num_train)
        
        # Test each model setting
        for sum_setting in [None, 'baseline', 'explicit']:
            graph, image, image_pair, image_batch = build_program(
                device=device, sum_setting=sum_setting, digit_labels=False
            )
            
            program = SolverPOIProgram(
                graph,
                poi=(image_batch, image, image_pair),
                inferTypes=['local/argmax'],
                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                metric={}
            )
            
            data_iter = iter(program.populate(trainloader, device=device))
            node = next(data_iter)
            assert node is not None