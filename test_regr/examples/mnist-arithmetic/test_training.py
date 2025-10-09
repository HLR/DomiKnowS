import pytest
import torch
from data import get_readers
from model import build_program
from domiknows.program import SolverPOIProgram
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.loss import NBCrossEntropyLoss

class TestTrainingPrograms:
    def test_sampling_program_creation(self, device, num_train):
        """Test SampleLossProgram creation and basic functionality"""
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
            sampleSize=10,  # Small for testing
            sampleGlobalLoss=True,
            beta=1
        )
        
        assert program is not None

    def test_semantic_program_creation(self, device, num_train):
        """Test Semantic SampleLossProgram creation"""
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
            sampleSize=-1,  # Semantic Sample when -1
            sampleGlobalLoss=True,
            beta=1
        )
        
        assert program is not None

    def test_primal_dual_program_creation(self, device, num_train):
        """Test PrimalDualProgram creation"""
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting=None, digit_labels=False
        )
        
        program = PrimalDualProgram(
            graph, SolverModel,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax'],
            metric={}
        )
        
        assert program is not None

    def test_baseline_program_creation(self, device, num_train):
        """Test baseline SolverPOIProgram creation"""
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
        
        assert program is not None

    def test_explicit_program_creation(self, device, num_train):
        """Test explicit SolverPOIProgram creation"""
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
        
        assert program is not None

    def test_digit_label_program_creation(self, device, num_train):
        """Test digit label SolverPOIProgram creation"""
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting=None, digit_labels=True
        )
        
        program = SolverPOIProgram(
            graph,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax'],
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={}
        )
        
        assert program is not None

    def test_gbi_program_creation(self, device, num_train):
        """Test GBI SolverPOIProgram creation"""
        trainloader, _, _, _ = get_readers(num_train)
        graph, image, image_pair, image_batch = build_program(
            device=device, sum_setting=None, digit_labels=False
        )
        
        program = SolverPOIProgram(
            graph,
            poi=(image_batch, image, image_pair),
            inferTypes=['local/argmax', 'GBI'],
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={}
        )
        
        assert program is not None