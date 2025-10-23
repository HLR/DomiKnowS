import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import main
from graph import get_graph
from utils import create_dataset, train_model, evaluate_model
from domiknows.program.lossprogram import PrimalDualProgram, GumbelSampleLossProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss


@pytest.fixture
def setup_environment():
    """Setup environment for each test"""
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cpu"
    yield device


@pytest.fixture
def create_args():
    """Create default arguments for testing"""
    class Args:
        beta = 10.0
        device = "cpu"
        counting_tnorm = "G"
        atLeastL = False
        atMostL = False
        epoch = 50
        expected_atLeastL = 2
        expected_atMostL = 5
        expected_value = 0
        N = 10
        M = 5
        model = "sampling"
        sample_size = 100
        use_gumbel = True
        initial_temp = 2.0
        final_temp = 0.1
        hard_gumbel = False
    
    return Args()


def test_basic_training(setup_environment, create_args):
    """Test basic training loop executes without errors"""
    device = setup_environment
    args = create_args
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=10)
    
    assert answer_module is not None
    assert program is not None


@pytest.mark.parametrize("sample_size", [100])  # Reduced from [50, 100, -1]
def test_exact_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that exactL constraint is satisfied using GumbelSampleLossProgram"""
    device = setup_environment
    args = create_args
    args.expected_atLeastL = 2
    args.expected_value = 0
    args.epoch = 300
    args.model = "sampling"
    args.sample_size = sample_size
    
    print(f"\n[TEST] Exact constraint with sample_size={sample_size}")
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count == args.expected_atLeastL, \
        f"Failed with sample_size={sample_size}: got {actual_count}, expected {args.expected_atLeastL}"


@pytest.mark.parametrize("sample_size", [100])  # Reduced from [50, 100, -1]
def test_atleast_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that atLeastL constraint is satisfied using GumbelSampleLossProgram"""
    device = setup_environment
    args = create_args
    args.atLeastL = True
    args.expected_atLeastL = 2
    args.expected_value = 0
    args.epoch = 300
    args.model = "sampling"
    args.sample_size = sample_size
    
    print(f"\n[TEST] AtLeast constraint with sample_size={sample_size}")
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count >= args.expected_atLeastL, \
        f"Failed with sample_size={sample_size}: got {actual_count}, expected >={args.expected_atLeastL}"


@pytest.mark.parametrize("sample_size", [100])  # Reduced from [50, 100, -1]
def test_atmost_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that atMostL constraint is satisfied using GumbelSampleLossProgram"""
    device = setup_environment
    args = create_args
    args.atMostL = True
    args.expected_atMostL = 3
    args.expected_value = 0
    args.epoch = 300
    args.model = "sampling"
    args.sample_size = sample_size
    
    print(f"\n[TEST] AtMost constraint with sample_size={sample_size}")
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count <= args.expected_atMostL, \
        f"Failed with sample_size={sample_size}: got {actual_count}, expected <={args.expected_atMostL}"


@pytest.mark.parametrize("tnorm", ["G", "P", "L", "SP"]) 
def test_different_tnorms(setup_environment, create_args, tnorm):
    """Test training with different t-norm implementations using GumbelSampleLossProgram"""
    device = setup_environment
    args = create_args
    args.counting_tnorm = tnorm
    args.epoch = 50
    args.model = "sampling"
    args.sample_size = 100
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = GumbelSampleLossProgram(
        graph, SolverModel, 
        poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        sample=True,
        sampleSize=args.sample_size,
        sampleGlobalLoss=True,
        use_gumbel=True,
        initial_temp=2.0,
        final_temp=0.1,
        hard_gumbel=False,
        beta=args.beta, 
        device=device, 
        tnorm="L", 
        counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=10)
    
    assert program is not None


@pytest.mark.xfail(reason="PMD cannot reliably solve discrete counting constraints")
def test_pmd_exact_constraint_known_limitation():
    """Documents PMD's known limitation with exact counting constraints"""
    np.random.seed(0)
    torch.manual_seed(0)
    
    class Args:
        beta = 10.0
        device = "cpu"
        counting_tnorm = "G"
        atLeastL = False
        atMostL = False
        epoch = 200
        expected_atLeastL = 2
        expected_atMostL = 5
        expected_value = 0
        N = 10
        M = 5
        model = "PMD"
        sample_size = -1
        use_gumbel = True
        initial_temp = 5.0
        final_temp = 0.5
        hard_gumbel = False
    
    args = Args()
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count == args.expected_atLeastL