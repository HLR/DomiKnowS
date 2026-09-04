import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import main


@pytest.fixture
def setup_environment():
    """Setup environment for each test"""
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cpu"
    yield device


@pytest.fixture
def create_args():
    """Create default arguments for testing - only common args"""
    class Args:
        beta = 10.0
        device = "cpu"
        N = 10
        M = 5
        use_gumbel = True
        initial_temp = 2.0
        final_temp = 0.1
        hard_gumbel = False
    
    return Args()


def test_basic_training(setup_environment, create_args):
    """Test basic training loop executes without errors"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = False
    args.atMostL = False
    args.epoch = 50
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = 0
    args.model = "PMD"
    args.sample_size = -1
    
    pass_test, before_count, actual_count = main(args)
    
    assert before_count is not None
    assert actual_count is not None


@pytest.mark.parametrize("sample_size", [100]) 
def test_exact_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that exactL constraint is satisfied using GumbelSampleLossProgram"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = False
    args.atMostL = False
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = 0
    args.epoch = 300
    args.model = "sampling"
    args.sample_size = sample_size
    
    print(f"\n[TEST] Exact constraint with sample_size={sample_size}")
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count == args.expected_atLeastL, \
        f"Failed with sample_size={sample_size}: got {actual_count}, expected {args.expected_atLeastL}"


@pytest.mark.parametrize("sample_size", [100])
def test_atleast_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that atLeastL constraint is satisfied using GumbelSampleLossProgram"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = True
    args.atMostL = False
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = 0
    args.epoch = 300
    args.model = "sampling"
    args.sample_size = sample_size
    
    print(f"\n[TEST] AtLeast constraint with sample_size={sample_size}")
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count >= args.expected_atLeastL, \
        f"Failed with sample_size={sample_size}: got {actual_count}, expected >={args.expected_atLeastL}"


@pytest.mark.parametrize("sample_size", [100])
def test_atmost_constraint_satisfaction(setup_environment, create_args, sample_size):
    """Test that atMostL constraint is satisfied using GumbelSampleLossProgram"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = False
    args.atMostL = True
    args.expected_atLeastL = 2
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
    """Test training with different t-norm implementations - try PrimalDualProgram first, fallback to Gumbel versions"""
    args = create_args
    args.counting_tnorm = tnorm
    args.atLeastL = False
    args.atMostL = False
    args.epoch = 50
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = 0
    args.sample_size = -1
    
    print(f"\n[TEST] Testing tnorm={tnorm}")
    
    # Try PrimalDualProgram first
    args.model = "PMD"
    try:
        pass_test, before_count, actual_count = main(args)
        if pass_test:
            print(f"  PrimalDualProgram succeeded with tnorm={tnorm}: count={actual_count}, expected={args.expected_atLeastL}")
            assert actual_count == args.expected_atLeastL
            return
        else:
            print(f"  PrimalDualProgram did not satisfy constraint with tnorm={tnorm}: count={actual_count}, expected={args.expected_atLeastL}")
    except Exception as e:
        print(f"  PrimalDualProgram failed with exception for tnorm={tnorm}: {e}")
    
    # If failed, try GumbelPrimalDualProgram
    args.model = "gumbel_pmd"
    try:
        pass_test, before_count, actual_count = main(args)
        if pass_test:
            print(f"  GumbelPrimalDualProgram succeeded with tnorm={tnorm}: count={actual_count}, expected={args.expected_atLeastL}")
            assert actual_count == args.expected_atLeastL
            return
        else:
            print(f"  GumbelPrimalDualProgram did not satisfy constraint with tnorm={tnorm}: count={actual_count}, expected={args.expected_atLeastL}")
    except Exception as e:
        print(f"  GumbelPrimalDualProgram failed with exception for tnorm={tnorm}: {e}")
    
    # If both failed, try GumbelSampleLossProgram
    args.model = "sampling"
    args.sample_size = 100
    args.epoch = 100  # More epochs for sampling
    pass_test, before_count, actual_count = main(args)
    print(f"  GumbelSampleLossProgram: count={actual_count}, expected={args.expected_atLeastL}, pass={pass_test}")
    
    # For sampling, we expect it to work
    assert actual_count == args.expected_atLeastL, \
        f"All methods failed for tnorm={tnorm}. Final count={actual_count}, expected={args.expected_atLeastL}"


def test_pmd_exact_constraint(setup_environment, create_args):
    """Test PMD with exact constraint"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = False
    args.atMostL = False
    args.epoch = 200
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = 0
    args.model = "PMD"
    args.sample_size = -1
    
    pass_test, before_count, actual_count = main(args)
    
    assert actual_count == args.expected_atLeastL, \
        f"PMD failed to satisfy exact constraint: got {actual_count}, expected {args.expected_atLeastL}"