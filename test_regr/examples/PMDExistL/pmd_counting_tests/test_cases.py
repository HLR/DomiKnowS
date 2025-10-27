import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import main


@pytest.fixture
def create_args():
    """Create default arguments for testing"""
    class Args:
        beta = 10.0
        device = "auto"
        N = 10
        M = 8
        use_gumbel = True
        initial_temp = 2.0
        final_temp = 0.1
        hard_gumbel = False
    
    return Args()


# PMD test combinations
@pytest.mark.parametrize("counting_tnorm", ["G", "P", "SP", "L"])
@pytest.mark.parametrize("atLeastL,atMostL", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("expected_value", [0, 1])
def test_pmd_combinations(create_args, counting_tnorm, atLeastL, atMostL, expected_value):
    """Test PMD with various parameter combinations"""
    args = create_args
    args.counting_tnorm = counting_tnorm
    args.atLeastL = atLeastL
    args.atMostL = atMostL
    args.epoch = 1000
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = expected_value
    args.model = "PMD"
    args.sample_size = -1
    
    pass_test, before_count, actual_count = main(args)
    
    assert pass_test, \
        f"PMD test failed: tnorm={counting_tnorm}, atLeastL={atLeastL}, atMostL={atMostL}, " \
        f"expected_value={expected_value}, count={actual_count}"


# Sampling test combinations
@pytest.mark.parametrize("sample_size", [10, 20, 50, 100, 200, -1])
@pytest.mark.parametrize("atLeastL,atMostL,expected_atLeastL,expected_atMostL", [
    (True, False, 1, 3),
    (True, False, 2, 4),
    (True, False, 3, 5),
    (False, True, 1, 3),
    (False, True, 2, 4),
    (False, True, 3, 5),
    (True, True, 1, 3),
    (True, True, 2, 4),
    (True, True, 3, 5),
    (False, False, 1, 3),
    (False, False, 2, 4),
    (False, False, 3, 5),
])
@pytest.mark.parametrize("expected_value", [0, 1])
def test_sampling_combinations(create_args, sample_size, atLeastL, atMostL, 
                                expected_atLeastL, expected_atMostL, expected_value):
    """Test sampling model with various parameter combinations"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = atLeastL
    args.atMostL = atMostL
    args.epoch = 1000
    args.expected_atLeastL = expected_atLeastL
    args.expected_atMostL = expected_atMostL
    args.expected_value = expected_value
    args.model = "sampling"
    args.sample_size = sample_size
    
    pass_test, before_count, actual_count = main(args)
    
    assert pass_test, \
        f"Sampling test failed: sample_size={sample_size}, atLeastL={atLeastL}, atMostL={atMostL}, " \
        f"expected_atLeastL={expected_atLeastL}, expected_atMostL={expected_atMostL}, " \
        f"expected_value={expected_value}, count={actual_count}"