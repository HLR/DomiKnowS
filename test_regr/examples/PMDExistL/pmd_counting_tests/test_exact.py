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


# PMD test combinations (focused on both constraints)
@pytest.mark.parametrize("counting_tnorm", ["G", "P", "SP", "L"])
@pytest.mark.parametrize("expected_value", [0, 1])
def test_pmd_exact(create_args, counting_tnorm, expected_value):
    """Test PMD with both atLeastL and atMostL constraints (True, True)"""
    # Mark known failing combinations as expected failures
    if (counting_tnorm == "L" and expected_value == 0) or \
       (counting_tnorm == "L" and expected_value == 1) or \
       (counting_tnorm == "G" and expected_value == 1):
        pytest.xfail(
            f"PMD with tnorm={counting_tnorm} and expected_value={expected_value} "
        )
    
    args = create_args
    args.counting_tnorm = counting_tnorm
    args.atLeastL = True
    args.atMostL = True
    args.epoch = 1000
    args.expected_atLeastL = 2
    args.expected_atMostL = 5
    args.expected_value = expected_value
    args.model = "PMD"
    args.sample_size = -1
    
    pass_test, before_count, actual_count = main(args)
    
    assert pass_test, \
        f"PMD test failed: tnorm={counting_tnorm}, " \
        f"expected_value={expected_value}, count={actual_count}"


# Sampling test combinations (focused on both constraints)
@pytest.mark.parametrize("sample_size", [50, 100])
@pytest.mark.parametrize("expected_atLeastL,expected_atMostL", [
    (2, 4),
    (2, 5),
    (3, 5),
])
@pytest.mark.parametrize("expected_value", [0])
def test_sampling_exact(create_args, sample_size, expected_atLeastL, expected_atMostL, expected_value):
    """Test sampling model with both atLeastL and atMostL constraints (True, True)"""
    args = create_args
    args.counting_tnorm = "G"
    args.atLeastL = True
    args.atMostL = True
    args.epoch = 1000
    args.expected_atLeastL = expected_atLeastL
    args.expected_atMostL = expected_atMostL
    args.expected_value = expected_value
    args.model = "sampling"
    args.sample_size = sample_size
    
    pass_test, before_count, actual_count = main(args)
    
    assert pass_test, \
        f"Sampling test failed: sample_size={sample_size}, " \
        f"expected_atLeastL={expected_atLeastL}, expected_atMostL={expected_atMostL}, " \
        f"expected_value={expected_value}, count={actual_count}"