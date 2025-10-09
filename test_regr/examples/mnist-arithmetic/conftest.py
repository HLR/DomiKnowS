import pytest
import torch
import random
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    """Set random seeds for reproducibility"""
    random.seed(10)
    torch.manual_seed(10)
    np.random.seed(10)

@pytest.fixture
def device():
    """Provide device for testing"""
    return 'cpu'

@pytest.fixture(params=['Sampling', 'Semantic', 'PrimalDual', 'Explicit', 'DigitLabel', 'Baseline', 'GBI'])
def model_name(request):
    """Parametrize model names for testing"""
    return request.param

@pytest.fixture
def num_train():
    """Default number of training samples for tests"""
    return 10  # Small number for fast testing

@pytest.fixture
def epochs():
    """Default number of epochs for tests"""
    return 1

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'num_train': 10,
        'epochs': 1,
        'cuda': False,
        'save_checkpoints': False,
        'num_workers': 0  # Use 0 for deterministic behavior in tests (especially on Windows)
    }