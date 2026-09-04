import pytest
import torch
from model import Net, SumLayer, SumLayerExplicit, build_program
import config

class TestNet:
    def test_net_forward(self):
        """Test Net forward pass"""
        net = Net()
        # Create dummy input: batch_size=1, 2 images, 784 pixels each
        x = torch.randn(1, 2, 784)
        output = net(x)
        
        assert output.shape == (2, 10)  # 2 digits, 10 classes each

    def test_net_output_range(self):
        """Test that Net produces reasonable outputs"""
        net = Net()
        x = torch.randn(1, 2, 784)
        output = net(x)
        
        # Check that outputs are finite
        assert torch.isfinite(output).all()

class TestSumLayer:
    def test_sum_layer_forward(self):
        """Test SumLayer forward pass"""
        sum_layer = SumLayer()
        # Input: 2 digits with 10 classes each
        digits = torch.randn(2, 10)
        output = sum_layer(digits, do_time=False)
        
        assert output.shape == (1, 19)  # 19 possible sums (0-18)

    def test_sum_layer_output_finite(self):
        """Test that SumLayer produces finite outputs"""
        sum_layer = SumLayer()
        digits = torch.randn(2, 10)
        output = sum_layer(digits, do_time=False)
        
        assert torch.isfinite(output).all()

class TestSumLayerExplicit:
    def test_explicit_sum_layer_forward(self):
        """Test SumLayerExplicit forward pass"""
        sum_layer = SumLayerExplicit(device='cpu')
        digits = torch.randn(2, 10)
        output = sum_layer(digits, do_time=False)
        
        assert output.shape == (1, 19)  # 19 possible sums (0-18)

    def test_explicit_sum_layer_probabilities(self):
        """Test that SumLayerExplicit produces valid probability-like outputs"""
        sum_layer = SumLayerExplicit(device='cpu')
        digits = torch.randn(2, 10)
        output = sum_layer(digits, do_time=False)
        
        # Should be non-negative (after softmax application)
        assert (output >= 0).all()

class TestBuildProgram:
    @pytest.mark.parametrize("sum_setting", [None, 'explicit', 'baseline'])
    def test_build_program_structure(self, sum_setting):
        """Test that build_program returns correct structure"""
        graph, image, image_pair, image_batch = build_program(
            sum_setting=sum_setting, 
            digit_labels=False, 
            device='cpu', 
            use_fixedL=True, 
            test=False
        )
        
        assert graph is not None
        assert image is not None
        assert image_pair is not None
        assert image_batch is not None

    def test_build_program_with_digit_labels(self):
        """Test build_program with digit labels enabled"""
        graph, image, image_pair, image_batch = build_program(
            sum_setting=None, 
            digit_labels=True, 
            device='cpu', 
            use_fixedL=True, 
            test=False
        )
        
        assert graph is not None