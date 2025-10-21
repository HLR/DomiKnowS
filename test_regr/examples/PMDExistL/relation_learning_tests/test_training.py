import pytest
import torch
import subprocess
import sys


class TestTraining:
    """Test suite for relation learning training pipeline"""

    def test_minimal_training_run(self):
        """Test that training completes successfully with minimal parameters"""
        result = subprocess.run(
            [
                sys.executable, "main_rel.py",
                "--N", "100",
                "--lr", "1e-4",
                "--epoch", "2",
                "--max_relation", "1",
                "--save_file", "test_model.pth"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Training failed with error: {result.stderr}"
        assert "Acc on training set after training:" in result.stdout

    def test_cuda_availability(self):
        """Verify CUDA is available on GPU runner"""
        assert torch.cuda.is_available(), "CUDA not available on GPU runner"
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_training_with_constraint(self):
        """Test training with logical constraints"""
        result = subprocess.run(
            [
                sys.executable, "main_rel.py",
                "--N", "50",
                "--lr", "1e-4",
                "--epoch", "1",
                "--max_relation", "2",
                "--constraint_2_existL"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0, f"Training with constraint failed: {result.stderr}"

    def test_model_save_and_load(self):
        """Test model persistence"""
        # Train and save
        result_train = subprocess.run(
            [
                sys.executable, "main_rel.py",
                "--N", "50",
                "--lr", "1e-4",
                "--epoch", "1",
                "--max_relation", "1",
                "--save_file", "test_save_model.pth"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result_train.returncode == 0
        
        # Load and evaluate
        result_eval = subprocess.run(
            [
                sys.executable, "main_rel.py",
                "--N", "50",
                "--evaluate",
                "--load_save", "test_save_model.pth",
                "--max_relation", "1"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result_eval.returncode == 0
        assert "Acc on training set after training:" in result_eval.stdout