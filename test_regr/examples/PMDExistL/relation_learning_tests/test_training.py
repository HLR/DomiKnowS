import pytest
import torch
import subprocess
import sys
import os
from pathlib import Path


class TestTraining:
    """Test suite for relation learning training pipeline"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test directory and environment"""
        self.test_dir = Path(__file__).parent
        self.env = os.environ.copy()
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["CUDA_LAUNCH_BLOCKING"] = "1"
        self.env["PYTHONPATH"] = str(self.test_dir) + os.pathsep + self.env.get("PYTHONPATH", "")
        
    def run_training(self, args, timeout=300):
        """Helper to run training with proper environment"""
        cmd = [sys.executable, str(self.test_dir / "main_rel.py")] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(self.test_dir),
            env=self.env
        )
        return result

    def test_cuda_availability(self):
        """Verify CUDA is available on GPU runner"""
        assert torch.cuda.is_available(), "CUDA not available on GPU runner"
        assert torch.cuda.device_count() > 0, "No CUDA devices found"
        print(f"GPU count: {torch.cuda.device_count()}")

    def test_minimal_training_run(self):
        """Test that training completes successfully with minimal parameters"""
        result = self.run_training([
            "--N", "100",
            "--lr", "1e-4",
            "--epoch", "2",
            "--max_relation", "1",
            "--save_file", "test_model.pth"
        ])
        
        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Acc on training set after training:" in result.stdout

    def test_training_with_existL_constraint(self):
        """Test training with existL logical constraint"""
        result = self.run_training([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "1",
            "--max_relation", "2",
            "--constraint_2_existL"
        ])
        
        assert result.returncode == 0, (
            f"Training with constraint failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_training_with_andL_constraint(self):
        """Test training with andL logical constraint"""
        result = self.run_training([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "1",
            "--max_relation", "2",
            "--use_andL"
        ])
        
        assert result.returncode == 0, (
            f"Training with andL constraint failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_property_only_training(self):
        """Test training with properties only (no relations)"""
        result = self.run_training([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "2",
            "--max_relation", "0"
        ])
        
        assert result.returncode == 0, (
            f"Property-only training failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "No relation" in result.stdout or "Acc on training set" in result.stdout

    def test_model_save_and_load(self):
        """Test model persistence"""
        save_file = "test_save_model.pth"
        
        # Train and save
        result_train = self.run_training([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "1",
            "--max_relation", "1",
            "--save_file", save_file
        ])
        
        assert result_train.returncode == 0, (
            f"Training failed\n"
            f"STDOUT:\n{result_train.stdout}\n"
            f"STDERR:\n{result_train.stderr}"
        )
        
        # Verify model file exists
        model_path = self.test_dir / save_file
        assert model_path.exists(), f"Model file {save_file} not created"
        
        # Load and evaluate
        result_eval = self.run_training([
            "--N", "50",
            "--evaluate",
            "--load_save", save_file,
            "--max_relation", "1"
        ])
        
        assert result_eval.returncode == 0, (
            f"Evaluation failed\n"
            f"STDOUT:\n{result_eval.stdout}\n"
            f"STDERR:\n{result_eval.stderr}"
        )
        assert "Acc on training set after training:" in result_eval.stdout
        
        # Cleanup
        if model_path.exists():
            model_path.unlink()

    def test_varying_relation_complexity(self):
        """Test training with different relation complexity levels"""
        for max_rel in [0, 1, 2, 3]:
            result = self.run_training([
                "--N", "50",
                "--lr", "1e-4",
                "--epoch", "1",
                "--max_relation", str(max_rel)
            ])
            
            assert result.returncode == 0, (
                f"Training with max_relation={max_rel} failed\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    @pytest.mark.slow
    def test_curriculum_learning_sequence(self):
        """Test curriculum learning from simple to complex"""
        for relation in range(4):
            result = self.run_training([
                "--N", "100",
                "--lr", "1e-4",
                "--epoch", str(2 * (relation + 1)),
                "--max_relation", str(relation),
                "--save_file", f"curriculum_model_{relation}.pth"
            ], timeout=600)
            
            assert result.returncode == 0, (
                f"Curriculum stage {relation} failed\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
            
            # Cleanup
            model_path = self.test_dir / f"curriculum_model_{relation}.pth"
            if model_path.exists():
                model_path.unlink()