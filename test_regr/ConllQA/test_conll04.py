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
        
        # Create environment that preserves current Python path
        self.env = os.environ.copy()
        
        # Add project root to PYTHONPATH if not already there
        project_root = Path(__file__).parent.parent.parent.parent
        current_pythonpath = self.env.get('PYTHONPATH', '')
        if current_pythonpath:
            self.env['PYTHONPATH'] = f"{project_root}{os.pathsep}{current_pythonpath}"
        else:
            self.env['PYTHONPATH'] = str(project_root)
    
    def _run_main(self, args, timeout=300):
        """Helper to run main_rel.py with proper environment"""
        cmd = [sys.executable, "-u", "main_rel.py"] + args
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.test_dir,
            env=self.env
        )
        return result
        
    def test_minimal_training_run(self):
        """Test that training completes successfully with minimal parameters"""
        result = self._run_main([
            "--N", "100",
            "--lr", "1e-4",
            "--epoch", "2",
            "--max_relation", "1",
            "--save_file", "test_model.pth"
        ])
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        assert result.returncode == 0, f"Training failed with error: {result.stderr}"
        assert "Acc on training set after training:" in result.stdout

    def test_cuda_availability(self):
        """Verify CUDA is available (skip if not available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available in this environment")
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_training_with_constraint(self):
        """Test training with logical constraints"""
        result = self._run_main([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "1",
            "--max_relation", "2",
            "--constraint_2_existL"
        ])
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        assert result.returncode == 0, f"Training with constraint failed: {result.stderr}"

    def test_model_save_and_load(self):
        """Test model persistence"""
        # Train and save
        result_train = self._run_main([
            "--N", "50",
            "--lr", "1e-4",
            "--epoch", "1",
            "--max_relation", "1",
            "--save_file", "test_save_model.pth"
        ])
        
        if result_train.returncode != 0:
            print("Training STDOUT:", result_train.stdout)
            print("Training STDERR:", result_train.stderr)
        
        assert result_train.returncode == 0, f"Training failed: {result_train.stderr}"
        
        # Load and evaluate
        result_eval = self._run_main([
            "--N", "50",
            "--evaluate",
            "--load_save", "test_save_model.pth",
            "--max_relation", "1"
        ])
        
        if result_eval.returncode != 0:
            print("Evaluation STDOUT:", result_eval.stdout)
            print("Evaluation STDERR:", result_eval.stderr)
        
        assert result_eval.returncode == 0, f"Evaluation failed: {result_eval.stderr}"
        assert "Acc on training set after training:" in result_eval.stdout

    def test_import_modules(self):
        """Test that required modules can be imported"""
        try:
            import domiknows
            from domiknows.program.model.base import Mode
            from domiknows.graph import Graph, Concept, Relation
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")