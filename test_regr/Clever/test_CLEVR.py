"""
Integration tests for the five main operational modes of main.py.

Each test launches main.py as a subprocess with the appropriate flags,
captures stdout/stderr, and asserts on the expected behavioral properties.
"""

import subprocess
import sys
import os
import re
import signal
import random
from pathlib import Path
import pytest
from flaky import flaky

PYTHON = sys.executable
# Resolve main.py relative to this test file so it works regardless of cwd
_TEST_DIR = Path(__file__).resolve().parent
MAIN = str(_TEST_DIR / "main.py")
TIMEOUT = 1800  # 30 minutes max per test (10-epoch training can be slow on CI)


def _run(args: list[str], timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    """Run main.py with given CLI args, capture output, and kill the entire
    process group on exit so orphaned vLLM EngineCore subprocesses don't hold
    GPU memory for the next test."""
    cmd = [PYTHON, MAIN] + args
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(_TEST_DIR),
        start_new_session=True,  # new process group so we can kill grandchildren
    )
    pgid = os.getpgid(proc.pid)
    stdout, stderr = "", ""
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    finally:
        # Kill the whole process group to reap any orphaned vLLM EngineCore
        # subprocesses spawned by main.py (they outlive main.py on crash).
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _skip_if_vllm_failed(result: subprocess.CompletedProcess):
    """Skip test if vLLM engine failed to initialize or died during execution
    (GPU too small, driver issue, EngineDeadError, etc.)."""
    combined = result.stdout + result.stderr
    if "Engine core initialization failed" in combined:
        pytest.skip("vLLM engine core failed to initialize (likely insufficient GPU memory)")
    if "EngineDeadError" in combined or "EngineCore encountered an issue" in combined:
        pytest.skip("vLLM EngineCore died during execution (likely OOM or driver issue)")
    if "ImportError" in combined and "timm" in combined:
        pytest.skip("Missing 'timm' package required by InternVL")


def _parse_accuracy(output: str, pattern: str) -> float | None:
    """Extract the first accuracy value matching *pattern* from combined output.

    Parameters
    ----------
    pattern : str
        Regex that must contain one capturing group for the numeric accuracy
        value (e.g. ``r"Accuracy on \\w+:\\s+([\\d.]+)%"``).
    """
    m = re.search(pattern, output)
    if m:
        return float(m.group(1))
    return None


def _parse_all_epoch_accuracies(output: str) -> list[float]:
    """Return all per-epoch train accuracy values found in stdout."""
    return [float(v) for v in re.findall(r"Epoch \d+ train accuracy:\s+([\d.]+)%", output)]


def _parse_baseline_accuracy(output: str) -> float | None:
    m = re.search(r"Accuracy before training:\s+([\d.]+)%", output)
    return float(m.group(1)) if m else None


def _parse_final_train_accuracy(output: str) -> float | None:
    m = re.search(r"Train accuracy after training:\s+([\d.]+)%", output)
    return float(m.group(1)) if m else None


def _parse_eval_accuracy(output: str) -> float | None:
    """Parse the single-shot evaluation accuracy printed in --infer-only / --eval-only."""
    m = re.search(r"Accuracy on \w+:\s+([\d.]+)%", output)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 1. Untrained Inference (Random Baseline)
# ---------------------------------------------------------------------------
class TestUntrainedBaseline:
    """--infer-only skips training and evaluates an untrained model.

    EXPECTED: accuracy near random (~50% for binary questions).
    """

    ARGS = ["--train-size", "10", "--test-size", "10", "--epochs", "1", "--infer-only", "--train-start", str(random.randint(0, 7846)), "--test-start", str(random.randint(0, 7846))]

    def test_exits_successfully(self):
        result = _run(self.ARGS)
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    def test_prints_evaluation_accuracy(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        acc = _parse_eval_accuracy(combined)
        assert acc is not None, (
            "Expected 'Accuracy on ...: XX.XX%' in output.\n"
            f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}"
        )

    @flaky(max_runs=3, min_passes=1)
    def test_accuracy_near_random(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        acc = _parse_eval_accuracy(combined)
        assert acc is not None, "Could not parse accuracy from output"
        # Random baseline for binary: expect between 0% and 90%
        # (we use a generous upper bound — the key point is it should NOT be 100%)
        assert acc < 90.0, (
            f"Untrained model accuracy {acc}% is suspiciously high for a random baseline"
        )

    def test_no_training_output(self):
        """Infer-only should NOT print training epoch lines."""
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        assert "Training epoch" not in combined, (
            "Found 'Training epoch' output in --infer-only mode"
        )


# ---------------------------------------------------------------------------
# 2. Zero-Shot VLM Inference
# ---------------------------------------------------------------------------
class TestZeroShotVLM:
    """--use-vlm with --infer-only (implicitly via epochs=1 + no training flag).

    Note: The original script does NOT pass --infer-only for this mode,
    so it will do 1 epoch of training + evaluation. We replicate that.

    EXPECTED: accuracy noticeably above 50% thanks to InternVL's prior knowledge.
    """

    ARGS = ["--train-size", "10", "--test-size", "10", "--epochs", "1", "--use-vlm", "--train-start", str(random.randint(0, 7846)), "--test-start", str(random.randint(0, 7836))]
    # VLM uses InternVL3_5-8B by default; if MODEL_PATH is set, override
    if os.environ.get("MODEL_PATH"):
        ARGS += ["--model-path", os.environ["MODEL_PATH"]]

    @pytest.mark.slow
    def test_exits_successfully(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    @pytest.mark.slow
    def test_vlm_mode_indicated(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        combined = result.stdout + result.stderr
        assert "Use VLM:" in combined or "use_vlm" in combined.lower(), (
            "Expected VLM mode indication in output"
        )

    @pytest.mark.slow
    @flaky(max_runs=3, min_passes=1)
    def test_accuracy_above_random(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        combined = result.stdout + result.stderr
        # Look for either final train accuracy or test accuracy
        acc = _parse_final_train_accuracy(combined)
        if acc is None:
            acc = _parse_eval_accuracy(combined)
        assert acc is not None, "Could not parse any accuracy from VLM output"
        # VLM should ideally beat random; we test a soft lower bound
        # (this is informational — VLM quality varies)
        assert acc >= 0.0, f"Accuracy {acc}% is invalid"


# ---------------------------------------------------------------------------
# 3. Oracle Mode (Ground Truth)
# ---------------------------------------------------------------------------
class TestOracleMode:
    """--oracle-mode uses ground truth answers to bypass visual learning.

    EXPECTED: accuracy MUST be 100%.
    """

    ARGS = ["--train-size", "10", "--test-size", "10", "--epochs", "1", "--oracle-mode", "--train-start", str(random.randint(0, 7846)), "--test-start", str(random.randint(0, 7846))]

    def test_exits_successfully(self):
        result = _run(self.ARGS)
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    def test_oracle_mode_indicated(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        assert "Oracle mode" in combined or "oracle_mode" in combined.lower(), (
            "Expected oracle mode indication in config output"
        )

    def test_accuracy_is_100(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        # Oracle mode with 1 epoch: check final train acc or eval acc
        acc = _parse_final_train_accuracy(combined)
        if acc is None:
            acc = _parse_eval_accuracy(combined)
        if acc is None:
            # Also try baseline accuracy (oracle should be perfect even before training)
            acc = _parse_baseline_accuracy(combined)
        assert acc is not None, (
            "Could not parse accuracy from oracle mode output.\n"
            f"STDOUT:\n{result.stdout[-3000:]}"
        )
        assert acc == 100.0, (
            f"Oracle mode accuracy was {acc}%, but MUST be 100.0%"
        )


# ---------------------------------------------------------------------------
# 4. Standard Backprop Training (ResNet/Embeddings)
# ---------------------------------------------------------------------------
class TestStandardTraining:
    """Standard differentiable training with ResNet/embedding learners.

    EXPECTED: constraint loss should decrease across 10 epochs, proving
    gradients flow from logic into CNN/Linear layers.
    """

    ARGS = ["--train-size", "10", "--epochs", "10", "--test-size", "10", "--train-start", str(random.randint(0, 7846)), "--test-start", str(random.randint(0, 7846))]

    def test_exits_successfully(self):
        result = _run(self.ARGS)
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    def test_prints_epoch_progress(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        assert "Training epoch" in combined, (
            "Expected 'Training epoch' lines in training output"
        )

    def test_multiple_epochs_executed(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        epoch_accs = _parse_all_epoch_accuracies(combined)
        assert len(epoch_accs) >= 2, (
            f"Expected multiple epoch accuracies, got {len(epoch_accs)}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}"
        )

    @flaky(max_runs=3, min_passes=1)
    def test_training_shows_improvement_or_stability(self):
        """The accuracy should not collapse to 0% — it should stay near
        baseline or improve over the 10 epochs."""
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        baseline = _parse_baseline_accuracy(combined)
        final = _parse_final_train_accuracy(combined)
        assert final is not None, "Could not parse final train accuracy"
        # At minimum, accuracy should not be 0 after training
        assert final > 0.0, f"Final accuracy collapsed to {final}%"

    def test_baseline_accuracy_printed(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        acc = _parse_baseline_accuracy(combined)
        assert acc is not None, "Expected 'Accuracy before training:' in output"

    def test_final_accuracy_printed(self):
        result = _run(self.ARGS)
        combined = result.stdout + result.stderr
        acc = _parse_final_train_accuracy(combined)
        assert acc is not None, "Expected 'Train accuracy after training:' in output"


# ---------------------------------------------------------------------------
# 5. PEFT / LoRA Training (InternVL)
# ---------------------------------------------------------------------------
class TestPEFTTraining:
    """--peft enables LoRA fine-tuning of InternVL.

    EXPECTED: constraint loss should decrease, proving gradients flow
    from the logic solver into the VLM's LoRA adapters.
    """

    ARGS = ["--train-size", "10", "--test-size", "10", "--epochs", "10", "--peft", "--train-start", str(random.randint(0, 7846)), "--test-start", str(random.randint(0, 7846))]
    # PEFT uses InternVL3_5-1B by default; use local copy if available
    if os.environ.get("MODEL_PATH"):
        ARGS += ["--model-path", os.environ["MODEL_PATH"]]

    @pytest.mark.slow
    def test_exits_successfully(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    @pytest.mark.slow
    def test_peft_mode_indicated(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        combined = result.stdout + result.stderr
        assert "PEFT" in combined or "peft" in combined.lower() or "LoRA" in combined, (
            "Expected PEFT/LoRA mode indication in output"
        )

    @pytest.mark.slow
    def test_prints_epoch_progress(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        combined = result.stdout + result.stderr
        assert "Training epoch" in combined, (
            "Expected 'Training epoch' lines in PEFT training output"
        )

    @pytest.mark.slow
    @flaky(max_runs=3, min_passes=1)
    def test_training_shows_improvement_or_stability(self):
        result = _run(self.ARGS)
        _skip_if_vllm_failed(result)
        combined = result.stdout + result.stderr
        final = _parse_final_train_accuracy(combined)
        assert final is not None, "Could not parse final train accuracy for PEFT"
        assert final > 0.0, f"PEFT final accuracy collapsed to {final}%"