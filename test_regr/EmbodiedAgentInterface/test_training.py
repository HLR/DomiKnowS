import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from main import build_program, labels_through_first_eos
from dataset import dummy_dataset
from modules import EOSMaskedCrossEntropyLoss
from rl_sequence_program import AutoregressiveSequenceReinforcementProgram


def test_supervised_loss_ignores_eos_padding():
    logits = torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[1, 0, 0, 0]])
    loss = EOSMaskedCrossEntropyLoss(eos_label=0)(logits, labels)
    expected = F.cross_entropy(logits[:, :2, :].reshape(-1, 3), labels[:, :2].reshape(-1))
    assert torch.allclose(loss, expected)

    loss.backward()
    assert logits.grad[:, :2, :].abs().sum() > 0
    assert logits.grad[:, 2:, :].abs().sum() == 0


def test_effective_metric_keeps_one_eos_only():
    assert labels_through_first_eos([4, 7, 0, 0, 0], eos_label=0) == [4, 7, 0]
    assert labels_through_first_eos([4, 7], eos_label=0) == [4, 7]


class _PrefixRecordingHead(torch.nn.Module):
    supports_batched_prefixes = True

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.calls = []

    def sequence_logits(self, _text, prefixes):
        self.calls.append(prefixes.detach().clone())
        batch, length = prefixes.shape
        logits = self.bias.reshape(1, 1, 3).expand(batch, length, 3).clone()
        if length == 1:
            logits[:, -1, 2] = logits[:, -1, 2] - 100.0
        else:
            logits[:, -1, 2] = logits[:, -1, 2] + 100.0
        return logits


def test_rl_rollout_conditions_on_its_sampled_prefix():
    torch.manual_seed(7)
    head = _PrefixRecordingHead()
    program = SimpleNamespace(
        autoregressive_head=head,
        eos_label=2,
        max_steps=3,
        num_samples=8,
    )
    trajectories, logprob = (
        AutoregressiveSequenceReinforcementProgram._sample_trajectories(program, "task")
    )

    assert len(head.calls) == 2
    sampled_first_tokens = [trajectory[0] for trajectory in trajectories]
    assert head.calls[1][:, 1].tolist() == sampled_first_tokens
    assert all(trajectory[-1] == 2 for trajectory in trajectories)
    assert logprob.shape == torch.Size([8])
    assert logprob.requires_grad


def test_rl_program_is_constructed_with_shared_supervised_head():
    examples = dummy_dataset(device="cpu", max_steps=8)
    vocabulary = examples[0]["generation_vocab"]
    solver, solver_bundle = build_program(
        max_steps=8,
        vocab=vocabulary,
        object_tokens=examples[0]["object_tokens"],
        action_tokens=("close",),
        program_type="solver",
        world_constraint_builders=(),
    )
    rl, rl_bundle = build_program(
        max_steps=8,
        vocab=vocabulary,
        object_tokens=examples[0]["object_tokens"],
        action_tokens=("close",),
        program_type="reinforcement",
        rl_num_samples=1,
        shared_autoregressive_head=solver.autoregressive_head,
    )
    assert rl.autoregressive_head is solver.autoregressive_head
    assert rl.model is not solver.model
    assert rl_bundle is not solver_bundle
    assert rl.graph.name == "eai_generation_graph"
    assert rl.graph is not rl_bundle.world.graph
    assert rl_bundle.world.graph is not solver_bundle.world.graph
    assert not solver_bundle.world.has_constraints
    assert rl_bundle.world.has_constraints


def run_tests():
    tests = [
        test_supervised_loss_ignores_eos_padding,
        test_effective_metric_keeps_one_eos_only,
        test_rl_rollout_conditions_on_its_sampled_prefix,
        test_rl_program_is_constructed_with_shared_supervised_head,
    ]
    for test in tests:
        test()
    print(f"TRAINING_REGRESSION_DONE {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    run_tests()
