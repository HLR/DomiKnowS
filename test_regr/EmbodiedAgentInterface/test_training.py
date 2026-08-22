import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from main import (
    build_program,
    build_stage2_program,
    labels_through_first_eos,
    load_eai_checkpoint,
    save_eai_checkpoint,
    stage1_allows_rl,
)
from dataset import dummy_dataset
from modules import EOSMaskedCrossEntropyLoss, PretrainedLabelAdapter
from domiknows.generation import (
    TokenVocabulary,
    after_token_allowed_map_dfa,
    first_token_allowed_dfa,
    product_dfa,
)
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
        self.grad_enabled = []

    def sequence_logits(self, _text, prefixes):
        self.calls.append(prefixes.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
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

    assert len(head.calls) == 3
    sampled_first_tokens = [trajectory[0] for trajectory in trajectories]
    assert head.calls[1][:, 1].tolist() == sampled_first_tokens
    assert head.calls[2][:, 0].tolist() == [2] * 8
    assert head.calls[2][:, 1].tolist() == sampled_first_tokens
    assert head.grad_enabled == [False, False, True]
    assert all(trajectory[-1] == 2 for trajectory in trajectories)
    assert logprob.shape == torch.Size([8])
    assert logprob.requires_grad


def test_rl_supervised_anchor_is_differentiable():
    head = _PrefixRecordingHead()
    program = SimpleNamespace(autoregressive_head=head, eos_label=2)
    loss = AutoregressiveSequenceReinforcementProgram.supervised_anchor_loss(
        program,
        {
            "text": "synthetic",
            "target_action_labels": torch.tensor([0, 1, 2, 2]),
        },
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert head.bias.grad is not None


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
    assert rl_bundle.reward_mode == "dense"
    assert rl.supervised_weight == 0.1


def test_two_stage_handoff_reuses_exact_graph_bundle_head_and_dfa():
    examples = dummy_dataset(device="cpu", max_steps=8)
    solver, bundle = build_program(
        max_steps=8,
        vocab=examples[0]["generation_vocab"],
        object_tokens=examples[0]["object_tokens"],
        action_tokens=("close",),
        action_sequence_tokens=examples[0]["action_tokens"],
        program_type="solver",
        world_constraint_builders=(),
    )
    args = SimpleNamespace(
        max_steps=8,
        rl_supervised_weight=0.1,
        rl_num_samples=1,
        rl_estimator="reinforce",
        generation_constraints="always",
        rl_reward_mode="dense",
        rl_constraint_weight=0.25,
        rl_constraint_aggregate="mean",
    )
    rl, rl_bundle = build_stage2_program(args, solver, bundle, examples, "cpu")
    assert rl.graph is solver.graph
    assert rl_bundle is bundle
    assert rl.autoregressive_head is solver.autoregressive_head
    assert rl.policy_dfa is bundle.policy_dfa


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": {" alpha": [1], " beta": [2, 3]}.get(text, [1])}


class _FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_embeddings = torch.nn.Embedding.from_pretrained(
            torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]
            )
        )

    def get_output_embeddings(self):
        return self.output_embeddings


def test_pretrained_label_adapter_uses_native_label_vectors_and_trains_residual():
    vocabulary = TokenVocabulary(["<eos>", "alpha", "beta"], eos_token="<eos>")
    adapter = PretrainedLabelAdapter(
        _FakeCausalLM(), _FakeTokenizer(), vocabulary, eos_label=0, hidden_size=2, rank=1
    )
    expected = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 2.0]])
    assert torch.allclose(adapter.base_label_vectors, expected)
    assert torch.count_nonzero(adapter.residual_up.weight) == 0

    hidden = torch.tensor([[1.0, 2.0]], requires_grad=True)
    adapter(hidden).sum().backward()
    assert adapter.bias.grad is not None
    assert adapter.log_temperature.grad is not None
    assert adapter.residual_up.weight.grad is not None


class _UniformPolicyHead(torch.nn.Module):
    supports_batched_prefixes = True

    def __init__(self, label_count):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(label_count))

    def sequence_logits(self, _text, prefixes):
        return self.bias.reshape(1, 1, -1).expand(
            prefixes.shape[0], prefixes.shape[1], -1
        )


def test_rl_rollout_and_rescore_share_the_graph_compiled_policy():
    vocabulary = TokenVocabulary(["a", "b", "<eos>"], eos_token="<eos>")
    policy = product_dfa(
        (
            first_token_allowed_dfa(vocabulary, ["a"]),
            after_token_allowed_map_dfa(
                vocabulary, {"a": ["b"], "b": ["<eos>"]}
            ),
        )
    )
    head = _UniformPolicyHead(vocabulary.label_count)
    program = SimpleNamespace(
        autoregressive_head=head,
        eos_label=vocabulary.eos_label,
        max_steps=3,
        num_samples=4,
        policy_dfa=policy,
    )
    trajectories, logprob = (
        AutoregressiveSequenceReinforcementProgram._sample_trajectories(program, "task")
    )
    expected = [
        vocabulary.label_for_token("a"),
        vocabulary.label_for_token("b"),
        vocabulary.eos_label,
    ]
    assert trajectories == [expected] * 4
    assert all(policy.accepts(trajectory) for trajectory in trajectories)
    assert torch.isfinite(logprob).all()
    assert logprob.requires_grad


def test_stage1_gate_requires_positive_task_reward():
    assert not stage1_allows_rl({"positive_reward_rate": 0.0})
    assert stage1_allows_rl({"positive_reward_rate": 1.0 / 88.0})


def test_eai_checkpoint_metadata_roundtrip_and_compatibility_rejection():
    vocabulary = TokenVocabulary(["<eos>", "a"], eos_token="<eos>")
    args = SimpleNamespace(
        baseline_model="causal-lm",
        llm_backbone_path="fake/qwen",
        causal_label_head="pretrained-adapter",
        label_adapter_rank=64,
    )
    bundle = SimpleNamespace(vocabulary=vocabulary)
    source = SimpleNamespace(model=torch.nn.Linear(2, 2))
    target = SimpleNamespace(model=torch.nn.Linear(2, 2))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        save_eai_checkpoint(source, bundle, args, path, stage="stage1", epoch=5)
        metadata = load_eai_checkpoint(target, bundle, args, path, map_location="cpu")
        assert metadata["stage"] == "stage1"
        assert metadata["epoch"] == 5
        incompatible = SimpleNamespace(**vars(args))
        incompatible.label_adapter_rank = 32
        try:
            load_eai_checkpoint(target, bundle, incompatible, path, map_location="cpu")
        except ValueError as exc:
            assert "label_adapter_rank" in str(exc)
        else:
            raise AssertionError("incompatible checkpoint was accepted")


def run_tests():
    tests = [
        test_supervised_loss_ignores_eos_padding,
        test_effective_metric_keeps_one_eos_only,
        test_rl_rollout_conditions_on_its_sampled_prefix,
        test_rl_supervised_anchor_is_differentiable,
        test_rl_program_is_constructed_with_shared_supervised_head,
        test_two_stage_handoff_reuses_exact_graph_bundle_head_and_dfa,
        test_pretrained_label_adapter_uses_native_label_vectors_and_trains_residual,
        test_rl_rollout_and_rescore_share_the_graph_compiled_policy,
        test_stage1_gate_requires_positive_task_reward,
        test_eai_checkpoint_metadata_roundtrip_and_compatibility_rejection,
    ]
    for test in tests:
        test()
    print(f"TRAINING_REGRESSION_DONE {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    run_tests()
