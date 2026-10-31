"""Cached collection must preserve causal probabilities and differentiable replay."""

from types import SimpleNamespace

import pytest
import torch

from ..models import JointQwenVLPlanner
from .test_stage1_dfa_loss import joint_fixture
from .test_causal_alignment import ExpandingImageProcessor


class Processor(ExpandingImageProcessor):
    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        return "@" * sum(item["type"] == "image" for item in content) + content[-1]["text"]

    def __call__(self, **kwargs):
        batch = super().__call__(**kwargs)
        if kwargs.get("images"):
            batch["pixel_values"] = torch.zeros(1, 3)
            batch["image_grid_thw"] = torch.ones(1, 3, dtype=torch.long)
        return batch


class CachedBackbone(torch.nn.Module):
    def __init__(self, returns_cache=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(1024, 8)
        self.embedding.weight.data.copy_(torch.arange(8).expand(1024, 8) / 1024)
        self.config = SimpleNamespace(hidden_size=8)
        self.gradient_checkpointing = True
        self.returns_cache = returns_cache
        self.calls = []
        self.rope_deltas = None
        self.cached_length = 0
        self.visual_prefill = False

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        self.calls.append((input_ids.shape[1], use_cache, self.gradient_checkpointing))
        if use_cache:
            assert not torch.is_grad_enabled()
            assert not self.gradient_checkpointing
        if past_key_values is not None:
            assert "pixel_values" not in kwargs
            positions = kwargs["position_ids"]
            delta = -15 if self.visual_prefill else 0
            expected = torch.arange(self.cached_length, self.cached_length + input_ids.shape[1]) + delta
            assert torch.equal(positions.reshape(-1, input_ids.shape[1])[0], expected)
            self.cached_length += input_ids.shape[1]
        elif use_cache:
            self.cached_length = input_ids.shape[1]
            self.visual_prefill = "pixel_values" in kwargs
            if self.visual_prefill:
                self.rope_deltas = torch.tensor([[-15]])
        hidden = self.embedding(input_ids).cumsum(1)
        if past_key_values is not None:
            hidden = hidden + past_key_values
        cache = hidden[:, -1:].detach() if use_cache and self.returns_cache else None
        return SimpleNamespace(hidden_states=(hidden,), past_key_values=cache)


def planner_for(runtime, returns_cache=True):
    planner = JointQwenVLPlanner(CachedBackbone(returns_cache), Processor(),
        eai_vocabulary=runtime.eai_vocabulary, vlabench_vocabulary=runtime.vlabench_vocabulary)
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        for domain in ("eai", "vlabench"):
            head = planner.label_heads[domain]
            head.weight.copy_(torch.randn(head.weight.shape, generator=generator) / 10)
            head.bias.zero_()
            head.bias[planner.vocabulary(domain).eos_label] = -10000
    planner.free_decoding = True
    return planner


@pytest.mark.parametrize("domain", ["eai", "vlabench"])
def test_cached_sampling_matches_dense_and_replay_with_less_prefill_work(joint_fixture, domain):
    _, runtime = joint_fixture
    planner = planner_for(runtime)
    context = {"instruction": "test", "images": (object(),), "entity_table": ("button0", "bowl")}
    with torch.no_grad():
        planner.sampling_use_cache = False
        dense_labels, dense_logprob = planner.sample_labels(domain, context, None, max_steps=4, deterministic=True)
        dense_work = sum(call[0] for call in planner.model.calls)
        planner.model.calls.clear()
        planner.sampling_use_cache = True
        cached_labels, cached_logprob = planner.sample_labels(domain, context, None, max_steps=4, deterministic=True)
    assert cached_labels == dense_labels
    torch.testing.assert_close(cached_logprob, dense_logprob)
    assert sum(call[0] for call in planner.model.calls) < dense_work
    assert planner.model.gradient_checkpointing and planner.model.training
    # Replay still builds a full differentiable trajectory, with checkpointing restored.
    replayed = planner.replay_labels_logprob(domain, context, cached_labels, None, max_steps=4)
    torch.testing.assert_close(replayed, cached_logprob)
    (-replayed).backward()
    assert planner.label_heads[domain].weight.grad is not None
    assert planner.model.calls[-1][1:] == (False, True)


def test_cacheless_backbones_keep_working(joint_fixture):
    _, runtime = joint_fixture
    planner = planner_for(runtime, returns_cache=False)
    with torch.no_grad():
        labels, _ = planner.sample_labels("eai", {"instruction": "test"}, None, max_steps=3, deterministic=True)
    assert len(labels) == 3
    assert all(call[0] > 3 for call in planner.model.calls)
    assert planner.model.gradient_checkpointing


def test_text_sampling_after_images_does_not_reuse_stale_visual_rope_offsets(joint_fixture):
    _, runtime = joint_fixture
    planner = planner_for(runtime)
    with torch.no_grad():
        planner.sample_labels("vlabench", {"instruction": "test", "images": (object(),)}, None, max_steps=3)
        assert planner.model.rope_deltas is not None
        planner.sample_labels("eai", {"instruction": "test"}, None, max_steps=3)


def test_checkpoint_flags_are_restored_if_sampling_forward_raises(joint_fixture, monkeypatch):
    _, runtime = joint_fixture
    planner = planner_for(runtime)

    def fail(**kwargs):
        assert not planner.model.gradient_checkpointing
        raise RuntimeError("forward failed")

    monkeypatch.setattr(planner.model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failed"), torch.no_grad():
        planner.sample_labels("eai", {"instruction": "test"}, None, max_steps=3)
    assert planner.model.gradient_checkpointing and planner.model.training


@pytest.mark.parametrize("replay_available", [False, True])
def test_simulator_collection_disables_gradients_only_when_replay_is_available(replay_available):
    from test_regr.VLABenchAgentInterface.simulation.program import VLABenchHierarchicalReinforcementProgram

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    seen = []

    def sample(**kwargs):
        seen.append(torch.is_grad_enabled())
        return parameter.square()

    program = SimpleNamespace(planner_head=SimpleNamespace(sample_with_logprob=sample))
    result = VLABenchHierarchicalReinforcementProgram._sample_planner_candidate(
        program, replay_available=replay_available)
    assert seen == [not replay_available]
    assert result.requires_grad == (not replay_available)
    assert torch.is_grad_enabled()
