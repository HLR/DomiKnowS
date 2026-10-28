"""Causal positions must refer to text after expanded image patches."""

from types import SimpleNamespace

import pytest
import torch

from ..models import JointQwenVLPlanner
from ..program import DomainUpdate, JointSolverPOIProgram


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class ExpandingImageProcessor:
    tokenizer = CharacterTokenizer()

    def __call__(self, *, text, **kwargs):
        ids = self.tokenizer.encode(text[0])
        expanded = []
        for token in ids:
            expanded.extend([999] * 16 if token == ord("@") else [token])
        return {"input_ids": torch.tensor([expanded])}


def alignment_probe(image_count):
    return SimpleNamespace(
        processor=ExpandingImageProcessor(),
        _prompt_base=lambda *args: "instruction\nPlan:\n",
        _chat=lambda prompt, count: "@" * count + prompt,
    )


@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_teacher_forcing_and_generation_use_the_same_text_positions(image_count):
    probe = alignment_probe(image_count)
    tokens = ("skill:press", "arg:target_entity_name", "obj:0")
    context = {"images": [object()] * image_count}
    prepare = JointQwenVLPlanner._prepare_inputs_and_boundaries
    batch, boundaries = prepare(probe, "vlabench", context, tokens)
    ids = batch["input_ids"][0]

    assert len(boundaries) == len(tokens) + 1
    assert boundaries[-1] == len(ids) - 1
    assert ids[boundaries].tolist() == [ord("\n"), ord("s"), ord("e"), ord("0")]
    assert all(int(ids[index]) != 999 for index in boundaries)
    for count in range(len(tokens) + 1):
        prefix_batch, prefix_positions = prepare(probe, "vlabench", context, tokens[:count])
        assert prefix_positions == boundaries[:count + 1]
        assert prefix_positions[-1] == prefix_batch["input_ids"].shape[1] - 1


def test_alignment_rejects_a_processor_that_changes_the_label_suffix():
    probe = alignment_probe(1)
    processor = probe.processor

    def corrupt(**kwargs):
        batch = processor(**kwargs)
        batch["input_ids"][0, -1] = -1
        return batch

    corrupt.tokenizer = processor.tokenizer
    probe.processor = corrupt
    with pytest.raises(ValueError, match="preserve the causal label suffix"):
        JointQwenVLPlanner._prepare_inputs_and_boundaries(
            probe, "vlabench", {"images": [object()]}, ("skill:press",))


def test_alignment_rejects_tokenization_that_merges_across_a_label_boundary():
    probe = alignment_probe(0)
    probe._prompt_base = lambda *args: "x"
    probe.processor.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [1] if text == "x y" else [ord(char) for char in text])
    with pytest.raises(ValueError, match="crosses a causal label boundary"):
        JointQwenVLPlanner._prepare_inputs_and_boundaries(probe, "eai", {}, ("y",))


@pytest.mark.parametrize("eai_count,vlabench_count", [(2, 5), (5, 2)])
@pytest.mark.parametrize("rounds", [None, 2])
def test_stage1_epoch_covers_both_domains_unless_explicitly_capped(eai_count, vlabench_count, rounds):
    seen = {"eai": [], "vlabench": []}

    def train_round(eai, vlabench, **kwargs):
        seen["eai"].append(eai)
        seen["vlabench"].append(vlabench)
        return DomainUpdate("eai", 1.0), DomainUpdate("vlabench", 2.0)

    program = SimpleNamespace(dfa_loss=None, train_round=train_round)
    metrics = JointSolverPOIProgram.train_alternating_epoch(
        program, list(range(eai_count)), list(range(vlabench_count)), rounds=rounds)
    expected_rounds = max(eai_count, vlabench_count) if rounds is None else rounds
    assert metrics["rounds"] == expected_rounds
    assert all(len(values) == expected_rounds for values in seen.values())
    if rounds is None:
        assert set(seen["eai"]) == set(range(eai_count))
        assert set(seen["vlabench"]) == set(range(vlabench_count))
