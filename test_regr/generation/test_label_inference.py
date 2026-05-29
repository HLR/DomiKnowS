import pytest
import torch

from domiknows.generation import (
    LabelInferenceResult,
    TokenVocabulary,
    beam_label_inference,
    greedy_label_inference,
    sample_label_inference,
)
from domiknows.generation.learners import CompactLabelGenerationHead


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.map = {"<eos>": 0, "A": 1, "B": 2}

    def encode(self, token):
        return [self.map[token]]


class BranchingCompactModel:
    label_to_token_id = (0, 1, 2, None)

    def next_label_logits(self, input_ids):
        generated = [int(token_id) for token_id in input_ids[0].tolist()[1:]]
        logits = torch.full((4,), -10.0)
        logits[3] = 50.0
        if not generated:
            logits[1] = 8.0
            logits[2] = 7.0
        elif generated[-1] == 1:
            logits[0] = 6.0
            logits[2] = 1.0
        else:
            logits[0] = 3.0
            logits[1] = 5.0
        return logits

    def token_id_for_label(self, label):
        token_id = self.label_to_token_id[int(label)]
        if token_id is None:
            raise ValueError("label does not map to one token")
        return token_id


class EmptyPrefixCompactModel(BranchingCompactModel):
    def next_label_logits(self, input_ids):
        generated = [int(token_id) for token_id in input_ids[0].tolist()]
        logits = torch.full((4,), -10.0)
        logits[3] = 50.0
        if not generated:
            logits[1] = 8.0
            logits[2] = 7.0
        elif generated[-1] == 1:
            logits[0] = 6.0
            logits[2] = 1.0
        else:
            logits[0] = 3.0
            logits[1] = 5.0
        return logits


class KwargCompactModel(BranchingCompactModel):
    def __init__(self):
        self.seen_biases = []

    def next_label_logits(self, input_ids, bias_label=None):
        self.seen_biases.append(bias_label)
        logits = super().next_label_logits(input_ids)
        if bias_label is not None:
            logits[int(bias_label)] += 20.0
        return logits


class MethodCompactModel(CompactLabelGenerationHead):
    def __init__(self):
        super().__init__(label_count=4, pad_size=3, label_to_token_id=[0, 1, 2, None])

    def next_label_logits(self, input_ids, **_kwargs):
        return BranchingCompactModel().next_label_logits(input_ids)

    def sequence_log_probs(self, target_labels, *, lengths=None, **kwargs):
        raise NotImplementedError


def _vocabulary():
    return TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=FakeTokenizer())


def test_greedy_label_inference_stops_at_eos_and_skips_non_emittable_labels():
    result = greedy_label_inference(
        BranchingCompactModel(),
        _vocabulary(),
        [9],
        max_new_tokens=3,
    )

    assert isinstance(result, LabelInferenceResult)
    assert result.labels == (1, 0)
    assert result.symbols == ("A", "<eos>")
    assert result.token_ids == (9, 1, 0)
    assert result.finished
    assert result.score == pytest.approx(sum(result.scores))


def test_label_inference_requires_opt_in_for_empty_input():
    with pytest.raises(ValueError, match="allow_empty_input=True"):
        greedy_label_inference(
            BranchingCompactModel(),
            _vocabulary(),
            [],
            max_new_tokens=3,
        )


def test_label_inference_can_decode_from_empty_input_when_enabled():
    vocab = _vocabulary()

    greedy = greedy_label_inference(
        EmptyPrefixCompactModel(),
        vocab,
        [],
        max_new_tokens=3,
        allow_empty_input=True,
    )
    beam = beam_label_inference(
        EmptyPrefixCompactModel(),
        vocab,
        [],
        max_new_tokens=1,
        beam_size=2,
        num_return_sequences=2,
        allow_empty_input=True,
    )
    sample = sample_label_inference(
        EmptyPrefixCompactModel(),
        vocab,
        [],
        max_new_tokens=3,
        top_k=1,
        generator=torch.Generator().manual_seed(3),
        allow_empty_input=True,
    )

    assert greedy.labels == (1, 0)
    assert greedy.token_ids == (1, 0)
    assert len(beam.candidates) == 2
    assert sample.labels == (1, 0)


def test_beam_label_inference_returns_ranked_candidates():
    result = beam_label_inference(
        BranchingCompactModel(),
        _vocabulary(),
        [9],
        max_new_tokens=1,
        beam_size=2,
        early_stopping=False,
        num_return_sequences=2,
    )

    assert result.labels == result.candidates[0].labels
    assert len(result.candidates) == 2
    assert {candidate.labels[0] for candidate in result.candidates} == {1, 2}
    assert result.candidates[0].normalized_score() >= result.candidates[1].normalized_score()
    assert all(3 not in candidate.labels for candidate in result.candidates)


def test_beam_label_inference_validates_search_arguments():
    vocab = _vocabulary()
    model = BranchingCompactModel()

    with pytest.raises(ValueError, match="beam_size"):
        beam_label_inference(model, vocab, [9], max_new_tokens=1, beam_size=0)
    with pytest.raises(ValueError, match="length_penalty"):
        beam_label_inference(model, vocab, [9], max_new_tokens=1, length_penalty=0.0)
    with pytest.raises(ValueError, match="num_return_sequences"):
        beam_label_inference(model, vocab, [9], max_new_tokens=1, num_return_sequences=0)


def test_sample_label_inference_is_seeded_and_supports_top_k_filtering():
    vocab = _vocabulary()

    first = sample_label_inference(
        BranchingCompactModel(),
        vocab,
        [9],
        max_new_tokens=3,
        top_k=1,
        generator=torch.Generator().manual_seed(7),
    )
    second = sample_label_inference(
        BranchingCompactModel(),
        vocab,
        [9],
        max_new_tokens=3,
        top_k=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert first.labels == second.labels == (1, 0)
    assert first.symbols == second.symbols == ("A", "<eos>")
    assert first.score == pytest.approx(second.score)


def test_sample_label_inference_validates_sampling_arguments():
    vocab = _vocabulary()
    model = BranchingCompactModel()

    with pytest.raises(ValueError, match="temperature"):
        sample_label_inference(model, vocab, [9], max_new_tokens=1, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        sample_label_inference(model, vocab, [9], max_new_tokens=1, top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        sample_label_inference(model, vocab, [9], max_new_tokens=1, top_p=0.0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_label_inference(model, vocab, [9], max_new_tokens=-1)


def test_label_inference_forwards_next_label_kwargs():
    vocab = _vocabulary()
    model = KwargCompactModel()

    result = greedy_label_inference(
        model,
        vocab,
        [9],
        max_new_tokens=1,
        next_label_kwargs={"bias_label": 2},
    )

    assert result.labels == (2,)
    assert model.seen_biases == [2]


def test_compact_label_head_exposes_method_style_inference():
    model = MethodCompactModel()
    vocab = _vocabulary()

    greedy = model.greedy_label_inference(vocab, [9], max_new_tokens=3)
    beam = model.beam_label_inference(vocab, [9], max_new_tokens=1, beam_size=2, num_return_sequences=2)
    sample = model.sample_label_inference(
        vocab,
        [9],
        max_new_tokens=3,
        top_k=1,
        generator=torch.Generator().manual_seed(3),
    )

    assert greedy.labels == (1, 0)
    assert len(beam.candidates) == 2
    assert sample.labels == (1, 0)
