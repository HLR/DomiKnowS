import torch
import pytest

from domiknows.generation import (
    HuggingFaceGenerationAdapter,
    TokenVocabulary,
    constraints_to_dfa,
    required_token,
)
from domiknows.generation.decoder import (
    constrained_beam_search_decode,
    constrained_greedy_decode,
    constrained_label_beam_search_decode,
    constrained_label_greedy_decode,
    constrained_label_sample_decode,
    constrained_sample_decode,
    mask_label_logits_for_dfa,
)


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.map = {"<eos>": 0, "A": 1, "B": 2}

    def encode(self, token):
        return [self.map[token]]


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class CacheOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class FakeModel:
    def __call__(self, input_ids):
        logits = torch.zeros((1, input_ids.shape[1], 4))
        logits[0, -1, 1] = 10.0
        logits[0, -1, 2] = 1.0
        return FakeOutput(logits)


class FakeLabelModel:
    label_to_token_id = (0, 1, 2, None)

    def next_label_logits(self, input_ids):
        logits = torch.zeros(4)
        logits[1] = 10.0
        logits[2] = 1.0
        logits[3] = 20.0
        return logits

    def token_id_for_label(self, label):
        token_id = self.label_to_token_id[int(label)]
        if token_id is None:
            raise ValueError("label does not map to one token")
        return token_id


class BranchingLabelModel(FakeLabelModel):
    def next_label_logits(self, input_ids):
        logits = torch.full((4,), -10.0)
        if input_ids.shape[1] == 1:
            logits[1] = 10.0
            logits[2] = 9.0
            logits[3] = 20.0
        else:
            logits[0] = 8.0
            logits[1] = 3.0
            logits[2] = 4.0
            logits[3] = 20.0
        return logits


class KwargLabelModel(FakeLabelModel):
    def __init__(self):
        self.seen_kwargs = []

    def next_label_logits(self, input_ids, transition_potential=None):
        self.seen_kwargs.append(transition_potential)
        logits = torch.zeros(4)
        logits[2] = 10.0
        logits[3] = 20.0
        return logits


class BranchingModel:
    def __call__(self, input_ids):
        logits = torch.full((1, input_ids.shape[1], 4), -10.0)
        if input_ids.shape[1] == 1:
            logits[0, -1, 1] = 10.0
            logits[0, -1, 2] = 9.0
        else:
            logits[0, -1, 0] = 4.0
            logits[0, -1, 1] = 1.0
            logits[0, -1, 2] = 1.0
        return FakeOutput(logits)


class CacheAwareModel:
    def __init__(self, vocab_size=10, prompt_len=1):
        self.vocab_size = vocab_size
        self.prompt_len = prompt_len
        self.calls = []

    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        self.calls.append(
            {
                "length": int(input_ids.shape[1]),
                "use_cache": bool(use_cache),
                "has_past": past_key_values is not None,
                "ids": [int(token_id) for token_id in input_ids[0].tolist()],
            }
        )
        prefix = []
        if past_key_values is not None:
            prefix = [int(token_id) for token_id in past_key_values[0].tolist()]
        current = prefix + [int(token_id) for token_id in input_ids[0].tolist()]
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -10.0)
        generated = current[self.prompt_len :]

        if not generated:
            logits[0, -1, 1] = 10.0
            logits[0, -1, 2] = 9.0
        elif 2 not in generated:
            logits[0, -1, 2] = 10.0
            logits[0, -1, 0] = 1.0
        else:
            logits[0, -1, 0] = 10.0
            logits[0, -1, 1] = 1.0
            logits[0, -1, 2] = 1.0

        return CacheOutput(logits, (torch.tensor(current),))


class CacheIgnoringModel(CacheAwareModel):
    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        output = super().__call__(input_ids, use_cache=use_cache, past_key_values=None)
        return FakeOutput(output.logits)


class UncloneableCache:
    __slots__ = ()

    def __deepcopy__(self, memo):
        raise RuntimeError("no clone")


class UncloneableCacheModel(CacheAwareModel):
    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        output = super().__call__(input_ids, use_cache=use_cache, past_key_values=None)
        return CacheOutput(output.logits, UncloneableCache())


def test_constrained_greedy_masks_logits_to_satisfy_dfa():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_greedy_decode(
        FakeModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted


def test_constrained_greedy_uses_kv_cache_when_available():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = CacheAwareModel()

    result = constrained_greedy_decode(
        model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert result.labels == [vocab.label_for_token("A"), vocab.label_for_token("B")]
    assert result.accepted
    assert [call["length"] for call in model.calls] == [1, 1]
    assert [call["has_past"] for call in model.calls] == [False, True]


def test_constrained_greedy_can_disable_kv_cache():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = CacheAwareModel()

    result = constrained_greedy_decode(
        model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    assert result.labels == [vocab.label_for_token("A"), vocab.label_for_token("B")]
    assert [call["length"] for call in model.calls] == [1, 2]
    assert all(not call["use_cache"] for call in model.calls)


def test_constrained_greedy_falls_back_when_model_omits_past_key_values():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = CacheIgnoringModel()

    result = constrained_greedy_decode(
        model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert result.labels == [vocab.label_for_token("A"), vocab.label_for_token("B")]
    assert [call["length"] for call in model.calls] == [1, 2]


def test_mask_label_logits_for_dfa_keeps_only_allowed_labels():
    logits = torch.tensor([1.0, 2.0, 3.0])

    masked = mask_label_logits_for_dfa(logits, {0, 2})

    assert masked[0].item() == 1.0
    assert masked[1].item() < -5e8
    assert masked[2].item() == 3.0


def test_mask_label_logits_for_dfa_rejects_empty_allowed_set():
    with pytest.raises(ValueError, match="every label"):
        mask_label_logits_for_dfa(torch.tensor([1.0, 2.0]), set())


def test_constrained_label_greedy_masks_compact_head_logits_to_satisfy_dfa():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_greedy_decode(
        FakeLabelModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted
    assert result.score is not None


def test_constrained_label_beam_search_masks_invalid_high_logit_label():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_beam_search_decode(
        FakeLabelModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        beam_size=2,
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted
    assert result.score is not None


def test_constrained_label_beam_search_keeps_separate_dfa_states():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_beam_search_decode(
        BranchingLabelModel(),
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        beam_size=2,
        early_stopping=False,
        num_return_sequences=2,
    )

    assert result.accepted
    assert result.candidates is not None
    first_generated = {candidate.labels[0] for candidate in result.candidates}
    assert first_generated == {vocab.label_for_token("A"), vocab.label_for_token("B")}


def test_constrained_label_beam_search_returns_unaccepted_when_no_solution_reached():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_beam_search_decode(
        FakeLabelModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=0,
    )

    assert result.labels == []
    assert not result.accepted


def test_constrained_label_sampling_never_emits_dfa_disallowed_labels():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_sample_decode(
        FakeLabelModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        generator=torch.Generator().manual_seed(13),
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted


def test_constrained_label_sampling_is_deterministic_with_generator():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([], vocab)

    first = constrained_label_sample_decode(
        BranchingLabelModel(),
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        generator=torch.Generator().manual_seed(7),
    )
    second = constrained_label_sample_decode(
        BranchingLabelModel(),
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert first.token_ids == second.token_ids
    assert first.labels == second.labels


def test_constrained_label_sampling_validates_sampling_arguments():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    with pytest.raises(ValueError, match="temperature"):
        constrained_label_sample_decode(FakeLabelModel(), [1], vocab, dfa, 1, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        constrained_label_sample_decode(FakeLabelModel(), [1], vocab, dfa, 1, top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        constrained_label_sample_decode(FakeLabelModel(), [1], vocab, dfa, 1, top_p=0.0)


def test_constrained_label_sampling_filters_after_dfa_mask():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_label_sample_decode(
        FakeLabelModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        top_k=1,
        top_p=0.01,
        generator=torch.Generator().manual_seed(3),
    )

    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted


def test_constrained_label_decoders_forward_next_label_kwargs():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = KwargLabelModel()

    greedy = constrained_label_greedy_decode(
        model,
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        next_label_kwargs={"transition_potential": "soft-block"},
    )
    beam = constrained_label_beam_search_decode(
        model,
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        next_label_kwargs={"transition_potential": "soft-block"},
    )
    sample = constrained_label_sample_decode(
        model,
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        next_label_kwargs={"transition_potential": "soft-block"},
        generator=torch.Generator().manual_seed(1),
    )

    assert greedy.accepted
    assert beam.accepted
    assert sample.accepted
    assert model.seen_kwargs == ["soft-block", "soft-block", "soft-block"]


def test_constrained_beam_search_masks_logits_to_satisfy_dfa():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_beam_search_decode(
        FakeModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        beam_size=2,
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted
    assert result.score is not None


def test_constrained_beam_search_keeps_separate_dfa_states():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_beam_search_decode(
        BranchingModel(),
        torch.tensor([[0]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        beam_size=2,
        early_stopping=False,
        num_return_sequences=2,
    )

    assert result.accepted
    assert result.candidates is not None
    first_generated = {candidate.labels[0] for candidate in result.candidates}
    assert first_generated == {vocab.label_for_token("A"), vocab.label_for_token("B")}


def test_constrained_beam_search_uses_separate_kv_cache_per_beam():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = CacheAwareModel()

    result = constrained_beam_search_decode(
        model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        beam_size=2,
        early_stopping=False,
        num_return_sequences=2,
    )

    assert result.accepted
    assert result.candidates is not None
    assert [call["length"] for call in model.calls] == [1, 1, 1]
    assert model.calls[1]["has_past"]
    assert model.calls[2]["has_past"]


def test_constrained_beam_search_reports_uncloneable_cache():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([], vocab)

    with pytest.raises(ValueError, match="clone past_key_values"):
        constrained_beam_search_decode(
            UncloneableCacheModel(),
            torch.tensor([[9]]),
            vocab,
            dfa,
            max_new_tokens=1,
            eos_token_id=tokenizer.eos_token_id,
            beam_size=2,
        )


def test_constrained_beam_search_returns_unaccepted_when_no_solution_reached():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_beam_search_decode(
        FakeModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=0,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert result.labels == []
    assert not result.accepted


def test_constrained_sampling_masks_invalid_highest_logit():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_sample_decode(
        FakeModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(13),
    )

    assert result.token_ids[-1] == 2
    assert result.labels == [vocab.label_for_token("B")]
    assert result.accepted


def test_constrained_sampling_is_deterministic_with_generator():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([], vocab)

    first = constrained_sample_decode(
        BranchingModel(),
        torch.tensor([[0]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(7),
    )
    second = constrained_sample_decode(
        BranchingModel(),
        torch.tensor([[0]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(7),
    )

    assert first.token_ids == second.token_ids
    assert first.labels == second.labels


def test_constrained_sampling_uses_kv_cache_and_stays_deterministic():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    first_model = CacheAwareModel()
    second_model = CacheAwareModel()

    first = constrained_sample_decode(
        first_model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(7),
    )
    second = constrained_sample_decode(
        second_model,
        torch.tensor([[9]]),
        vocab,
        dfa,
        max_new_tokens=2,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(7),
    )

    assert first.token_ids == second.token_ids
    assert first.labels == second.labels
    assert first.accepted
    assert [call["length"] for call in first_model.calls] == [1, 1]


def test_constrained_sampling_validates_sampling_arguments():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    with pytest.raises(ValueError, match="temperature"):
        constrained_sample_decode(FakeModel(), [1], vocab, dfa, 1, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        constrained_sample_decode(FakeModel(), [1], vocab, dfa, 1, top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        constrained_sample_decode(FakeModel(), [1], vocab, dfa, 1, top_p=0.0)


def test_constrained_sampling_filters_after_dfa_mask():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)

    result = constrained_sample_decode(
        FakeModel(),
        torch.tensor([[1]]),
        vocab,
        dfa,
        max_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        top_k=1,
        top_p=0.01,
        generator=torch.Generator().manual_seed(3),
    )

    assert result.token_ids[-1] == 2
    assert result.accepted


def test_huggingface_adapter_exposes_beam_and_sampling():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    adapter = HuggingFaceGenerationAdapter(FakeModel(), tokenizer, vocab)

    beam = adapter.constrained_beam_search(torch.tensor([[1]]), dfa, max_new_tokens=1, beam_size=2)
    sample = adapter.constrained_sample(
        torch.tensor([[1]]),
        dfa,
        max_new_tokens=1,
        generator=torch.Generator().manual_seed(5),
    )

    assert beam.labels == [vocab.label_for_token("B")]
    assert sample.labels == [vocab.label_for_token("B")]
    assert beam.accepted
    assert sample.accepted


def test_huggingface_adapter_forwards_use_cache_flag():
    tokenizer = FakeTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = constraints_to_dfa([required_token("B")], vocab)
    model = CacheAwareModel()
    adapter = HuggingFaceGenerationAdapter(model, tokenizer, vocab)

    result = adapter.constrained_greedy(torch.tensor([[9]]), dfa, max_new_tokens=2, use_cache=False)

    assert result.accepted
    assert [call["length"] for call in model.calls] == [1, 2]
