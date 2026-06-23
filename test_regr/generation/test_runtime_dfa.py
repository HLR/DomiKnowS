from __future__ import annotations

from types import SimpleNamespace

import torch

from domiknows.generation import (
    DiscreteHMM,
    HuggingFaceGenerationAdapter,
    HybridController,
    TokenVocabulary,
    accept_all_dfa,
    compose_runtime_dfa,
    pending_token_allowed_set_overlay,
    token_class_sequence_overlay,
    token_set_sequence_overlay,
)


class TinyTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.map = {"<eos>": 0, "act": 1, "obj": 2, "bad": 3}
        self.inverse = {value: key for key, value in self.map.items()}

    def encode(self, token):
        return [self.map[token]]

    def decode(self, token_ids):
        return "".join(self.inverse[int(token_id)] for token_id in token_ids)


class MappingTokenizer:
    eos_token_id = 0

    def __init__(self, tokens):
        self.map = {token: index for index, token in enumerate(tokens)}
        self.inverse = {value: key for key, value in self.map.items()}

    def encode(self, token):
        return [self.map[token]]

    def decode(self, token_ids):
        return "".join(self.inverse[int(token_id)] for token_id in token_ids)


class BackendLogitModel(torch.nn.Module):
    def forward(self, input_ids):
        logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def _runtime_vocab_and_labels():
    tokenizer = TinyTokenizer()
    vocab = TokenVocabulary(["<eos>", "act", "obj", "bad"], eos_token="<eos>", tokenizer=tokenizer)
    return tokenizer, vocab, vocab.label_for_token("act"), vocab.label_for_token("obj"), vocab.label_for_token("bad")


def test_composed_runtime_dfa_exposes_product_metadata_and_filters_tokens():
    _tokenizer, vocab, act, obj, bad = _runtime_vocab_and_labels()
    base = accept_all_dfa(vocab)
    dfa = compose_runtime_dfa(
        base,
        [
            token_class_sequence_overlay(["act"], ["obj"], "<eos>", vocabulary=vocab),
            pending_token_allowed_set_overlay({"act": ["obj"]}, vocabulary=vocab),
        ],
    )

    assert dfa.start_state in dfa.states
    assert dfa.accepting_states <= dfa.states
    assert dfa.dead_states <= dfa.states
    assert dfa.allowed_tokens(dfa.start_state, remaining_steps=3) == {act}

    after_act = dfa.step(dfa.start_state, act)
    assert after_act in dfa.states
    assert dfa.allowed_tokens(after_act, remaining_steps=2) == {obj}
    assert dfa.step(after_act, bad) is None
    assert dfa.accepts([act, obj])
    assert dfa.accepts([act, obj, vocab.eos_label])
    assert not dfa.accepts([act, bad, vocab.eos_label])

    label_only = compose_runtime_dfa(
        base,
        [
            token_class_sequence_overlay([act], [obj], vocab.eos_label, alphabet=vocab.alphabet),
            pending_token_allowed_set_overlay({act: [obj]}, alphabet=vocab.alphabet),
        ],
    )
    assert label_only.accepts([act, obj, vocab.eos_label])

    action_only = compose_runtime_dfa(
        base,
        [token_set_sequence_overlay([act], vocab.eos_label, vocabulary=vocab)],
    )
    assert action_only.accepts([act])
    assert action_only.accepts([act, act, vocab.eos_label])
    assert not action_only.accepts([vocab.eos_label])
    assert not action_only.accepts([act, obj, vocab.eos_label])


def test_composed_runtime_dfa_works_with_hmm_static_lookahead():
    tokenizer, vocab, act, obj, _bad = _runtime_vocab_and_labels()
    base = accept_all_dfa(vocab)
    dfa = compose_runtime_dfa(
        base,
        [
            token_class_sequence_overlay([act], [obj], vocab.eos_label, vocabulary=vocab),
            pending_token_allowed_set_overlay({act: [obj]}, vocabulary=vocab),
        ],
    )
    label_count = int(vocab.label_count)
    emission = torch.zeros((3, label_count), dtype=torch.float32)
    emission[0, act] = 1.0
    emission[1, obj] = 1.0
    emission[2, vocab.eos_label] = 1.0
    hmm = DiscreteHMM(
        transition=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        emission=emission,
        initial=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        symbols=tuple(range(label_count)),
        normalize=False,
    )
    generator = HuggingFaceGenerationAdapter(BackendLogitModel(), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(
        torch.tensor([[9]], dtype=torch.long),
        search="beam",
        beam_size=2,
        max_new_tokens=3,
        lookahead_weight=1.0,
        hf_weight=0.0,
    )

    assert results[0].metadata["lookahead_backend"] == "static_dp"
    assert results[0].labels == [act, obj, vocab.eos_label]
    assert results[0].accepted


def test_lazy_static_lookahead_avoids_full_composed_state_table():
    action_count = 20
    tokens = ["<eos>"]
    actions = [f"act{i}" for i in range(action_count)]
    objects = [f"obj{i}" for i in range(action_count)]
    tokens.extend(actions)
    tokens.extend(objects)
    tokenizer = MappingTokenizer(tokens)
    vocab = TokenVocabulary(tokens, eos_token="<eos>", tokenizer=tokenizer)
    action_labels = [vocab.label_for_token(token) for token in actions]
    object_labels = [vocab.label_for_token(token) for token in objects]
    base = accept_all_dfa(vocab)
    dfa = compose_runtime_dfa(
        base,
        [
            token_class_sequence_overlay(actions, objects, "<eos>", vocabulary=vocab),
            pending_token_allowed_set_overlay(
                {action: [obj] for action, obj in zip(actions, objects)},
                vocabulary=vocab,
            ),
        ],
    )
    label_count = int(vocab.label_count)
    emission = torch.zeros((3, label_count), dtype=torch.float32)
    emission[0, action_labels[0]] = 1.0
    emission[1, object_labels[0]] = 1.0
    emission[2, vocab.eos_label] = 1.0
    hmm = DiscreteHMM(
        transition=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        emission=emission,
        initial=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        symbols=tuple(range(label_count)),
        normalize=False,
    )
    controller = HybridController(
        generator=HuggingFaceGenerationAdapter(BackendLogitModel(), tokenizer, vocab),
        vocabulary=vocab,
        dfa=dfa,
        scorer_head=hmm,
        tokenizer=tokenizer,
    )

    results = controller.decode_hmm_dfa(
        torch.tensor([[99]], dtype=torch.long),
        search="beam",
        beam_size=2,
        max_new_tokens=3,
        lookahead_weight=1.0,
        lookahead_max_steps=3,
        hf_weight=0.0,
    )

    metadata = results[0].metadata
    full_table_entries = len(dfa.states) * int(metadata["lookahead_depth"])
    assert results[0].labels == [action_labels[0], object_labels[0], vocab.eos_label]
    assert metadata["lookahead_backend"] == "static_dp"
    assert metadata["lookahead_dfa_states"] == len(dfa.states)
    assert metadata["lookahead_entries"] < full_table_entries // 2
