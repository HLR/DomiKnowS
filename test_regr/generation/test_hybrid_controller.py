from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest
import torch

from domiknows.generation import (
    ConstraintBundle,
    GenerationCandidate,
    GraphHMMGenerationHead,
    HybridController,
    HybridScoreWeights,
    HuggingFaceGenerationAdapter,
    DiscreteHMM,
    HMMGenerationHead,
    LatentLossBreakdown,
    ManualConstraintSelector,
    DFA,
    TokenVocabulary,
    discover_generation_enforcement,
    eos_closure_dfa,
    forbidden_token_dfa,
    product_dfa,
    required_token_dfa,
)
from domiknows.generation.dfa.stop_policy import StopPolicy
from domiknows.generation.applications.hmm_dfa_decoder import HMMDFADecoder


def import_hf_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


class StepwiseScorer(torch.nn.Module):
    def __init__(self, logits_by_prefix, label_to_token_id):
        super().__init__()
        self.logits_by_prefix = {tuple(key): torch.tensor(value, dtype=torch.float32) for key, value in logits_by_prefix.items()}
        self.label_to_token_id = tuple(label_to_token_id)
        self.label_count = len(label_to_token_id)

    def token_id_for_label(self, label):
        token_id = self.label_to_token_id[int(label)]
        if token_id is None:
            raise ValueError("not emittable")
        return int(token_id)

    def next_label_logits(self, input_ids):
        ids = tuple(int(item) for item in input_ids.reshape(-1).tolist())
        return self.logits_by_prefix.get(ids, torch.zeros(self.label_count))


class SequenceScorer(torch.nn.Module):
    def __init__(self, label_count: int):
        super().__init__()
        self.label_count = label_count

    def sequence_log_probs(self, labels):
        labels = labels.long().reshape(-1)
        rows = torch.full((labels.numel(), self.label_count), -5.0)
        for index, label in enumerate(labels.tolist()):
            rows[index, int(label)] = -0.1
        return torch.log_softmax(rows, dim=-1)


class TinyTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.map = {"<eos>": 0, "A": 1, "B": 2}
        self.inverse = {value: key for key, value in self.map.items()}

    def encode(self, text):
        return [self.map[text]]

    def decode(self, token_ids):
        return "".join(self.inverse[int(token_id)] for token_id in token_ids)

    def __call__(self, _text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[9]], dtype=torch.long))


class BackendLogitModel(torch.nn.Module):
    def __init__(self, logits_by_prefix):
        super().__init__()
        self.logits_by_prefix = {
            tuple(key): torch.tensor(value, dtype=torch.float32)
            for key, value in logits_by_prefix.items()
        }

    def forward(self, input_ids):
        ids = tuple(int(item) for item in input_ids.reshape(-1).tolist())
        logits = self.logits_by_prefix.get(ids, torch.zeros(3, dtype=torch.float32))
        return SimpleNamespace(logits=logits.reshape(1, 1, -1))


def _ab_vocab_and_dfa(*, allow_b: bool = True):
    tokenizer = TinyTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    start, saw_a, saw_b, done = "start", "saw_a", "saw_b", "done"
    transitions = {
        (start, vocab.label_for_token("A")): saw_a,
        (saw_a, vocab.eos_label): done,
    }
    if allow_b:
        transitions[(start, vocab.label_for_token("B"))] = saw_b
        transitions[(saw_b, vocab.eos_label)] = done
    dfa = DFA(
        states=frozenset({start, saw_a, saw_b, done}),
        alphabet=frozenset(vocab.alphabet),
        transitions=transitions,
        start_state=start,
        accepting_states=frozenset({done}),
    )
    return tokenizer, vocab, dfa


def build_controller(scorer=None, *, enforcement=None, weights=None):
    graph_module = import_hf_module("graph")
    mock_hf = import_hf_module("mock_hf")
    tokenizer = mock_hf.MockTokenizer()
    graph, bundle = graph_module.build_generation_graph(tokenizer)
    discovered = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = discovered.dfa
    scorer = scorer or SequenceScorer(bundle.vocabulary.label_count)
    return HybridController(
        vocabulary=bundle.vocabulary,
        dfa=dfa,
        scorer_head=scorer,
        enforcement=enforcement,
        tokenizer=tokenizer,
        weights=weights,
        constraints=discovered.dfa_constraints,
    ), tokenizer


def test_valid_candidates_rank_above_rejected_by_default():
    controller, tokenizer = build_controller()
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    ranked = controller.rerank_candidates(
        prompt_ids,
        [
            GenerationCandidate(text=" The dog<eos>", token_ids=[1, 4, 0], labels=[1, 4, 0]),
            GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]),
        ],
        keep_rejected=True,
        explain=True,
    )

    assert ranked[0].score.accepted
    assert ranked[0].candidate.labels == [1, 2, 0]
    assert not ranked[1].score.accepted
    assert ranked[1].score.rejection


def test_stepwise_compact_head_scoring_path_is_used():
    graph_module = import_hf_module("graph")
    mock_hf = import_hf_module("mock_hf")
    tokenizer = mock_hf.MockTokenizer()
    _graph, bundle = graph_module.build_generation_graph(tokenizer)
    scorer = StepwiseScorer(
        {
            (5,): [-5, 5, 1, 0, -5, -5],
            (5, 1): [-5, -5, 5, 0, -5, -5],
            (5, 1, 2): [5, -5, -5, 0, -5, -5],
        },
        [0, 1, 2, 3, 4, None],
    )
    controller, _tokenizer = build_controller(scorer)
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    good = controller.score_candidate(prompt_ids, GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]))
    poor = controller.score_candidate(prompt_ids, GenerationCandidate(text=" cat The<eos>", token_ids=[2, 1, 0], labels=[2, 1, 0]))

    assert good.head_logprob > poor.head_logprob


def test_product_compact_learner_dfa_masks_blocked_compact_labels():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer(
        {
            (9,): [-5, 2, 9, -5],
            (9, 1): [9, -5, -5, -5],
        },
        [0, 1, 2, None],
    )
    generator = HuggingFaceGenerationAdapter(
        BackendLogitModel({(9,): [0, 0, 0], (9, 1): [0, 0, 0]}),
        tokenizer,
        vocab,
    )
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert ranked[0].score.accepted
    assert ranked[0].candidate.raw.final_state == "done"


def test_product_compact_learner_dfa_can_combine_backend_llm_logits():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer(
        {
            (9,): [-5, 5, 0, -5],
            (9, 2): [9, -5, -5, -5],
        },
        [0, 1, 2, None],
    )
    generator = HuggingFaceGenerationAdapter(
        BackendLogitModel({(9,): [-5, 0, 10], (9, 2): [10, 0, 0]}),
        tokenizer,
        vocab,
    )
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=2,
        temperature=0.0,
        backend_logit_weight=1.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("B"), vocab.eos_label]
    assert ranked[0].score.accepted


def test_product_compact_learner_dfa_does_not_stop_on_accepting_prefix():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    dfa = DFA(
        states=dfa.states,
        alphabet=dfa.alphabet,
        transitions=dfa.transitions,
        start_state=dfa.start_state,
        accepting_states=frozenset({"saw_a", "done"}),
    )
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer(
        {
            (9,): [-5, 9, -5, -5],
            (9, 1): [9, -5, -5, -5],
        },
        [0, 1, 2, None],
    )
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert ranked[0].score.accepted


def test_product_compact_learner_dfa_honors_stop_policy():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    dfa = DFA(
        states=dfa.states,
        alphabet=dfa.alphabet,
        transitions=dfa.transitions,
        start_state=dfa.start_state,
        accepting_states=frozenset({"saw_a", "done"}),
    )
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer({(9,): [-5, 9, -5, -5]}, [0, 1, 2, None])
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=None,
        stop_policy=StopPolicy(stop_on_accepting_state=True),
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A")]
    assert ranked[0].score.accepted


def test_product_compact_learner_dfa_omits_rejected_candidates_by_default():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer({(9,): [-5, 9, -5, -5]}, [0, 1, 2, None])
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=1,
        temperature=0.0,
    )
    rejected = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_compact_learner_dfa",
        max_new_tokens=1,
        temperature=0.0,
        keep_rejected=True,
    )

    assert ranked == []
    assert rejected
    assert not rejected[0].score.accepted


def _strict_hmm_for_ab_path():
    # Hidden state S0 emits A, then transitions to S1, which emits EOS.
    return DiscreteHMM(
        transition=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        emission=torch.tensor([[0.0, 0.9, 0.1], [1.0, 0.0, 0.0]], dtype=torch.float32),
        initial=torch.tensor([1.0, 0.0], dtype=torch.float32),
        symbols=(0, 1, 2),
        normalize=False,
    )


def _hmm_prefers_blocked_b_path():
    # The HMM's unconstrained first-step argmax is B, but the DFA tests below
    # make B non-productive. Deterministic decoding must still choose A.
    return DiscreteHMM(
        transition=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        emission=torch.tensor([[0.0, 0.01, 0.99], [1.0, 0.0, 0.0]], dtype=torch.float32),
        initial=torch.tensor([1.0, 0.0], dtype=torch.float32),
        symbols=(0, 1, 2),
        normalize=False,
    )


def _dfa_with_dead_b_branch():
    tokenizer = TinyTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    start, saw_a, done, bad = "start", "saw_a", "done", "bad"
    dfa = DFA(
        states=frozenset({start, saw_a, done, bad}),
        alphabet=frozenset(vocab.alphabet),
        transitions={
            (start, vocab.label_for_token("A")): saw_a,
            (start, vocab.label_for_token("B")): bad,
            (saw_a, vocab.eos_label): done,
            (bad, vocab.eos_label): bad,
            (bad, vocab.label_for_token("A")): bad,
            (bad, vocab.label_for_token("B")): bad,
        },
        start_state=start,
        accepting_states=frozenset({done}),
        dead_states=frozenset({bad}),
    )
    return tokenizer, vocab, dfa


def _lookahead_branch_hmm():
    return DiscreteHMM(
        transition=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        emission=torch.tensor(
            [
                [0.0, 0.99, 0.01],
                [0.0, 0.01, 0.99],
                [0.99, 0.005, 0.005],
                [0.01, 0.495, 0.495],
            ],
            dtype=torch.float32,
        ),
        initial=torch.tensor([0.45, 0.55, 0.0, 0.0], dtype=torch.float32),
        symbols=(0, 1, 2),
        normalize=False,
    )


def _dfa_with_two_accepting_branches():
    tokenizer = TinyTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    start, after_a, after_b, done = "start", "after_a", "after_b", "done"
    dfa = DFA(
        states=frozenset({start, after_a, after_b, done}),
        alphabet=frozenset(vocab.alphabet),
        transitions={
            (start, vocab.label_for_token("A")): after_a,
            (start, vocab.label_for_token("B")): after_b,
            (after_a, vocab.eos_label): done,
            (after_b, vocab.eos_label): done,
        },
        start_state=start,
        accepting_states=frozenset({done}),
    )
    return tokenizer, vocab, dfa


def _hmm_prefers_accepting_prefix_without_eos():
    return DiscreteHMM(
        transition=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        emission=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        initial=torch.tensor([0.9, 0.1, 0.0, 0.0], dtype=torch.float32),
        symbols=(0, 1, 2),
        normalize=False,
    )


def _dfa_accepting_prefix_requires_eos_for_stop():
    tokenizer = TinyTokenizer()
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    start, after_a, after_b, done = "start", "after_a", "after_b", "done"
    dfa = DFA(
        states=frozenset({start, after_a, after_b, done}),
        alphabet=frozenset(vocab.alphabet),
        transitions={
            (start, vocab.label_for_token("A")): after_a,
            (start, vocab.label_for_token("B")): after_b,
            (after_b, vocab.eos_label): done,
        },
        start_state=start,
        accepting_states=frozenset({after_a, done}),
    )
    return tokenizer, vocab, dfa


def _branchy_abc_vocab_dfa_and_hmm():
    tokenizer = TinyTokenizer()
    tokenizer.map = {"<eos>": 0, "A": 1, "B": 2, "C": 3}
    tokenizer.inverse = {value: key for key, value in tokenizer.map.items()}
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>", tokenizer=tokenizer)
    transitions = {}
    states = {f"q{index}" for index in range(6)} | {"done"}
    for index in range(5):
        state = f"q{index}"
        for token in ("A", "B", "C"):
            transitions[(state, vocab.label_for_token(token))] = f"q{index + 1}"
        if index >= 1:
            transitions[(state, vocab.eos_label)] = "done"
    transitions[("q5", vocab.eos_label)] = "done"
    dfa = DFA(
        states=frozenset(states),
        alphabet=frozenset(vocab.alphabet),
        transitions=transitions,
        start_state="q0",
        accepting_states=frozenset({"done"}),
    )
    hmm = DiscreteHMM(
        transition=torch.eye(8, dtype=torch.float32),
        emission=torch.full((8, 4), 0.25, dtype=torch.float32),
        initial=torch.ones(8, dtype=torch.float32) / 8,
        symbols=(0, 1, 2, 3),
        normalize=False,
    )
    return tokenizer, vocab, dfa, hmm


def test_product_hmm_dfa_tracks_explicit_discrete_hmm_belief():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert ranked[0].score.accepted
    assert ranked[0].candidate.metadata["tracks_hmm_belief"] is True
    assert torch.allclose(ranked[0].candidate.metadata["final_hmm_belief"], torch.tensor([0.0, 1.0]))


def test_decode_hmm_dfa_greedy_returns_one_deterministic_result():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    first = controller.decode_hmm_dfa(prompt_ids, search="greedy", max_new_tokens=2)
    second = controller.decode_hmm_dfa(prompt_ids, search="greedy", max_new_tokens=2)

    assert len(first) == 1
    assert first[0].search == "greedy"
    assert first[0].labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert first[0].accepted
    assert second[0].labels == first[0].labels
    assert torch.allclose(first[0].final_hmm_belief, torch.tensor([0.0, 1.0]))


def test_decode_hmm_dfa_sample_returns_dfa_valid_results():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = DiscreteHMM(
        transition=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        emission=torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=torch.float32),
        initial=torch.tensor([1.0, 0.0], dtype=torch.float32),
        symbols=(0, 1, 2),
        normalize=False,
    )
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="sample",
        num_return_sequences=4,
        max_new_tokens=2,
        temperature=1.0,
        generator_seed=17,
        product_decode_max_attempts=8,
    )

    assert len(results) == 4
    assert all(result.search == "sample" for result in results)
    assert all(result.accepted for result in results)
    assert all(dfa.accepts(result.labels) for result in results)


def test_decode_hmm_dfa_beam_returns_best_accepted_result():
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )

    assert len(results) == 1
    assert results[0].search == "beam"
    assert results[0].labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert results[0].accepted


def test_decode_hmm_dfa_lookahead_weight_changes_ranking():
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    immediate_only = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=0.0,
    )
    with_lookahead = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )

    assert immediate_only[0].labels == [vocab.label_for_token("B"), vocab.eos_label]
    assert with_lookahead[0].labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert immediate_only[0].accepted
    assert with_lookahead[0].accepted


def test_decode_hmm_dfa_static_dp_matches_recursive_lookahead(monkeypatch):
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    static_result = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )
    monkeypatch.setattr(HMMDFADecoder, "_build_static_lookahead", lambda *_args, **_kwargs: None)
    recursive_result = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )

    assert static_result[0].metadata["lookahead_backend"] == "static_dp"
    assert recursive_result[0].metadata["lookahead_backend"] == "recursive"
    assert static_result[0].labels == recursive_result[0].labels
    assert torch.allclose(torch.tensor(static_result[0].scores), torch.tensor(recursive_result[0].scores))


def test_decode_hmm_dfa_static_dp_avoids_recursive_success_probability(monkeypatch):
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    def fail_recursive(*_args, **_kwargs):
        raise AssertionError("static HMM+DFA lookahead should use the DP table")

    monkeypatch.setattr(HMMDFADecoder, "_success_probability", fail_recursive)
    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )

    assert results[0].metadata["lookahead_backend"] == "static_dp"
    assert results[0].accepted


def test_decode_hmm_dfa_static_vectorized_updates_match_scalar_fallback(monkeypatch):
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    vectorized = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )
    monkeypatch.setattr(HMMDFADecoder, "_static_next_beliefs", staticmethod(lambda *_args, **_kwargs: {}))
    scalar = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=2.0,
    )

    assert vectorized[0].labels == scalar[0].labels
    assert torch.allclose(vectorized[0].final_hmm_belief, scalar[0].final_hmm_belief)
    assert torch.allclose(torch.tensor(vectorized[0].scores), torch.tensor(scalar[0].scores))


def test_decode_hmm_dfa_caches_dfa_allowed_transitions(monkeypatch):
    tokenizer, vocab, dfa, hmm = _branchy_abc_vocab_dfa_and_hmm()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)
    calls = 0
    original = DFA.allowed_tokens

    def spy_allowed_tokens(self, state, remaining_steps=None):
        nonlocal calls
        if self is dfa:
            calls += 1
        return original(self, state, remaining_steps=remaining_steps)

    monkeypatch.setattr(DFA, "allowed_tokens", spy_allowed_tokens)
    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=4,
        max_new_tokens=6,
        lookahead_weight=1.0,
        hf_weight=0.0,
    )

    assert results[0].metadata["lookahead_backend"] == "static_dp"
    assert results[0].accepted
    assert calls <= len(dfa.states) * 6


@pytest.mark.benchmark
def test_decode_hmm_dfa_static_dp_is_faster_than_recursive_lookahead(monkeypatch):
    tokenizer, vocab, dfa, hmm = _branchy_abc_vocab_dfa_and_hmm()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    def decode():
        return controller.decode_hmm_dfa(
            prompt_ids,
            search="beam",
            beam_size=4,
            max_new_tokens=6,
            lookahead_weight=1.0,
            hf_weight=0.0,
        )

    static_warmup = decode()
    assert static_warmup[0].metadata["lookahead_backend"] == "static_dp"
    assert static_warmup[0].accepted

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        static_result = decode()
    static_time = time.perf_counter() - start

    monkeypatch.setattr(HMMDFADecoder, "_build_static_lookahead", lambda *_args, **_kwargs: None)
    recursive_warmup = decode()
    assert recursive_warmup[0].metadata["lookahead_backend"] == "recursive"
    assert recursive_warmup[0].accepted

    start = time.perf_counter()
    for _ in range(runs):
        recursive_result = decode()
    recursive_time = time.perf_counter() - start

    speedup = recursive_time / static_time if static_time else float("inf")
    print(
        "\nHMM+DFA lookahead benchmark "
        f"({runs} runs): static_dp={static_time:.4f}s "
        f"recursive={recursive_time:.4f}s speedup={speedup:.2f}x"
    )

    assert static_result[0].labels == recursive_result[0].labels
    assert static_result[0].accepted
    assert recursive_result[0].accepted
    assert static_time <= recursive_time * 0.75


def test_decode_hmm_dfa_caches_backend_logits_for_repeated_sample_prefixes():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    backend = BackendLogitModel({(9,): [0, 1, 0], (9, 1): [1, 0, 0]})
    backend.calls = 0
    original_forward = backend.forward

    def counted_forward(input_ids):
        backend.calls += 1
        return original_forward(input_ids)

    backend.forward = counted_forward
    generator = HuggingFaceGenerationAdapter(backend, tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="sample",
        num_return_sequences=3,
        max_new_tokens=2,
        temperature=0.0,
        hf_weight=1.0,
        product_decode_max_attempts=3,
    )

    assert len(results) == 3
    assert all(result.accepted for result in results)
    assert backend.calls == 2


def test_decode_hmm_dfa_keep_rejected_returns_output_when_no_acceptance_possible():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    without_rejected = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        max_new_tokens=1,
        keep_rejected=False,
    )
    with_rejected = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        max_new_tokens=1,
        keep_rejected=True,
    )

    assert without_rejected == []
    assert len(with_rejected) == 1
    assert not with_rejected[0].accepted


def test_decode_hmm_dfa_stop_policy_requires_eos_and_accepting_by_default():
    tokenizer, vocab, dfa = _dfa_accepting_prefix_requires_eos_for_stop()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _hmm_prefers_accepting_prefix_without_eos()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(
        prompt_ids,
        search="beam",
        beam_size=2,
        max_new_tokens=2,
        lookahead_weight=5.0,
    )

    assert results[0].labels == [vocab.label_for_token("B"), vocab.eos_label]
    assert results[0].accepted
    assert results[0].final_state == "done"


def test_decode_hmm_dfa_accepts_graph_hmm_generation_head():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=3,
        transition_mask=torch.ones((2, 2), dtype=torch.float32),
        emission_mask=torch.ones((2, 3), dtype=torch.float32),
        label_to_token_id=[0, 1, 2],
        trainable=False,
        initial=hmm.initial_probs,
        transition=hmm.transition_probs,
        emission=hmm.emission_probs,
    )
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=head, tokenizer=tokenizer)

    results = controller.decode_hmm_dfa(prompt_ids, search="beam", max_new_tokens=2)

    assert results[0].labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert results[0].accepted


def test_product_hmm_dfa_deterministically_avoids_unacceptable_dfa_states():
    tokenizer, vocab, dfa = _dfa_with_dead_b_branch()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _hmm_prefers_blocked_b_path()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    expected = [vocab.label_for_token("A"), vocab.eos_label]
    for seed in range(5):
        ranked = controller.generate_verify_rerank(
            prompt_ids,
            1,
            decode_strategy="product_hmm_dfa",
            max_new_tokens=2,
            temperature=0.0,
            generator_seed=seed,
        )

        assert ranked[0].candidate.labels == expected
        assert ranked[0].score.accepted
        assert ranked[0].candidate.raw.final_state == "done"
        assert vocab.label_for_token("B") not in ranked[0].candidate.labels


def test_product_hmm_dfa_lookahead_scores_future_acceptance_mass():
    tokenizer, vocab, dfa = _dfa_with_two_accepting_branches()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _lookahead_branch_hmm()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    immediate_only = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
        lookahead_weight=0.0,
    )
    with_lookahead = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
        lookahead_weight=2.0,
    )

    assert immediate_only[0].candidate.labels == [vocab.label_for_token("B"), vocab.eos_label]
    assert with_lookahead[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert immediate_only[0].score.accepted
    assert with_lookahead[0].score.accepted


def test_product_hmm_dfa_lookahead_requires_eos_for_default_success():
    tokenizer, vocab, dfa = _dfa_accepting_prefix_requires_eos_for_stop()
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _hmm_prefers_accepting_prefix_without_eos()
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=hmm, tokenizer=tokenizer)

    immediate_only = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
        lookahead_weight=0.0,
    )
    with_lookahead = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
        lookahead_weight=5.0,
    )

    assert immediate_only[0].candidate.labels == [vocab.label_for_token("A")]
    assert with_lookahead[0].candidate.labels == [vocab.label_for_token("B"), vocab.eos_label]
    assert with_lookahead[0].score.accepted


def test_product_hmm_dfa_accepts_hmm_generation_head():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    head = HMMGenerationHead(
        label_count=3,
        state_count=2,
        label_to_token_id=[0, 1, 2],
        trainable=False,
    )
    hmm = _strict_hmm_for_ab_path()
    with torch.no_grad():
        head.initial_logits.copy_(torch.log(hmm.initial_probs.clamp_min(1e-6)))
        head.transition_logits.copy_(torch.log(hmm.transition_probs.clamp_min(1e-6)))
        head.emission_logits.copy_(torch.log(hmm.emission_probs.clamp_min(1e-6)))
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=head, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert ranked[0].score.accepted


def test_product_hmm_dfa_accepts_graph_hmm_generation_head():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    hmm = _strict_hmm_for_ab_path()
    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=3,
        transition_mask=torch.ones((2, 2), dtype=torch.float32),
        emission_mask=torch.ones((2, 3), dtype=torch.float32),
        label_to_token_id=[0, 1, 2],
        trainable=False,
        initial=hmm.initial_probs,
        transition=hmm.transition_probs,
        emission=hmm.emission_probs,
    )
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=head, tokenizer=tokenizer)

    ranked = controller.generate_verify_rerank(
        prompt_ids,
        1,
        decode_strategy="product_hmm_dfa",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert ranked[0].candidate.labels == [vocab.label_for_token("A"), vocab.eos_label]
    assert ranked[0].score.accepted


def test_product_hmm_dfa_rejects_generic_compact_scorer():
    tokenizer, vocab, dfa = _ab_vocab_and_dfa(allow_b=False)
    prompt_ids = torch.tensor([[9]], dtype=torch.long)
    scorer = StepwiseScorer({(9,): [-5, 9, -5, -5]}, [0, 1, 2, None])
    generator = HuggingFaceGenerationAdapter(BackendLogitModel({}), tokenizer, vocab)
    controller = HybridController(generator=generator, vocabulary=vocab, dfa=dfa, scorer_head=scorer, tokenizer=tokenizer)

    try:
        controller.generate_verify_rerank(
            prompt_ids,
            1,
            decode_strategy="product_hmm_dfa",
            max_new_tokens=2,
            temperature=0.0,
        )
    except ValueError as exc:
        assert "HMMGenerationHead, GraphHMMGenerationHead, or DiscreteHMM" in str(exc)
    else:
        raise AssertionError("product_hmm_dfa should reject non-HMM scorer heads")


def test_latent_preference_weight_can_change_ranking():
    def latent_breakdown(probs, **_kwargs):
        # Penalize " mat" probability; candidates without mat should rank higher
        # when latent preference is weighted strongly.
        return LatentLossBreakdown(total=probs[:, 3].mean(), items=())

    enforcement = SimpleNamespace(latent_specs=(object(),), latent_breakdown=latent_breakdown)
    controller, tokenizer = build_controller(
        enforcement=enforcement,
        weights=HybridScoreWeights(head_logprob=0.0, validity=10.0, latent_preference=5.0, risk=0.0),
    )
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    ranked = controller.rerank_candidates(
        prompt_ids,
        [
            GenerationCandidate(text=" The cat mat<eos>", token_ids=[1, 2, 3, 0], labels=[1, 2, 3, 0]),
            GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]),
        ],
        keep_rejected=True,
    )

    assert ranked[0].candidate.labels == [1, 2, 0]


def test_failure_risk_increases_when_scorer_prefers_dfa_blocked_label():
    scorer = StepwiseScorer({(5,): [-5, 0, 0, 0, 8, -5]}, [0, 1, 2, 3, 4, None])
    controller, tokenizer = build_controller(scorer, weights=HybridScoreWeights(risk=1.0))
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    risk = controller.predict_failure_risk(prompt_ids, [])

    assert risk > 0.9


def test_repair_suggestions_find_missing_and_forbidden_tokens():
    controller, tokenizer = build_controller()
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    repair = controller.suggest_repair(
        GenerationCandidate(text=" The dog<eos>", token_ids=[1, 4, 0], labels=[1, 4, 0]),
        prompt_ids=prompt_ids,
    )

    assert not repair["accepted"]
    assert any("add ' cat'" in item for item in repair["suggestions"])


def test_constraint_selector_returns_expected_named_bundle():
    controller, _tokenizer = build_controller(
        weights=HybridScoreWeights(),
    )
    controller.constraint_selector = ManualConstraintSelector({"strict": "strict"}, default="loose")
    vocab = controller.vocabulary
    loose_dfa = product_dfa([eos_closure_dfa(vocab), required_token_dfa(vocab, " cat")])
    strict_dfa = product_dfa(
        [
            eos_closure_dfa(vocab),
            required_token_dfa(vocab, " cat"),
            forbidden_token_dfa(vocab, " dog"),
        ]
    )
    bundles = [
        ConstraintBundle("loose", loose_dfa),
        ConstraintBundle("strict", strict_dfa),
    ]

    selected = controller.select_constraints("use strict constraints", bundles)

    assert selected.name == "strict"
