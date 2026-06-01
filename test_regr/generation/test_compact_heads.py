from __future__ import annotations

import itertools

import torch

from domiknows.generation.dfa._constraints import (
    accept_all_dfa,
    required_token_dfa,
)
from domiknows.generation.dfa.vocabulary import TokenVocabulary
from domiknows.generation.applications import GenerationCandidate, HybridController
from domiknows.generation.learners import (
    CompactLabelSequenceModel,
    CRFCompactLabelScorer,
    EnergyCompactLabelGenerationHead,
    GRUCompactLabelGenerationHead,
    GraphHMMGenerationHead,
    GraphSpectralGenerationHead,
    HMMGenerationHead,
    NeuralNGramCompactLabelGenerationHead,
    SpectralWFAGenerationHead,
    TransformerCompactLabelGenerationHead,
)


class TinyTokenizer:
    def __init__(self):
        self.map = {"<eos>": 0, "A": 1, "B": 2}

    def encode(self, text):
        return [self.map[text]]


def _vocab():
    return TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=TinyTokenizer())


def _label_to_token_id():
    return [0, 1, 2, None]


def test_existing_and_neural_heads_implement_compact_label_protocol():
    heads = [
            HMMGenerationHead(label_count=4, state_count=2, label_to_token_id=_label_to_token_id(), trainable=False),
            SpectralWFAGenerationHead(label_count=4, state_count=2, label_to_token_id=_label_to_token_id(), trainable=False),
            GraphHMMGenerationHead(n_hidden_states=2, label_count=4, label_to_token_id=_label_to_token_id(), trainable=False),
            GraphSpectralGenerationHead(label_count=4, state_count=2, label_to_token_id=_label_to_token_id(), trainable=False),
        GRUCompactLabelGenerationHead(
                label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            hidden_size=8,
            embedding_dim=6,
            trainable=False,
        ),
        TransformerCompactLabelGenerationHead(
                label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            embedding_dim=8,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            trainable=False,
        ),
        NeuralNGramCompactLabelGenerationHead(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            hidden_size=8,
            embedding_dim=6,
            trainable=False,
        ),
        EnergyCompactLabelGenerationHead(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            hidden_size=8,
            embedding_dim=6,
            trainable=False,
        ),
        CRFCompactLabelScorer(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            embedding_dim=6,
            trainable=False,
        ),
    ]

    for head in heads:
        assert isinstance(head, CompactLabelSequenceModel)
        log_probs = head.sequence_log_probs(torch.tensor([1, 2, 0]))
        assert log_probs.shape == (head.pad_size, head.label_count)
        assert torch.isfinite(log_probs).all()
        logits = head.next_label_logits(torch.tensor([[1]]))
        assert logits.shape == (head.label_count,)
        assert torch.isfinite(logits).all()


def test_gru_compact_head_trainable_parameter_reporting():
    head_classes = [
        lambda trainable: GRUCompactLabelGenerationHead(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            hidden_size=8,
            embedding_dim=6,
            trainable=trainable,
        ),
        lambda trainable: NeuralNGramCompactLabelGenerationHead(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            hidden_size=8,
            embedding_dim=6,
            trainable=trainable,
        ),
        lambda trainable: CRFCompactLabelScorer(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            embedding_dim=6,
            trainable=trainable,
        ),
        lambda trainable: EnergyCompactLabelGenerationHead(
            label_count=4,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            hidden_size=8,
            embedding_dim=6,
            trainable=trainable,
        ),
    ]

    for factory in head_classes:
        assert factory(False).trainable_parameter_names() == []
        assert factory(True).trainable_parameter_names()


def test_gru_compact_head_works_with_label_decoders():
    vocab = _vocab()
    dfa = required_token_dfa(vocab, "B")
    head = GRUCompactLabelGenerationHead(
        label_count=vocab.label_count,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        hidden_size=8,
        embedding_dim=6,
        trainable=True,
    )

    greedy = constrained_label_greedy_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1)
    beam = constrained_label_beam_search_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1, beam_size=2)
    sample = constrained_label_sample_decode(
        head,
        torch.tensor([[9]]),
        vocab,
            max_new_tokens=1,
        generator=torch.Generator().manual_seed(3),
    )

    assert greedy.labels == [vocab.label_for_token("B")]
    assert beam.labels == [vocab.label_for_token("B")]
    assert sample.labels == [vocab.label_for_token("B")]
    assert greedy.accepted and beam.accepted and sample.accepted


def test_transformer_compact_head_works_with_label_decoders():
    vocab = _vocab()
    dfa = required_token_dfa(vocab, "B")
    head = TransformerCompactLabelGenerationHead(
        label_count=vocab.label_count,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        embedding_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        trainable=True,
    )

    greedy = constrained_label_greedy_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1)
    beam = constrained_label_beam_search_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1, beam_size=2)
    sample = constrained_label_sample_decode(
        head,
        torch.tensor([[9]]),
        vocab,
            max_new_tokens=1,
        generator=torch.Generator().manual_seed(3),
    )

    assert greedy.labels == [vocab.label_for_token("B")]
    assert beam.labels == [vocab.label_for_token("B")]
    assert sample.labels == [vocab.label_for_token("B")]
    assert greedy.accepted and beam.accepted and sample.accepted


def test_ngram_and_crf_compact_heads_work_with_label_decoders():
    vocab = _vocab()
    dfa = required_token_dfa(vocab, "B")
    heads = [
        NeuralNGramCompactLabelGenerationHead(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            embedding_dim=6,
            hidden_size=8,
            trainable=True,
        ),
        CRFCompactLabelScorer(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            embedding_dim=6,
            trainable=True,
        ),
        EnergyCompactLabelGenerationHead(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            embedding_dim=6,
            hidden_size=8,
            trainable=True,
        ),
    ]

    for head in heads:
        greedy = constrained_label_greedy_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1)
        beam = constrained_label_beam_search_decode(head, torch.tensor([[9]]), vocab, dfa, max_new_tokens=1, beam_size=2)
        sample = constrained_label_sample_decode(
            head,
            torch.tensor([[9]]),
            vocab,
                    max_new_tokens=1,
            generator=torch.Generator().manual_seed(3),
        )

        assert greedy.labels == [vocab.label_for_token("B")]
        assert beam.labels == [vocab.label_for_token("B")]
        assert sample.labels == [vocab.label_for_token("B")]
        assert greedy.accepted and beam.accepted and sample.accepted


def test_ngram_and_crf_next_logits_change_with_prefix():
    ngram = NeuralNGramCompactLabelGenerationHead(
        label_count=4,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        context_size=2,
        embedding_dim=6,
        hidden_size=8,
        trainable=True,
        random_seed=12,
    )
    crf = CRFCompactLabelScorer(
        label_count=4,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        embedding_dim=6,
        trainable=True,
    )
    energy = EnergyCompactLabelGenerationHead(
        label_count=4,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        context_size=2,
        embedding_dim=6,
        hidden_size=8,
        trainable=True,
        random_seed=12,
    )
    with torch.no_grad():
        crf.transition_logits.zero_()
        crf.transition_logits[1, 2] = 4.0
        crf.transition_logits[2, 1] = 4.0

    assert not torch.allclose(ngram.next_label_logits(torch.tensor([[9, 1]])), ngram.next_label_logits(torch.tensor([[9, 2]])))
    assert not torch.allclose(crf.next_label_logits(torch.tensor([[9, 1]])), crf.next_label_logits(torch.tensor([[9, 2]])))
    assert not torch.allclose(energy.next_label_logits(torch.tensor([[9, 1]])), energy.next_label_logits(torch.tensor([[9, 2]])))


def test_neural_compact_heads_support_pmd_style_forward_and_hybrid_scoring():
    vocab = _vocab()
    dfa = accept_all_dfa(vocab)
    head = GRUCompactLabelGenerationHead(
        label_count=vocab.label_count,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        hidden_size=8,
        embedding_dim=6,
        trainable=True,
    )
    prompt = torch.tensor([[9]])
    labels = torch.tensor([1, 2, 0])

    log_probs = head(None, prompt, labels)
    assert log_probs.shape == (head.pad_size, vocab.label_count)
    assert torch.isfinite(log_probs).all()

    controller = HybridController(vocabulary=vocab, dfa=dfa, scorer_head=head)
    score = controller.score_candidate(
        prompt,
        candidate=GenerationCandidate(token_ids=[1, 2, 0], labels=[1, 2, 0]),
    )
    assert score.accepted
    assert torch.isfinite(torch.tensor(score.head_logprob))


def test_ngram_and_crf_support_pmd_style_forward_and_hybrid_scoring():
    vocab = _vocab()
    dfa = accept_all_dfa(vocab)
    prompt = torch.tensor([[9]])
    labels = torch.tensor([1, 2, 0])
    heads = [
        NeuralNGramCompactLabelGenerationHead(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            embedding_dim=6,
            hidden_size=8,
            trainable=True,
        ),
        CRFCompactLabelScorer(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            embedding_dim=6,
            trainable=True,
        ),
        EnergyCompactLabelGenerationHead(
            label_count=vocab.label_count,
            label_to_token_id=_label_to_token_id(),
            vocab_size=16,
            context_size=2,
            embedding_dim=6,
            hidden_size=8,
            trainable=True,
        ),
    ]

    for head in heads:
        log_probs = head(None, prompt, labels)
        assert log_probs.shape == (head.pad_size, vocab.label_count)
        assert torch.isfinite(log_probs).all()
        batched = head.sequence_log_probs(torch.tensor([[1, 2, 0], [2, 1, 0]]), instruction_tokens=prompt)
        assert batched.shape == (2, head.pad_size, vocab.label_count)
        assert torch.isfinite(batched).all()

        controller = HybridController(vocabulary=vocab, dfa=dfa, scorer_head=head)
        score = controller.score_candidate(
            prompt,
            candidate=GenerationCandidate(token_ids=[1, 2, 0], labels=[1, 2, 0]),
        )
        assert score.accepted
        assert torch.isfinite(torch.tensor(score.head_logprob))


def test_crf_sequence_energy_ranks_configured_transition_path():
    crf = CRFCompactLabelScorer(
        label_count=4,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        embedding_dim=6,
        trainable=True,
    )
    with torch.no_grad():
        crf.start_logits.zero_()
        crf.transition_logits.zero_()
        crf.end_logits.zero_()
        crf.unary_projector.weight.zero_()
        crf.unary_projector.bias.zero_()
        crf.start_logits[1] = 2.0
        crf.transition_logits[1, 2] = 3.0
        crf.transition_logits[2, 0] = 1.0

    preferred = crf.sequence_energy(torch.tensor([1, 2, 0]), instruction_tokens=torch.tensor([[9]]))
    other = crf.sequence_energy(torch.tensor([2, 1, 0]), instruction_tokens=torch.tensor([[9]]))

    assert preferred > other


def test_energy_compact_head_scores_sequences_and_respects_lengths():
    head = EnergyCompactLabelGenerationHead(
        label_count=4,
        label_to_token_id=_label_to_token_id(),
        vocab_size=16,
        context_size=2,
        embedding_dim=6,
        hidden_size=8,
        trainable=True,
        random_seed=7,
    )
    prompt = torch.tensor([[9]])
    labels = torch.tensor([[1, 2, 0], [2, 1, 0]])
    lengths = torch.tensor([3, 2])

    energies = head.sequence_energy(labels, lengths=lengths, instruction_tokens=prompt)
    scores = head.sequence_score(labels, lengths=lengths, instruction_tokens=prompt)
    first_step = head.step_energy(prompt, [1, 2], 0)

    assert energies.shape == (2,)
    assert torch.isfinite(energies).all()
    assert torch.allclose(scores, -energies)
    assert torch.isfinite(first_step)
    assert torch.allclose(
        energies[1],
        head.sequence_energy(labels[1, :2], instruction_tokens=prompt),
        atol=1e-6,
    )


def test_crf_log_partition_and_nll_match_bruteforce_paths():
    crf = CRFCompactLabelScorer(
        label_count=2,
        label_to_token_id=[0, 1],
        vocab_size=8,
        embedding_dim=4,
        trainable=True,
        random_seed=0,
    )
    prompt = torch.tensor([[3]])
    labels = torch.tensor([0, 1, 0])
    with torch.no_grad():
        crf.start_logits.copy_(torch.tensor([0.2, -0.1]))
        crf.transition_logits.copy_(torch.tensor([[0.3, -0.4], [0.1, 0.5]]))
        crf.end_logits.copy_(torch.tensor([0.7, -0.2]))
        crf.prompt_embedding.weight.zero_()
        crf.unary_projector.weight.zero_()
        crf.unary_projector.bias.copy_(torch.tensor([0.4, -0.3]))

    unary = crf._prompt_unary(prompt)[0]
    path_scores = []
    for path in itertools.product(range(2), repeat=3):
        score = crf.start_logits[path[0]] + unary[path[0]]
        score = score + crf.transition_logits[path[0], path[1]] + unary[path[1]]
        score = score + crf.transition_logits[path[1], path[2]] + unary[path[2]]
        score = score + crf.end_logits[path[2]]
        path_scores.append(score)
    expected_log_z = torch.logsumexp(torch.stack(path_scores), dim=0)
    expected_gold = crf.sequence_score(labels, instruction_tokens=prompt)

    assert torch.allclose(crf.log_partition(3, instruction_tokens=prompt), expected_log_z)
    assert torch.allclose(crf.crf_nll(labels, instruction_tokens=prompt), expected_log_z - expected_gold)


def test_crf_marginals_are_normalized_and_batched_lengths_work():
    crf = CRFCompactLabelScorer(
        label_count=3,
        label_to_token_id=[0, 1, 2],
        vocab_size=8,
        embedding_dim=4,
        trainable=True,
        random_seed=4,
    )
    labels = torch.tensor([[0, 1, 2], [2, 1, 0]])
    lengths = torch.tensor([3, 2])

    marginal_log_probs = crf.marginal_log_probs(labels, lengths=lengths, instruction_tokens=torch.tensor([[3]]))
    probs = marginal_log_probs.exp()

    assert marginal_log_probs.shape == (2, 3, 3)
    assert torch.allclose(probs[0].sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(probs[1, :2].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(marginal_log_probs[1, 2], torch.zeros(3), atol=1e-6)


def test_crf_nll_has_gradients_for_all_crf_parameters():
    crf = CRFCompactLabelScorer(
        label_count=3,
        label_to_token_id=[0, 1, 2],
        vocab_size=8,
        embedding_dim=4,
        trainable=True,
    )
    loss = crf.crf_nll(
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        lengths=torch.tensor([3, 2]),
        instruction_tokens=torch.tensor([[3]]),
    )

    loss.backward()

    assert crf.start_logits.grad is not None
    assert crf.transition_logits.grad is not None
    assert crf.end_logits.grad is not None
    assert crf.unary_projector.weight.grad is not None


def test_crf_sequence_log_probs_returns_exact_marginals_not_local_scores():
    crf = CRFCompactLabelScorer(
        label_count=2,
        label_to_token_id=[0, 1],
        vocab_size=8,
        embedding_dim=4,
        trainable=True,
    )
    labels = torch.tensor([0, 1, 0])

    exact = crf.sequence_log_probs(labels, instruction_tokens=torch.tensor([[3]]))
    local = crf.local_sequence_log_probs(labels, instruction_tokens=torch.tensor([[3]]))

    assert exact.shape == local.shape == (crf.pad_size, 2)
    assert torch.allclose(exact[:3].exp().sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(exact[3], torch.zeros(2), atol=1e-6)
    assert torch.isfinite(local).all()
