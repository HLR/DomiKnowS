from __future__ import annotations

import torch

from domiknows.generation.dfa._constraints import (
    HMMGenerationHead,
    SpectralWFAGenerationHead,
    allowed_mass_loss,
    constraints_to_dfa,
    hmm_sequence_nll,
    no_token_after_eos,
    required_token,
    wfa_sequence_energy_loss,
)
from domiknows.generation.learners import DiscreteHMM, WeightedFiniteAutomaton


def test_hmm_head_returns_valid_log_probs():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=False)

    log_probs = head(None, torch.tensor([[9]]), torch.tensor([1, 2, 0]))

    assert log_probs.shape == (3, 4)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(3), atol=1e-5)


def test_hmm_head_exposes_batched_production_core():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    batched = head.sequence_log_probs(torch.tensor([[1, 2, 0], [2, 0, 0]]), lengths=torch.tensor([3, 2]))
    core = head.production_hmm()

    assert batched.shape == (2, 3, 4)
    assert torch.isfinite(batched).all()
    assert core.symbols == (0, 1, 2, 3)
    assert torch.allclose(core.transition, head.transition_probs)


def test_hmm_head_can_wrap_existing_discrete_hmm():
    hmm = DiscreteHMM(
        transition=((0.7, 0.3), (0.2, 0.8)),
        emission=((0.6, 0.3, 0.1), (0.1, 0.4, 0.5)),
        initial=(0.8, 0.2),
        symbols=(0, 1, 2),
    )
    head = HMMGenerationHead(hmm, pad_size=2, trainable=False)

    assert head.label_count == 3
    assert head.state_count == 2
    assert head.next_label_logits(torch.tensor([0, 1])).shape == (3,)


def test_wfa_head_returns_finite_log_probs_for_signed_scores():
    wfa = WeightedFiniteAutomaton(
        initial=(1.0, -0.4),
        transitions={
            0: ((0.2, -0.1), (0.3, 0.4)),
            1: ((-0.5, 0.2), (0.1, -0.3)),
            2: ((0.4, 0.1), (-0.2, 0.5)),
        },
        final=(0.7, -0.6),
        symbols=(0, 1, 2),
    )
    head = SpectralWFAGenerationHead(wfa, pad_size=3, trainable=False)

    logits = head.next_label_logits(torch.tensor([1, 2]))
    log_probs = head(None, torch.tensor([[9]]), torch.tensor([1, 2, 0]))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    assert log_probs.shape == (3, 3)
    assert torch.isfinite(log_probs).all()


def test_wfa_head_exposes_batched_production_core():
    head = SpectralWFAGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    batched = head.sequence_log_probs(torch.tensor([[1, 2, 0], [2, 0, 0]]), lengths=torch.tensor([3, 2]))
    core = head.production_wfa()

    assert batched.shape == (2, 3, 4)
    assert torch.isfinite(batched).all()
    assert core.symbols == (0, 1, 2, 3)
    assert torch.allclose(core.transition_tensor, head.transitions)


def test_frozen_and_trainable_head_parameters():
    frozen = HMMGenerationHead(label_count=3, state_count=2, trainable=False)
    trainable = HMMGenerationHead(label_count=3, state_count=2, trainable=True)
    wfa = SpectralWFAGenerationHead(label_count=3, state_count=2, trainable=True)

    assert frozen.trainable_parameter_names() == []
    assert trainable.trainable_parameter_names() == [
        "initial_logits",
        "transition_logits",
        "emission_logits",
    ]
    assert wfa.trainable_parameter_names() == ["initial", "transitions", "final"]


def test_hmm_auxiliary_loss_is_finite_and_differentiable():
    head = HMMGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    loss = hmm_sequence_nll(head, torch.tensor([1, 2, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_wfa_auxiliary_loss_is_finite_and_differentiable():
    head = SpectralWFAGenerationHead(label_count=4, state_count=2, pad_size=3, trainable=True)

    loss = wfa_sequence_energy_loss(head, torch.tensor([1, 2, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_allowed_mass_loss_is_finite_and_differentiable():
    vocab = TokenVocabulary(["<eos>", " A"], eos_token="<eos>")
    dfa = constraints_to_dfa([no_token_after_eos(), required_token(" A")], vocab)
    logits = torch.tensor(
        [
            [0.1, 3.0, -1.0],
            [2.0, 0.2, -1.0],
        ],
        requires_grad=True,
    )
    probs = torch.softmax(logits, dim=-1)

    loss = allowed_mass_loss(probs, dfa)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
