from __future__ import annotations

import torch

from domiknows.generation import (
    SpectralWFAFactorGraphEncoder,
    SpectralWFAFactorGraphHead,
    WeightedFiniteAutomaton,
    wfa_factor_consistency_loss,
    wfa_factor_sequence_energy_loss,
)


def test_wfa_factor_graph_builds_with_named_states():
    encoder = SpectralWFAFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_names=["A", "B", "C"],
    )

    graph, bundle = encoder.build_graph()

    assert graph.name == "wfa_factor_generation"
    assert bundle.state_names == ("A", "B", "C")
    assert bundle.wfa_state is not None
    assert bundle.is_next_rel is not None
    assert bundle.wfa_transition_pair is None
    assert bundle.context.state_index("C") == 2


def test_wfa_factor_graph_builds_with_transition_pair_concepts():
    encoder = SpectralWFAFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_names=["A", "B", "C"],
        include_transition_pairs=True,
    )

    _graph, bundle = encoder.build_graph()

    assert bundle.include_transition_pairs
    assert bundle.wfa_transition_pair is not None
    assert bundle.transition_pair_names == (
        "A->A",
        "A->B",
        "A->C",
        "B->A",
        "B->B",
        "B->C",
        "C->A",
        "C->B",
        "C->C",
    )
    assert bundle.context.transition_pair_index("A", "C") == 2


def test_wfa_factor_graph_builds_with_anonymous_state_count():
    encoder = SpectralWFAFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_count=3,
    )

    _graph, bundle = encoder.build_graph()

    assert bundle.state_names == ("S0", "S1", "S2")
    assert bundle.context.state_index("S1") == 1


def test_wfa_factor_head_returns_normalized_projections():
    head = SpectralWFAFactorGraphHead(label_count=5, state_names=["A", "B", "C"], pad_size=4)

    generated = head.generated_log_probs(torch.tensor([1, 2, 3, 0]))
    states = head.state_log_probs(torch.tensor([1, 2, 3, 0]))
    transition_pair = head.transition_pair_log_probs(torch.tensor([1, 2, 3, 0]))

    assert generated.shape == (4, 5)
    assert states.shape == (4, 3)
    assert transition_pair.shape == (3, 9)
    assert torch.isfinite(generated).all()
    assert torch.isfinite(states).all()
    assert torch.isfinite(transition_pair).all()
    assert torch.allclose(generated.exp().sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(states.exp().sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(transition_pair.exp().sum(dim=-1), torch.ones(3), atol=1e-5)


def test_wfa_factor_head_preserves_signed_raw_scores():
    wfa = WeightedFiniteAutomaton(
        initial=(1.0, -0.5),
        transitions={
            0: ((1.0, -0.2), (0.3, 0.7)),
            1: ((-0.4, 0.5), (0.1, -0.8)),
        },
        final=(0.7, -0.6),
        symbols=(0, 1),
    )
    head = SpectralWFAFactorGraphHead(wfa, pad_size=2, trainable=False)

    raw_logits = head.next_label_logits(torch.tensor([0]))
    log_probs = head.generated_log_probs(torch.tensor([0, 1]))

    assert raw_logits.shape == (2,)
    assert torch.any(raw_logits < 0)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2), atol=1e-5)


def test_wfa_factor_losses_are_finite_and_differentiable():
    head = SpectralWFAFactorGraphHead(label_count=5, state_count=3, pad_size=4, trainable=True)

    energy = wfa_factor_sequence_energy_loss(head, torch.tensor([1, 2, 3, 0]))
    consistency = wfa_factor_consistency_loss(head, torch.tensor([1, 2, 3, 0]))
    loss = energy + consistency
    loss.backward()

    assert torch.isfinite(energy)
    assert torch.isfinite(consistency)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_wfa_factor_projection_modules_share_parameters():
    head = SpectralWFAFactorGraphHead(label_count=5, state_count=3, pad_size=4, trainable=True)

    generated_model = head.generated_module()
    state_model = head.state_module()
    transition_pair_model = head.transition_pair_module()

    generated = generated_model(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    states = state_model(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    transition_pair = transition_pair_model(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    loss = generated[:, 1].mean() + states[:, 0].mean() + transition_pair[:, 0].mean()
    loss.backward()

    assert generated.shape == (4, 5)
    assert states.shape == (4, 3)
    assert transition_pair.shape == (3, 9)
    assert head.initial.grad is not None
