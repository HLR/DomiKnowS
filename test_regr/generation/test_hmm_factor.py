from __future__ import annotations

import torch

from domiknows.generation import (
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    hmm_factor_sequence_nll,
)


def test_hmm_factor_graph_builds_with_named_states():
    encoder = HMMFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_names=["PER", "O", "LOC"],
    )

    graph, bundle = encoder.build_graph()

    assert graph.name == "hmm_factor_generation"
    assert bundle.state_names == ("PER", "O", "LOC")
    assert bundle.latent_state is not None
    assert bundle.is_next_rel is not None
    assert bundle.context.state_index("LOC") == 2


def test_hmm_factor_graph_builds_with_anonymous_state_count():
    encoder = HMMFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_count=3,
    )

    _graph, bundle = encoder.build_graph()

    assert bundle.state_names == ("S0", "S1", "S2")
    assert bundle.context.state_index("S1") == 1


def test_hmm_factor_head_returns_normalized_marginals():
    head = HMMFactorGraphHead(label_count=5, state_names=["PER", "O", "LOC"], pad_size=4)

    generated = head.generated_log_probs(torch.tensor([1, 2, 3, 0]))
    latent = head.latent_log_probs(torch.tensor([1, 2, 3, 0]))

    assert generated.shape == (4, 5)
    assert latent.shape == (4, 3)
    assert torch.isfinite(generated).all()
    assert torch.isfinite(latent).all()
    assert torch.allclose(generated.exp().sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(latent.exp().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_hmm_factor_sequence_nll_is_finite_and_differentiable():
    head = HMMFactorGraphHead(
        label_count=5,
        state_count=3,
        pad_size=4,
        trainable=True,
    )

    loss = hmm_factor_sequence_nll(head, torch.tensor([1, 2, 3, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_hmm_factor_projection_modules_share_parameters():
    head = HMMFactorGraphHead(label_count=5, state_count=3, pad_size=4, trainable=True)

    generated_module = head.generated_module()
    latent_module = head.latent_module()

    generated = generated_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    latent = latent_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    loss = generated[:, 1].mean() + latent[:, 0].mean()
    loss.backward()

    assert generated.shape == (4, 5)
    assert latent.shape == (4, 3)
    assert head.initial_logits.grad is not None
