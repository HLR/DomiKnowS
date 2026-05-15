from __future__ import annotations

import torch

from domiknows.generation import (
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    hmm_dp_factor_consistency_loss,
    hmm_factor_sequence_nll,
    hmm_forward_backward_factors,
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
    assert bundle.forward_state is None
    assert bundle.backward_state is None
    assert bundle.transition_pair is None
    assert bundle.context.state_index("LOC") == 2


def test_hmm_factor_graph_builds_with_dp_factor_concepts():
    encoder = HMMFactorGraphEncoder(
        vocab=["<eos>", " A"],
        eos_token="<eos>",
        state_names=["PER", "O", "LOC"],
        include_dp_factors=True,
    )

    _graph, bundle = encoder.build_graph()

    assert bundle.include_dp_factors
    assert bundle.forward_state is not None
    assert bundle.backward_state is not None
    assert bundle.transition_pair is not None
    assert bundle.transition_pair_names == (
        "PER->PER",
        "PER->O",
        "PER->LOC",
        "O->PER",
        "O->O",
        "O->LOC",
        "LOC->PER",
        "LOC->O",
        "LOC->LOC",
    )
    assert bundle.context.transition_pair_index("PER", "LOC") == 2


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


def test_hmm_factor_head_returns_normalized_dp_factors():
    head = HMMFactorGraphHead(label_count=5, state_names=["PER", "O", "LOC"], pad_size=4)

    factors = hmm_forward_backward_factors(head, torch.tensor([1, 2, 3, 0]))
    forward = head.forward_log_probs(torch.tensor([1, 2, 3, 0]))
    backward = head.backward_log_probs(torch.tensor([1, 2, 3, 0]))
    transition_pair = head.transition_pair_log_probs(torch.tensor([1, 2, 3, 0]))

    assert factors["alpha"].shape == (4, 3)
    assert factors["beta"].shape == (4, 3)
    assert factors["gamma"].shape == (4, 3)
    assert factors["xi"].shape == (3, 3, 3)
    assert factors["scales"].shape == (4,)
    assert forward.shape == (4, 3)
    assert backward.shape == (4, 3)
    assert transition_pair.shape == (3, 9)
    assert torch.allclose(factors["alpha"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(factors["beta"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(factors["gamma"], head.latent_marginals(torch.tensor([1, 2, 3, 0])), atol=1e-6)
    assert torch.allclose(factors["xi"].sum(dim=(1, 2)), torch.ones(3), atol=1e-5)


def test_hmm_dp_factor_consistency_loss_is_finite_and_differentiable():
    head = HMMFactorGraphHead(label_count=5, state_count=3, pad_size=4, trainable=True)

    loss = hmm_dp_factor_consistency_loss(head, torch.tensor([1, 2, 3, 0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in head.parameters())


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
    forward_module = head.forward_module()
    backward_module = head.backward_module()
    transition_pair_module = head.transition_pair_module()

    generated = generated_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    latent = latent_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    forward = forward_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    backward = backward_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    transition_pair = transition_pair_module(None, torch.tensor([[9]]), torch.tensor([1, 2, 3, 0]))
    loss = (
        generated[:, 1].mean()
        + latent[:, 0].mean()
        + forward[:, 0].mean()
        + backward[:, 0].mean()
        + transition_pair[:, 0].mean()
    )
    loss.backward()

    assert generated.shape == (4, 5)
    assert latent.shape == (4, 3)
    assert forward.shape == (4, 3)
    assert backward.shape == (4, 3)
    assert transition_pair.shape == (3, 9)
    assert head.initial_logits.grad is not None
