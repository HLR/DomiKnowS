"""Unit tests for the reward-sampling math in domiknows.reinforcement.sampling."""
import torch

from domiknows.reinforcement.sampling import (
    sample_assignments, decoding_logprob, importance_weighted_loss, reinforce_loss,
)


def test_sample_and_logprob_shapes():
    torch.manual_seed(0)
    logits = torch.randn(5, 2, requires_grad=True)
    samples = sample_assignments([logits], num_samples=6, weighted=True)
    assert samples[0].shape == (6, 5)
    lp = decoding_logprob(samples, [torch.log_softmax(logits, -1)])
    assert lp.shape == (6,)
    assert lp.requires_grad


def test_importance_weighted_matches_reference_for_binary_reward():
    """With binary reward, importance_weighted_loss"""
    torch.manual_seed(0)
    logits = torch.randn(5, 2, requires_grad=True)
    samples = sample_assignments([logits], num_samples=6, weighted=True)
    lp = decoding_logprob(samples, [torch.log_softmax(logits, -1)])

    rewards = torch.tensor([1., 0., 1., 0., 1., 0.])
    loss = importance_weighted_loss(lp, rewards)

    ind = rewards.bool()
    sat = lp.clone()
    sat[~ind] = -float("inf")
    ref = -(torch.logsumexp(sat, 0) - torch.logsumexp(lp, 0))
    assert torch.allclose(loss, ref, atol=1e-4)

    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_importance_weighted_corrects_for_on_policy_proposal():
    """On-policy samples must not receive a second length/probability bias."""
    logprob = torch.tensor([-8.0, -1.0], requires_grad=True)
    proposal_logprob = logprob.detach().clone()
    rewards = torch.tensor([1.0, 0.25])

    loss = importance_weighted_loss(
        logprob, rewards, proposal_logprob=proposal_logprob
    )
    log_weight = logprob - proposal_logprob
    expected = -(
        torch.logsumexp(log_weight + torch.log(rewards), dim=0)
        - torch.logsumexp(log_weight, dim=0)
    )
    assert torch.allclose(loss, expected)

    loss.backward()
    # Gradient descent increases the high-reward sample and decreases the
    # low-reward sample, regardless of their very different proposal masses.
    assert logprob.grad[0] < 0
    assert logprob.grad[1] > 0


def test_reinforce_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(3, 3, requires_grad=True)
    samples = sample_assignments([logits], num_samples=4)
    lp = decoding_logprob(samples, [torch.log_softmax(logits, -1)])
    loss = reinforce_loss(lp, [0., 1., 0.5, 1.])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_multi_group_logprob_sums_groups():
    torch.manual_seed(1)
    a = torch.randn(4, 2)
    b = torch.randn(3, 5)
    samples = sample_assignments([a, b], num_samples=7)
    lp = decoding_logprob(samples, [torch.log_softmax(a, -1), torch.log_softmax(b, -1)])
    assert lp.shape == (7,)
