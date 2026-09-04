"""Sampling-based loss helpers for reward-driven training.

A decoding is a joint assignment of a discrete value to every decision variable
(every instance of every target concept).  We draw a handful of decodings,
score each one with a reward, and build a differentiable loss that pushes the
model's probability mass toward the high-reward decodings.

Two estimators are provided:

* :func:`importance_weighted_loss` -- the log-ratio of the satisfying probability mass to
  the total sampled mass.
* :func:`reinforce_loss` -- the classic REINFORCE policy-gradient estimator with
  an optional baseline for variance reduction.
"""

import torch

__all__ = [
    "sample_assignments",
    "decoding_logprob",
    "importance_weighted_loss",
    "reinforce_loss",
]


def sample_assignments(logits_list, num_samples, weighted=True):
    """Draw ``num_samples`` joint decodings of the decision variables.

    :param logits_list: list of logit tensors, one per concept group, each of
        shape ``[n_instances, n_classes]``.
    :param num_samples: how many decodings to draw.
    :param weighted: when ``True`` (importance_weighted ``WeightedSamplingSolver`` style) sample
        from the model's own categorical distribution; when ``False`` (plain
        ``SamplingSolver`` style) sample uniformly over the classes.
    :return: list parallel to ``logits_list``; each element is an index tensor of
        shape ``[num_samples, n_instances]`` (``long``).
    """
    samples = []
    for logits in logits_list:
        if logits is None or logits.numel() == 0:
            samples.append(None)
            continue
        if weighted:
            dist = torch.distributions.Categorical(logits=logits)
        else:
            probs = torch.full_like(logits, 1.0 / logits.shape[-1])
            dist = torch.distributions.Categorical(probs=probs)
        # sample shape: [num_samples, n_instances]
        samples.append(dist.sample((num_samples,)))
    return samples


def decoding_logprob(sample_idx_list, log_probs_list):
    """Log-probability of each decoding under the model.

    The log-probability of a joint decoding is
    the sum of the per-variable log-probabilities of the chosen classes.

    :param sample_idx_list: list of index tensors ``[num_samples, n_instances]``
        (output of :func:`sample_assignments`).
    :param log_probs_list: list of log-softmax tensors ``[n_instances, n_classes]``.
    :return: tensor of shape ``[num_samples]`` (differentiable w.r.t. the logits).
    """
    total = None
    for sample_idx, log_probs in zip(sample_idx_list, log_probs_list):
        if sample_idx is None or log_probs is None:
            continue
        # log_probs: [n, c] -> expand to [num_samples, n, c]
        num_samples = sample_idx.shape[0]
        lp = log_probs.unsqueeze(0).expand(num_samples, *log_probs.shape)
        chosen = lp.gather(-1, sample_idx.unsqueeze(-1)).squeeze(-1)  # [num_samples, n]
        contribution = chosen.sum(dim=-1)  # [num_samples]
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("decoding_logprob received no usable variable groups.")
    return total


def _as_reward_tensor(rewards, reference):
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=reference.dtype, device=reference.device)
    return rewards.to(dtype=reference.dtype, device=reference.device).reshape(-1)


def importance_weighted_loss(logprob, rewards, proposal_logprob=None, eps=1e-12):
    """
    Self-normalized importance-weighted reward loss.

    When samples were drawn from a proposal distribution ``q``, pass their
    detached proposal log-probabilities.  The loss then uses
    ``log w = log p - stop_gradient(log q)`` and is a valid proposal-corrected
    estimator.  In particular, on-policy samples have equal forward weights
    while retaining the target-policy gradient.

    Omitting ``proposal_logprob`` preserves the historical sampled-mass
    objective for generic callers that do not record a proposal distribution.

    :param logprob: ``[num_samples]`` log-probabilities from :func:`decoding_logprob`.
    :param rewards: ``[num_samples]`` non-negative rewards (tensor or list).
    :param proposal_logprob: optional ``[num_samples]`` log-probabilities under
        the distribution that generated the samples.
    """
    rewards = _as_reward_tensor(rewards, logprob).clamp_min(0)
    if proposal_logprob is None:
        log_weight = logprob
    else:
        proposal_logprob = _as_reward_tensor(proposal_logprob, logprob).detach()
        if proposal_logprob.shape != logprob.reshape(-1).shape:
            raise ValueError("proposal_logprob must have the same shape as logprob")
        log_weight = logprob.reshape(-1) - proposal_logprob
    log_reward = torch.log(rewards + eps)
    numerator = torch.logsumexp(log_weight + log_reward, dim=0)
    denominator = torch.logsumexp(log_weight, dim=0)
    return -(numerator - denominator)


def reinforce_loss(logprob, rewards, baseline="mean"):
    """Classic REINFORCE estimator: ``-mean((reward - baseline) * logprob)``.

    :param logprob: ``[num_samples]`` log-probabilities from :func:`decoding_logprob`.
    :param rewards: ``[num_samples]`` rewards (tensor or list).
    :param baseline: ``'mean'`` to subtract the batch-mean reward (variance
        reduction), ``None`` for no baseline, or a scalar value.
    """
    rewards = _as_reward_tensor(rewards, logprob)
    if baseline == "mean":
        b = rewards.mean()
    elif baseline is None:
        b = 0.0
    else:
        b = baseline
    advantage = (rewards - b).detach()
    return -(advantage * logprob).mean()
