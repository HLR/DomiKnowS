"""Per-instance test-time adaptation of the HMM emissions (after CLAMP's ``ttt/per_instance.py``).

Before decoding one episode, the frozen model is sampled ``K`` times on that episode's prompt, the
samples are rewritten to the canonical plan text the HMM was fitted on, and one Baum-Welch step
re-estimates the emission matrix ``B`` from them. The new matrix is mixed with the old one,
``B' = (1 - lam) B + lam B_EM``, so a handful of samples nudges the HMM towards this prompt without
overwriting what it learned from the whole benchmark. The initial distribution and the transitions
are left alone. No reference plan is read.
"""
from __future__ import annotations

import zlib
from typing import Any, Sequence

import torch


def expected_counts(initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor,
                    sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected number of times each hidden state emits each token that occurs in ``sequences``.

    Returns ``(tokens, counts)``: the distinct token ids and a ``(H, len(tokens))`` tensor, from the
    scaled forward-backward algorithm on all sequences at once. Sequences the HMM gives zero
    probability are skipped.
    """
    device, dtype = emission.device, torch.float64
    initial, transition, emission = initial.to(dtype), transition.to(dtype), emission.to(dtype)
    lengths = [len(sequence) for sequence in sequences]
    count, steps, size = len(sequences), max(lengths), emission.shape[0]
    tokens = torch.zeros(count, steps, dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(list(sequence), device=device)
    valid = torch.arange(steps, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]  # (K, T)
    likelihood = emission[:, tokens.reshape(-1)].T.reshape(count, steps, size)  # B[:, x_t] per position
    likelihood = torch.where(valid[:, :, None], likelihood, torch.ones_like(likelihood))
    forward = torch.zeros(count, steps, size, dtype=dtype, device=device)
    scale = torch.ones(count, steps, dtype=dtype, device=device)
    first = initial[None, :] * likelihood[:, 0]
    scale[:, 0] = first.sum(-1).clamp_min(1e-300)
    forward[:, 0] = first / scale[:, 0:1]
    for step in range(1, steps):
        reached = (forward[:, step - 1] @ transition) * likelihood[:, step]
        norm = reached.sum(-1).clamp_min(1e-300)
        live = valid[:, step]
        forward[:, step] = torch.where(live[:, None], reached / norm[:, None], forward[:, step - 1])
        scale[:, step] = torch.where(live, norm, torch.ones_like(norm))
    backward = torch.ones(count, steps, size, dtype=dtype, device=device)
    for step in range(steps - 2, -1, -1):
        following = likelihood[:, step + 1] * backward[:, step + 1] / scale[:, step + 1:step + 2]
        following = torch.where(valid[:, step + 1][:, None], following, torch.ones_like(following))
        backward[:, step] = following @ transition.T
    posterior = forward * backward
    posterior = posterior / posterior.sum(-1, keepdim=True).clamp_min(1e-300)
    posterior = posterior * valid[:, :, None]
    possible = (torch.log(scale).sum(-1) > -600)  # drop sequences whose probability underflowed to ~0
    posterior = posterior * possible[:, None, None]
    unique, inverse = torch.unique(tokens[valid], return_inverse=True)
    flat = posterior[valid]  # (N, H)
    counts = torch.zeros(len(unique), size, dtype=dtype, device=device).index_add_(0, inverse, flat)
    return unique, counts.T


def adapt_emission(initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor,
                   sequences: Sequence[Sequence[int]], lam: float) -> torch.Tensor:
    """``(1 - lam) B + lam B_EM`` for the states that emitted something, rows still summing to one."""
    if not sequences or lam <= 0:
        return emission
    tokens, counts = expected_counts(initial, transition, emission, sequences)
    totals = counts.sum(-1)  # (H,)
    seen = totals > 1e-12
    estimate = counts / totals.clamp_min(1e-300)[:, None]
    mix = torch.where(seen, torch.full_like(totals, float(lam)), torch.zeros_like(totals))
    adapted = emission.to(counts.dtype) * (1 - mix)[:, None]
    adapted[:, tokens] += mix[:, None] * estimate
    return adapted


def episode_seed(items: Sequence[Sequence[Any]], seed: int) -> int:
    """A reproducible sampling seed from the episode's own text and the run seed."""
    text = "\n".join(str(value) for kind, value in items if kind == "text")
    return (zlib.crc32(text.encode("utf-8")) ^ int(seed)) & 0x7FFFFFFF


@torch.no_grad()
def sample_plans(planner: Any, inputs: Any, tokenizer: Any, *, samples: int, max_new_tokens: int, seed: int,
                 temperature: float = 0.8, top_p: float = 0.95) -> list[list[int]]:
    """Canonical plan tokens (ending in EOS) of ``samples`` unconstrained replies to one prompt."""
    try:
        from .official_hmm import plan_tokens
    except ImportError:  # pragma: no cover - flat-module execution
        from official_hmm import plan_tokens

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    length = inputs.input_ids.shape[1]
    generated = planner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                       top_p=top_p, num_return_sequences=samples, use_cache=True)[:, length:]
    result = []
    for row in generated.tolist():
        tokens, _ = plan_tokens(tokenizer.decode(row, skip_special_tokens=True), tokenizer)
        if tokens is not None:
            result.append(tokens)
    return result
