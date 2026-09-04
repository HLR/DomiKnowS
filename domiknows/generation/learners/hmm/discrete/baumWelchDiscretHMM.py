from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .discreteHMM import DiscreteHMM


@dataclass(frozen=True)
class HMMParameters:
    """Raw (unnormalised) HMM parameter arrays used to seed Baum-Welch training."""

    transition: tuple[tuple[float, ...], ...]
    emission: tuple[tuple[float, ...], ...]
    initial: tuple[float, ...]


@dataclass(frozen=True)
class BaumWelchResult:
    """Result bundle for discrete-HMM Baum-Welch training."""

    model: DiscreteHMM
    log_likelihoods: tuple[float, ...]
    iterations: int
    converged: bool


def run_baum_welch(
    cls: type[DiscreteHMM],
    sequences: Sequence[Sequence[object]],
    symbols: Sequence[object],
    state_count: int,
    *,
    max_iter: int,
    tol: float,
    smoothing: float,
    init: DiscreteHMM | HMMParameters | None,
    random_seed: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> BaumWelchResult:
    """Run Baum-Welch EM and return fitted model plus convergence metadata."""

    encoded, symbols = _validate_and_encode_sequences(sequences, symbols, state_count)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    obs = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(encoded):
        obs[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    if init is None:
        gen_device = torch.device(device) if device is not None else torch.device("cpu")
        gen = torch.Generator(device=gen_device).manual_seed(int(random_seed))
        initial = _normalize_tensor(torch.rand(state_count, generator=gen, dtype=dtype, device=device) + 0.1, dim=0)
        transition = _normalize_tensor(torch.rand((state_count, state_count), generator=gen, dtype=dtype, device=device) + 0.1, dim=-1)
        emission = _normalize_tensor(torch.rand((state_count, len(symbols)), generator=gen, dtype=dtype, device=device) + 0.1, dim=-1)
        model = cls(transition, emission, initial, symbols, normalize=False)
    elif isinstance(init, cls):
        if init.symbols != tuple(symbols) or init.state_count != state_count:
            raise ValueError("init must match symbols and state_count")
        model = init.to(device=device, dtype=dtype)
    else:
        params = _coerce_init(init, symbols, state_count, cls)
        model = cls(params.transition, params.emission, params.initial, symbols, device=device, dtype=dtype)

    log_likelihoods: list[float] = []
    converged = False
    for iteration in range(max_iter):
        factors = model.forward_backward(obs, lengths)
        mask = factors.mask
        init_counts = factors.gamma[:, 0, :].sum(dim=0)
        trans_counts = factors.xi.sum(dim=(0, 1))
        emit_counts = torch.zeros((state_count, len(symbols)), dtype=model.dtype, device=model.device)
        for symbol_idx in range(len(symbols)):
            symbol_mask = (obs == symbol_idx) & mask
            emit_counts[:, symbol_idx] = (factors.gamma * symbol_mask.unsqueeze(-1)).sum(dim=(0, 1))

        initial = _normalize_tensor(init_counts + smoothing, dim=0)
        transition = _normalize_tensor(trans_counts + smoothing, dim=-1)
        emission = _normalize_tensor(emit_counts + smoothing, dim=-1)
        total_ll = float(factors.log_likelihood.sum().item())
        log_likelihoods.append(total_ll)
        model = cls(transition, emission, initial, symbols, state_names=model.state_names, normalize=False)
        if len(log_likelihoods) > 1 and log_likelihoods[-1] - log_likelihoods[-2] < tol:
            converged = True
            break

    return BaumWelchResult(
        model=model,
        log_likelihoods=tuple(log_likelihoods),
        iterations=iteration + 1,
        converged=converged,
    )


def _validate_and_encode_sequences(
    sequences: Sequence[Sequence[object]],
    symbols: Sequence[object],
    state_count: int,
) -> tuple[list[list[int]], tuple[object, ...]]:
    if state_count < 1:
        raise ValueError("state_count must be at least 1")
    symbols = tuple(symbols)
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    if not sequences:
        raise ValueError("sequences must not be empty")

    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    encoded = []
    for seq_idx, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError("empty sequences are not supported in Baum-Welch v1")
        encoded_sequence = []
        for symbol in sequence:
            if symbol not in symbol_index:
                raise ValueError(f"unknown symbol {symbol!r} in sequence {seq_idx}")
            encoded_sequence.append(symbol_index[symbol])
        encoded.append(encoded_sequence)
    return encoded, symbols


def _coerce_init(
    init: DiscreteHMM | HMMParameters,
    symbols: tuple[object, ...],
    state_count: int,
    discrete_hmm_cls: type[DiscreteHMM],
) -> HMMParameters:
    if isinstance(init, discrete_hmm_cls):
        if init.symbols != symbols:
            raise ValueError("init symbols must match symbols")
        params = HMMParameters(
            tuple(tuple(float(v) for v in row.tolist()) for row in init.transition_probs.detach().cpu()),
            tuple(tuple(float(v) for v in row.tolist()) for row in init.emission_probs.detach().cpu()),
            tuple(float(v) for v in init.initial_probs.detach().cpu().tolist()),
        )
    elif isinstance(init, HMMParameters):
        params = init
    else:
        raise TypeError("init must be DiscreteHMM, HMMParameters, or None")

    if len(params.initial) != state_count:
        raise ValueError("init initial length must match state_count")
    if len(params.transition) != state_count or any(len(row) != state_count for row in params.transition):
        raise ValueError("init transition shape must be state_count x state_count")
    if len(params.emission) != state_count or any(len(row) != len(symbols) for row in params.emission):
        raise ValueError("init emission shape must be state_count x len(symbols)")
    initial = tuple(_normalize_row(params.initial, 0.0))
    transition = tuple(tuple(_normalize_row(row, 0.0)) for row in params.transition)
    emission = tuple(tuple(_normalize_row(row, 0.0)) for row in params.emission)
    return HMMParameters(transition=transition, emission=emission, initial=initial)


def _normalize_row(row: Sequence[float], smoothing: float) -> list[float]:
    if not row:
        raise ValueError("cannot normalize an empty row")
    smoothed = [max(float(value), 0.0) + smoothing for value in row]
    total = sum(smoothed)
    if total <= 0.0:
        return [1.0 / len(row) for _ in row]
    return [value / total for value in smoothed]


def _normalize_tensor(values: torch.Tensor, dim: int) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("probability tensors must contain only finite values")
    values = values.clamp_min(0)
    total = values.sum(dim=dim, keepdim=True)
    if torch.any(total <= 0):
        size = values.shape[dim]
        return torch.full_like(values, 1.0 / float(size))
    return values / total


def baum_welch_train(
    sequences: Sequence[Sequence[str]],
    symbols: Sequence[str],
    state_count: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    smoothing: float = 1e-9,
    init: "DiscreteHMM | HMMParameters | None" = None,
    random_seed: int = 0,
) -> BaumWelchResult:
    """Train a discrete HMM with Baum-Welch expectation maximization."""

    from .discreteHMM import DiscreteHMM

    return DiscreteHMM.baum_welch(
        sequences,
        symbols,
        state_count,
        max_iter=max_iter,
        tol=tol,
        smoothing=smoothing,
        init=init,
        random_seed=random_seed,
    )


__all__ = ["HMMParameters", "BaumWelchResult", "run_baum_welch", "baum_welch_train"]
