"""Hidden Markov Model (HMM) training and inference utilities.

Provides:
- ``DiscreteHMM``: a Torch-backed discrete HMM that can score batched
  observation sequences, expose forward/backward factors, run Viterbi, sample,
  serialize, and be extracted into a deterministic finite automaton (DFA).
- ``ProbabilisticAutomaton``: a legacy compatibility wrapper for examples.
- ``HMMParameters`` / ``BaumWelchResult``: lightweight data containers.
- ``baum_welch_train``: batched Torch Baum-Welch EM training with scaled
  forward-backward to avoid floating-point underflow.
- ``compare_hmm_dfa``: evaluation helper that compares HMM acceptance against
  a reference DFA over a corpus of sequences.
- ``all_sequences``: enumerates all symbol sequences up to a given length.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import torch

from .dfa import DFA
from ..latent_potentials import LatentTransitionPotential, apply_hmm_transition_potential


@dataclass(frozen=True)
class HMMForwardBackward:
    """Batched HMM dynamic-programming factors."""

    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    xi: torch.Tensor
    scales: torch.Tensor
    log_likelihood: torch.Tensor
    mask: torch.Tensor


class DiscreteHMM:
    """Torch-backed discrete HMM/PFA for production scoring and training.

    Parameters are stored as row-stochastic tensors and all sequence APIs accept
    batched integer observations shaped ``[batch, seq]`` plus optional lengths.
    Small string-sequence helpers remain available for inspection and examples.
    """

    def __init__(
        self,
        transition: torch.Tensor | Sequence[Sequence[float]],
        emission: torch.Tensor | Sequence[Sequence[float]],
        initial: torch.Tensor | Sequence[float],
        symbols: Sequence[object],
        *,
        state_names: Sequence[str] | None = None,
        normalize: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        symbols = tuple(symbols)
        _validate_symbol_tuple(symbols)
        dtype = dtype or torch.float32
        transition_t = torch.as_tensor(transition, dtype=dtype, device=device)
        emission_t = torch.as_tensor(emission, dtype=dtype, device=device)
        initial_t = torch.as_tensor(initial, dtype=dtype, device=device)
        if normalize:
            initial_t = _normalize_tensor(initial_t, dim=0)
            transition_t = _normalize_tensor(transition_t, dim=-1)
            emission_t = _normalize_tensor(emission_t, dim=-1)
        _validate_hmm_tensors(initial_t, transition_t, emission_t, len(symbols))
        if state_names is None:
            state_names = tuple(f"S{i}" for i in range(initial_t.numel()))
        else:
            state_names = tuple(str(name) for name in state_names)
            if len(state_names) != initial_t.numel():
                raise ValueError("state_names length must match state count")
            if len(set(state_names)) != len(state_names):
                raise ValueError("state_names must be unique")
        self.initial_probs = initial_t
        self.transition_probs = transition_t
        self.emission_probs = emission_t
        self.symbols = symbols
        self.state_names = tuple(state_names)
        self._symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    @property
    def state_count(self) -> int:
        return int(self.initial_probs.numel())

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def device(self) -> torch.device:
        return self.initial_probs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.initial_probs.dtype

    @property
    def initial(self) -> torch.Tensor:
        return self.initial_probs

    @property
    def transition(self) -> torch.Tensor:
        return self.transition_probs

    @property
    def emission(self) -> torch.Tensor:
        return self.emission_probs

    def transition_with_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> torch.Tensor:
        """Return transitions after optional latent-potential reweighting."""
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def with_transition_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]],
    ) -> "DiscreteHMM":
        """Return a new HMM with transition dynamics reweighted by *transition_potential*."""
        return DiscreteHMM(
            self.transition_with_potential(transition_potential),
            self.emission_probs,
            self.initial_probs,
            self.symbols,
            state_names=self.state_names,
            normalize=False,
        )

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "DiscreteHMM":
        return DiscreteHMM(
            self.transition_probs.to(device=device, dtype=dtype or self.dtype),
            self.emission_probs.to(device=device, dtype=dtype or self.dtype),
            self.initial_probs.to(device=device, dtype=dtype or self.dtype),
            self.symbols,
            state_names=self.state_names,
            normalize=False,
        )

    def encode(self, sequences: Sequence[Sequence[object]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("sequences must not be empty")
        encoded: list[list[int]] = []
        lengths: list[int] = []
        for seq_idx, sequence in enumerate(sequences):
            if not sequence:
                raise ValueError("empty sequences are not supported")
            row = []
            for symbol in sequence:
                if symbol not in self._symbol_index:
                    raise ValueError(f"unknown symbol {symbol!r} in sequence {seq_idx}")
                row.append(self._symbol_index[symbol])
            encoded.append(row)
            lengths.append(len(row))
        max_len = max(lengths)
        padded = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)
        for idx, row in enumerate(encoded):
            padded[idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
        return padded, torch.tensor(lengths, dtype=torch.long, device=self.device)

    def _prepare_observations(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.as_tensor(observations, dtype=torch.long, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.dim() != 2:
            raise ValueError("observations must have shape [seq] or [batch, seq]")
        if obs.shape[1] == 0:
            raise ValueError("observations must contain at least one timestep")
        if torch.any((obs < 0) | (obs >= self.symbol_count)):
            raise ValueError("observations contain labels outside the symbol vocabulary")
        if lengths is None:
            lengths_t = torch.full((obs.shape[0],), obs.shape[1], dtype=torch.long, device=self.device)
        else:
            lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=self.device).reshape(-1)
        if lengths_t.numel() != obs.shape[0]:
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(lengths_t < 1) or torch.any(lengths_t > obs.shape[1]):
            raise ValueError("lengths must be in [1, seq_len]")
        mask = torch.arange(obs.shape[1], device=self.device).unsqueeze(0) < lengths_t.unsqueeze(1)
        return obs, lengths_t, mask

    def log_prob(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> torch.Tensor:
        factors = self.forward_backward(observations, lengths, transition_potential=transition_potential)
        return factors.log_likelihood

    def sequence_probability(
        self,
        sequence: Sequence[object],
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> float:
        if not sequence:
            return 1.0
        obs, lengths = self.encode([sequence])
        return float(torch.exp(self.log_prob(obs, lengths, transition_potential=transition_potential))[0].item())

    def forward_backward(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> HMMForwardBackward:
        obs, lengths_t, mask = self._prepare_observations(observations, lengths)
        batch, seq_len = obs.shape
        state_count = self.state_count
        eps = torch.finfo(self.dtype).eps
        transition = self.transition_with_potential(transition_potential)
        alpha_rows = []
        scale_rows = []
        first = self.initial_probs.unsqueeze(0) * self.emission_probs[:, obs[:, 0]].transpose(0, 1)
        scale = first.sum(dim=-1).clamp_min(eps)
        alpha_t = first / scale.unsqueeze(-1)
        alpha_rows.append(alpha_t)
        scale_rows.append(scale)
        for t in range(1, seq_len):
            current = torch.matmul(alpha_t, transition) * self.emission_probs[:, obs[:, t]].transpose(0, 1)
            scale = current.sum(dim=-1).clamp_min(eps)
            current = current / scale.unsqueeze(-1)
            alpha_t = torch.where(mask[:, t].unsqueeze(-1), current, alpha_t)
            alpha_rows.append(alpha_t)
            scale_rows.append(torch.where(mask[:, t], scale, torch.ones_like(scale)))
        alpha = torch.stack(alpha_rows, dim=1)
        scales = torch.stack(scale_rows, dim=1)

        beta_rows: list[torch.Tensor | None] = [None] * seq_len
        beta_t = torch.ones((batch, state_count), dtype=self.dtype, device=self.device)
        beta_rows[-1] = beta_t
        for t in range(seq_len - 2, -1, -1):
            next_emit = self.emission_probs[:, obs[:, t + 1]].transpose(0, 1)
            current = torch.matmul(transition.unsqueeze(0), (next_emit * beta_t).unsqueeze(-1)).squeeze(-1)
            current = current / scales[:, t + 1].clamp_min(eps).unsqueeze(-1)
            beta_t = torch.where(mask[:, t + 1].unsqueeze(-1), current, beta_t)
            beta_rows[t] = beta_t
        beta = torch.stack([row for row in beta_rows if row is not None], dim=1)

        gamma = alpha * beta
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(eps)
        gamma = torch.where(mask.unsqueeze(-1), gamma, torch.zeros_like(gamma))

        xi_rows = []
        for t in range(seq_len - 1):
            next_emit = self.emission_probs[:, obs[:, t + 1]].transpose(0, 1)
            pair = alpha[:, t, :, None] * transition.unsqueeze(0) * (next_emit * beta[:, t + 1, :])[:, None, :]
            pair = pair / pair.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
            xi_rows.append(torch.where(mask[:, t + 1].view(batch, 1, 1), pair, torch.zeros_like(pair)))
        xi = (
            torch.stack(xi_rows, dim=1)
            if xi_rows
            else torch.zeros((batch, 0, state_count, state_count), dtype=self.dtype, device=self.device)
        )

        log_likelihood = torch.log(scales.clamp_min(eps)).masked_fill(~mask, 0.0).sum(dim=-1)
        return HMMForwardBackward(alpha=alpha, beta=beta, gamma=gamma, xi=xi, scales=scales, log_likelihood=log_likelihood, mask=mask)

    def viterbi(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs, lengths_t, _mask = self._prepare_observations(observations, lengths)
        batch, seq_len = obs.shape
        log_initial = torch.log(self.initial_probs.clamp_min(torch.finfo(self.dtype).eps))
        transition = self.transition_with_potential(transition_potential)
        log_transition = torch.log(transition.clamp_min(torch.finfo(self.dtype).eps))
        log_emission = torch.log(self.emission_probs.clamp_min(torch.finfo(self.dtype).eps))
        delta = log_initial.unsqueeze(0) + log_emission[:, obs[:, 0]].transpose(0, 1)
        backpointers = torch.zeros((batch, seq_len, self.state_count), dtype=torch.long, device=self.device)
        deltas = [delta]
        for t in range(1, seq_len):
            scores = delta.unsqueeze(2) + log_transition.unsqueeze(0)
            best, arg = scores.max(dim=1)
            delta = best + log_emission[:, obs[:, t]].transpose(0, 1)
            deltas.append(delta)
            backpointers[:, t, :] = arg
        paths = torch.zeros((batch, seq_len), dtype=torch.long, device=self.device)
        log_scores = torch.empty((batch,), dtype=self.dtype, device=self.device)
        stacked = torch.stack(deltas, dim=1)
        for b in range(batch):
            last = int(lengths_t[b].item()) - 1
            score, state = stacked[b, last].max(dim=0)
            log_scores[b] = score
            paths[b, last] = state
            for t in range(last, 0, -1):
                state = backpointers[b, t, state]
                paths[b, t - 1] = state
        return paths, log_scores

    def sample(
        self,
        batch_size: int,
        max_length: int,
        *,
        generator: torch.Generator | None = None,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        transition = self.transition_with_potential(transition_potential)
        states = torch.multinomial(self.initial_probs, batch_size, replacement=True, generator=generator)
        outputs = []
        for _ in range(max_length):
            probs = self.emission_probs.index_select(0, states)
            symbol = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            outputs.append(symbol)
            next_probs = transition.index_select(0, states)
            states = torch.multinomial(next_probs, 1, generator=generator).squeeze(-1)
        return torch.stack(outputs, dim=1)

    def extract_argmax_dfa(self, accept_probability_threshold: float = 0.0) -> DFA:
        alphabet = frozenset(self.symbols)
        states = frozenset(range(self.state_count))
        transitions = {}
        scores = self.transition_probs.unsqueeze(-1) * self.emission_probs.transpose(0, 1).unsqueeze(0)
        for state in states:
            for sym_idx, symbol in enumerate(self.symbols):
                transitions[(state, symbol)] = int(torch.argmax(scores[state, :, sym_idx]).item())
        start = int(torch.argmax(self.initial_probs).item())
        accepting = {
            state
            for state in states
            if float(torch.max(self.emission_probs[state]).item()) >= accept_probability_threshold
        }
        if not accepting:
            accepting = set(states)
        return DFA(states=states, alphabet=alphabet, transitions=transitions, start_state=start, accepting_states=frozenset(accepting))

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"symbols": list(self.symbols), "state_names": list(self.state_names), "dtype": str(self.dtype).replace("torch.", "")}
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
        torch.save(
            {"initial": self.initial_probs.detach().cpu(), "transition": self.transition_probs.detach().cpu(), "emission": self.emission_probs.detach().cpu()},
            path / "model.pt",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "DiscreteHMM":
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        weights = torch.load(path / "model.pt", map_location=device or "cpu", weights_only=True)
        return cls(weights["transition"], weights["emission"], weights["initial"], config["symbols"], state_names=config.get("state_names"), normalize=False, device=device, dtype=dtype)

    @classmethod
    def baum_welch(
        cls,
        sequences: Sequence[Sequence[object]],
        symbols: Sequence[object],
        state_count: int,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        smoothing: float = 1e-9,
        init: "DiscreteHMM | ProbabilisticAutomaton | HMMParameters | None" = None,
        random_seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "BaumWelchResult":
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
        elif isinstance(init, DiscreteHMM):
            if init.symbols != tuple(symbols) or init.state_count != state_count:
                raise ValueError("init must match symbols and state_count")
            model = init.to(device=device, dtype=dtype)
        else:
            params = _coerce_init(init, symbols, state_count)
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
        return BaumWelchResult(model=model, log_likelihoods=tuple(log_likelihoods), iterations=iteration + 1, converged=converged)


@dataclass(frozen=True)
class ProbabilisticAutomaton:
    """Legacy immutable discrete Hidden Markov Model (HMM) wrapper.

    Prefer :class:`DiscreteHMM` for new production code. This frozen dataclass
    remains for source compatibility with older examples and tests.

    Attributes:
        transition: Square matrix of shape ``(S, S)`` where
            ``transition[i][j]`` is the probability of moving from state *i*
            to state *j*.
        emission: Matrix of shape ``(S, V)`` where ``emission[i][k]`` is the
            probability of emitting the *k*-th symbol from state *i*.
        initial: Length-*S* vector of initial state probabilities.
        symbols: Ordered tuple of observable symbol strings (vocabulary).
    """
    transition: tuple[tuple[float, ...], ...]
    emission: tuple[tuple[float, ...], ...]
    initial: tuple[float, ...]
    symbols: tuple[str, ...]

    def __init__(
        self,
        transition: Sequence[Sequence[float]],
        emission: Sequence[Sequence[float]],
        initial: Sequence[float],
        symbols: Sequence[str],
    ):
        transition = tuple(tuple(float(v) for v in row) for row in transition)
        emission = tuple(tuple(float(v) for v in row) for row in emission)
        initial = tuple(float(v) for v in initial)
        symbols = tuple(symbols)
        if len(transition) != len(emission) or len(transition) != len(initial):
            raise ValueError("transition, emission, and initial must agree on state count")
        if any(len(row) != len(transition) for row in transition):
            raise ValueError("transition matrix must be square")
        if any(len(row) != len(symbols) for row in emission):
            raise ValueError("emission rows must match symbol count")
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "emission", emission)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "symbols", symbols)

    @property
    def state_count(self) -> int:
        """Number of hidden states in the model."""
        return len(self.initial)

    def sequence_probability(self, sequence: Iterable[str]) -> float:
        """Compute the total probability of an observation sequence.

        Uses the forward algorithm without scaling; suitable for short
        sequences or when numerical precision is not a concern.

        Args:
            sequence: Ordered iterable of symbol strings from ``self.symbols``.

        Returns:
            The marginal probability P(sequence) summed over all hidden paths.
        """
        # NOTE: This is intentionally the *unscaled* forward pass.  For long
        # sequences use ``baum_welch_train`` which relies on the scaled version.
        probs = list(self.initial)
        symbol_index = {symbol: i for i, symbol in enumerate(self.symbols)}
        for symbol in sequence:
            e_idx = symbol_index[symbol]
            emitted = [probs[state] * self.emission[state][e_idx] for state in range(self.state_count)]
            probs = [
                sum(emitted[src] * self.transition[src][dst] for src in range(self.state_count))
                for dst in range(self.state_count)
            ]
        return sum(probs)

    def extract_argmax_dfa(self, accept_probability_threshold: float = 0.0) -> DFA:
        """Convert the HMM to a deterministic finite automaton (DFA).

        At each state the transition for a symbol is wired to the successor
        state that maximises ``transition[state][dst] * emission[dst][symbol]``
        (the most likely next state given the observed symbol).  The initial
        DFA state is the most probable initial HMM state.

        Args:
            accept_probability_threshold: A state is made accepting if its
                maximum emission probability exceeds this threshold.  Defaults
                to ``0.0`` (all states are accepting unless the resulting set
                would be empty).

        Returns:
            A :class:`~.dfa.DFA` whose alphabet equals ``self.symbols``.
        """
        # Build argmax transitions: for each (state, symbol) pair choose the
        # destination state with the highest joint transition × emission weight.
        alphabet = frozenset(self.symbols)
        states = frozenset(range(self.state_count))
        transitions = {}
        for state in states:
            for sym_idx, symbol in enumerate(self.symbols):
                best_dst = max(
                    range(self.state_count),
                    key=lambda dst: self.transition[state][dst] * self.emission[dst][sym_idx],
                )
                transitions[(state, symbol)] = best_dst

        start = max(range(self.state_count), key=lambda state: self.initial[state])
        accepting = {
            state
            for state in states
            if max(self.emission[state]) >= accept_probability_threshold
        }
        if not accepting:
            accepting = set(states)
        return DFA(
            states=states,
            alphabet=alphabet,
            transitions=transitions,
            start_state=start,
            accepting_states=frozenset(accepting),
        )


@dataclass(frozen=True)
class HMMParameters:
    """Raw (unnormalised) HMM parameter arrays used to seed ``baum_welch_train``.

    Unlike :class:`ProbabilisticAutomaton` this dataclass does *not* validate
    or normalise its contents; normalisation happens inside ``_coerce_init``.

    Attributes:
        transition: Square ``(S, S)`` transition probability matrix.
        emission: ``(S, V)`` emission probability matrix.
        initial: Length-*S* vector of initial state probabilities.
    """
    transition: tuple[tuple[float, ...], ...]
    emission: tuple[tuple[float, ...], ...]
    initial: tuple[float, ...]


@dataclass(frozen=True)
class BaumWelchResult:
    """Output of :func:`baum_welch_train`.

    Attributes:
        model: The trained :class:`ProbabilisticAutomaton`.
        log_likelihoods: Per-iteration total log-likelihood of all training
            sequences.  Monotonically non-decreasing for a correct EM run.
        iterations: Number of EM iterations actually performed.
        converged: ``True`` if the log-likelihood improvement fell below
            ``tol`` before ``max_iter`` was reached.
    """
    model: DiscreteHMM
    log_likelihoods: tuple[float, ...]
    iterations: int
    converged: bool


def baum_welch_train(
    sequences: Sequence[Sequence[str]],
    symbols: Sequence[str],
    state_count: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    smoothing: float = 1e-9,
    init: DiscreteHMM | ProbabilisticAutomaton | HMMParameters | None = None,
    random_seed: int = 0,
) -> BaumWelchResult:
    """Train a discrete HMM with Baum-Welch expectation maximization.

    Production implementation backed by batched Torch forward/backward. String
    sequences are encoded once, then EM updates are vectorized across the
    training batch.
    """
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


def compare_hmm_dfa(
    hmm: ProbabilisticAutomaton,
    dfa: DFA,
    sequences: Iterable[Sequence[str]],
    probability_threshold: float = 0.0,
) -> dict[str, float]:
    """Compare HMM acceptance against a reference DFA over a sequence corpus.

    The HMM is treated as a binary classifier that *accepts* a sequence when
    ``hmm.sequence_probability(sequence) > probability_threshold``.
    Precision, recall, and confusion-matrix counts are computed using the DFA
    as the ground-truth acceptor.

    Args:
        hmm: Trained probabilistic automaton.
        dfa: Reference deterministic finite automaton acting as ground truth.
        sequences: Corpus of symbol sequences to evaluate.
        probability_threshold: HMM probability above which a sequence is
            classified as accepted.  Defaults to ``0.0``.

    Returns:
        Dictionary with keys ``precision``, ``recall``, ``true_positive``,
        ``false_positive``, ``true_negative``, ``false_negative``, and
        ``mean_hmm_probability``.
    """
    tp = fp = tn = fn = 0
    total_probability = 0.0
    total = 0
    for sequence in sequences:
        probability = hmm.sequence_probability(sequence)
        hmm_accepts = probability > probability_threshold
        dfa_accepts = dfa.accepts(sequence)
        total_probability += probability
        total += 1
        # Update confusion matrix using DFA as ground truth.
        if hmm_accepts and dfa_accepts:
            tp += 1  # both accept
        elif not hmm_accepts and dfa_accepts:
            fp += 1  # DFA accepts, HMM rejects → false positive for the DFA class
        elif hmm_accepts and not dfa_accepts:
            fn += 1  # HMM accepts, DFA rejects → false negative for the DFA class
        else:
            tn += 1  # both reject
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "true_negative": float(tn),
        "false_negative": float(fn),
        "mean_hmm_probability": total_probability / total if total else 0.0,
    }


def all_sequences(symbols: Sequence[str], max_length: int) -> list[tuple[str, ...]]:
    """Return all symbol sequences of length 0 through *max_length*.

    The empty sequence ``()`` is always included as the first element.

    Args:
        symbols: The alphabet to draw symbols from.
        max_length: Maximum sequence length (inclusive).

    Returns:
        A list of tuples, ordered by increasing length then lexicographic order
        within each length (as determined by ``itertools.product``).
    """
    output: list[tuple[str, ...]] = [()]  # start with the empty sequence
    for length in range(1, max_length + 1):
        output.extend(tuple(seq) for seq in product(symbols, repeat=length))
    return output


def _validate_symbol_tuple(symbols: tuple[object, ...]) -> None:
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")


def _normalize_tensor(values: torch.Tensor, dim: int) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("probability tensors must contain only finite values")
    values = values.clamp_min(0)
    total = values.sum(dim=dim, keepdim=True)
    if torch.any(total <= 0):
        size = values.shape[dim]
        return torch.full_like(values, 1.0 / float(size))
    return values / total


def _validate_hmm_tensors(initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor, symbol_count: int) -> None:
    if initial.dim() != 1 or initial.numel() < 1:
        raise ValueError("initial must be a non-empty vector")
    state_count = initial.numel()
    if transition.shape != (state_count, state_count):
        raise ValueError("transition must have shape [state_count, state_count]")
    if emission.shape != (state_count, symbol_count):
        raise ValueError("emission must have shape [state_count, symbol_count]")
    if not torch.isfinite(initial).all() or not torch.isfinite(transition).all() or not torch.isfinite(emission).all():
        raise ValueError("HMM parameters must be finite")
    if torch.any(initial < 0) or torch.any(transition < 0) or torch.any(emission < 0):
        raise ValueError("HMM probability parameters must be non-negative")


def _validate_and_encode_sequences(
    sequences: Sequence[Sequence[str]],
    symbols: Sequence[str],
    state_count: int,
) -> tuple[list[list[int]], tuple[str, ...]]:
    """Validate inputs and encode symbol strings to integer indices.

    Args:
        sequences: Training sequences of symbol strings.
        symbols: Ordered vocabulary.
        state_count: Requested number of hidden states.

    Returns:
        A tuple of ``(encoded_sequences, symbols)`` where each inner list
        contains integer indices into *symbols*.

    Raises:
        ValueError: On empty sequences/symbols, duplicate symbols, unknown
            symbols, or ``state_count < 1``.
    """
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
    init: DiscreteHMM | ProbabilisticAutomaton | HMMParameters,
    symbols: tuple[str, ...],
    state_count: int,
) -> HMMParameters:
    """Normalise a warm-start object into a validated :class:`HMMParameters`.

    Accepts either a fully-fledged :class:`ProbabilisticAutomaton` or a raw
    :class:`HMMParameters` dataclass, validates the shapes against
    *state_count* and *symbols*, then row-normalises every probability vector.

    Raises:
        ValueError: If shapes are inconsistent or symbols don't match.
        TypeError: If *init* is neither of the expected types.
    """
    if isinstance(init, DiscreteHMM):
        if init.symbols != symbols:
            raise ValueError("init symbols must match symbols")
        params = HMMParameters(
            tuple(tuple(float(v) for v in row.tolist()) for row in init.transition_probs.detach().cpu()),
            tuple(tuple(float(v) for v in row.tolist()) for row in init.emission_probs.detach().cpu()),
            tuple(float(v) for v in init.initial_probs.detach().cpu().tolist()),
        )
    elif isinstance(init, ProbabilisticAutomaton):
        if init.symbols != symbols:
            raise ValueError("init symbols must match symbols")
        params = HMMParameters(init.transition, init.emission, init.initial)
    elif isinstance(init, HMMParameters):
        params = init
    else:
        raise TypeError("init must be DiscreteHMM, ProbabilisticAutomaton, HMMParameters, or None")

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


def _random_stochastic_vector(size: int, rng: Random) -> list[float]:
    """Generate a random normalised probability vector of length *size*.

    Each element is drawn from Uniform(0,1) + 1e-3 to avoid zero entries
    before normalisation.
    """
    return _normalize_row([rng.random() + 1e-3 for _ in range(size)], 0.0)


def _random_stochastic_matrix(rows: int, cols: int, rng: Random) -> list[list[float]]:
    """Generate a random row-stochastic matrix of shape ``(rows, cols)``."""
    return [_random_stochastic_vector(cols, rng) for _ in range(rows)]


def _normalize_row(row: Sequence[float], smoothing: float) -> list[float]:
    """Normalise *row* to a valid probability distribution.

    Negative values are clamped to zero before *smoothing* is added to every
    element.  If the resulting sum is still non-positive (all-zero after
    clamping and no smoothing) the uniform distribution is returned.

    Args:
        row: Raw non-negative counts or probability estimates.
        smoothing: Additive (Laplace-style) smoothing constant.

    Returns:
        A list of floats that sums to 1.0.
    """
    if not row:
        raise ValueError("cannot normalize an empty row")
    smoothed = [max(float(value), 0.0) + smoothing for value in row]
    total = sum(smoothed)
    if total <= 0.0:
        # Fallback: return the uniform distribution to avoid division by zero.
        return [1.0 / len(row) for _ in row]
    return [value / total for value in smoothed]


def _forward_scaled(
    obs: Sequence[int],
    initial: Sequence[float],
    transition: Sequence[Sequence[float]],
    emission: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], float]:
    """Compute the *scaled* forward variable alpha and the sequence log-likelihood.

    At each time step the raw alpha vector is divided by its sum (the scale
    factor) and the log of the scale is accumulated into the log-likelihood.
    This prevents floating-point underflow for long sequences.

    Args:
        obs: Integer-encoded observation sequence.
        initial: Initial state distribution.
        transition: Row-stochastic transition matrix.
        emission: Row-stochastic emission matrix.

    Returns:
        ``(alpha, scales, log_likelihood)`` where *alpha* is ``T × S``,
        *scales* is length *T*, and *log_likelihood* is the total log P(obs).
    """
    state_count = len(initial)
    alpha: list[list[float]] = []
    scales: list[float] = []
    log_likelihood = 0.0

    # Initialise alpha at t=0 with initial × emission probabilities.
    first = [initial[state] * emission[state][obs[0]] for state in range(state_count)]
    first, scale = _scale_probabilities(first)
    alpha.append(first)
    scales.append(scale)
    log_likelihood += _safe_log(scale)

    # Induction step: propagate alpha forward one time step at a time.
    for t in range(1, len(obs)):
        row = []
        for dst in range(state_count):
            prob = sum(alpha[t - 1][src] * transition[src][dst] for src in range(state_count))
            row.append(prob * emission[dst][obs[t]])
        row, scale = _scale_probabilities(row)
        alpha.append(row)
        scales.append(scale)
        log_likelihood += _safe_log(scale)

    return alpha, scales, log_likelihood


def _backward_scaled(
    obs: Sequence[int],
    transition: Sequence[Sequence[float]],
    emission: Sequence[Sequence[float]],
    scales: Sequence[float],
) -> list[list[float]]:
    """Compute the *scaled* backward variable beta.

    The same scale factors produced by :func:`_forward_scaled` are reused
    here so that alpha[t] * beta[t] yields properly normalised posteriors
    without additional re-scaling.

    Args:
        obs: Integer-encoded observation sequence.
        transition: Row-stochastic transition matrix.
        emission: Row-stochastic emission matrix.
        scales: Per-timestep scale factors from the forward pass.

    Returns:
        Beta matrix of shape ``T × S``.
    """
    state_count = len(transition)
    beta = [[0.0 for _ in range(state_count)] for _ in obs]
    # Initialise beta at the last time step to all-ones (unscaled).
    beta[-1] = [1.0 for _ in range(state_count)]

    # Induction step: propagate beta backward, dividing by the forward scale
    # at the *next* time step to keep values numerically stable.
    for t in range(len(obs) - 2, -1, -1):
        scale = scales[t + 1] if scales[t + 1] > 0.0 else 1.0
        for src in range(state_count):
            beta[t][src] = (
                sum(
                    transition[src][dst] * emission[dst][obs[t + 1]] * beta[t + 1][dst]
                    for dst in range(state_count)
                )
                / scale
            )
    return beta


def _expectations(
    obs: Sequence[int],
    alpha: Sequence[Sequence[float]],
    beta: Sequence[Sequence[float]],
    transition: Sequence[Sequence[float]],
    emission: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """Compute the E-step posteriors gamma and xi from forward-backward variables.

    - **gamma[t][s]**: probability of being in state *s* at time *t* given the
      full observation sequence.
    - **xi[t][src][dst]**: probability of transitioning from *src* to *dst*
      between times *t* and *t+1* given the full observation sequence.

    Args:
        obs: Integer-encoded observation sequence.
        alpha: Scaled forward variable from :func:`_forward_scaled`.
        beta: Scaled backward variable from :func:`_backward_scaled`.
        transition: Current transition matrix estimate.
        emission: Current emission matrix estimate.

    Returns:
        ``(gamma, xi)`` — gamma is ``T × S``, xi is ``(T-1) × S × S``.
    """
    state_count = len(transition)
    # --- Gamma: state-occupation posteriors ---
    gamma = []
    for t in range(len(obs)):
        # alpha[t] * beta[t] is proportional to P(state | obs); normalise to
        # obtain a proper distribution.
        row = [alpha[t][state] * beta[t][state] for state in range(state_count)]
        gamma.append(_normalize_row(row, 0.0))

    # --- Xi: transition posteriors ---
    xi = []
    for t in range(len(obs) - 1):
        matrix = [[0.0 for _ in range(state_count)] for _ in range(state_count)]
        denom = 0.0
        for src in range(state_count):
            for dst in range(state_count):
                value = (
                    alpha[t][src]
                    * transition[src][dst]
                    * emission[dst][obs[t + 1]]
                    * beta[t + 1][dst]
                )
                matrix[src][dst] = value
                denom += value
        if denom <= 0.0:
            # Numerical underflow: fall back to a uniform joint distribution.
            uniform = 1.0 / (state_count * state_count)
            matrix = [[uniform for _ in range(state_count)] for _ in range(state_count)]
        else:
            matrix = [[value / denom for value in row] for row in matrix]
        xi.append(matrix)
    return gamma, xi


def _scale_probabilities(row: Sequence[float]) -> tuple[list[float], float]:
    """Normalise a probability row and return the scale factor.

    Args:
        row: Non-negative probability values at a single time step.

    Returns:
        ``(normalised_row, scale)`` where *scale* is the pre-normalisation sum.
        If the sum is zero (complete underflow) returns the uniform distribution
        and a scale of ``0.0`` so the caller can detect the degenerate case.
    """
    scale = sum(row)
    if scale <= 0.0:
        # All probabilities collapsed to zero — return uniform and signal via scale=0.
        return [1.0 / len(row) for _ in row], 0.0
    return [value / scale for value in row], scale


def _safe_log(value: float) -> float:
    """Return ``math.log(value)``, or ``-inf`` for non-positive inputs.

    Used when accumulating the log-likelihood from per-step scale factors;
    a scale of 0.0 indicates numerical underflow and contributes ``-inf``.
    """
    if value <= 0.0:
        return float("-inf")
    import math

    return math.log(value)
