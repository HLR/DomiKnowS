"""Hidden Markov Model (HMM) training and inference utilities.

Provides:
- ``DiscreteHMM``: a Torch-backed discrete HMM that can score batched
  observation sequences, expose forward/backward factors, run Viterbi, sample,
  serialize, and be extracted into a deterministic finite automaton (DFA).
- ``HMMParameters`` / ``BaumWelchResult``: lightweight data containers.
- ``baum_welch_train``: batched Torch Baum-Welch EM training with scaled
  forward-backward to avoid floating-point underflow.
- ``compare_hmm_dfa``: evaluation helper that compares HMM acceptance against
  a reference DFA over a corpus of sequences.

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .baumWelchDiscretHMM import BaumWelchResult, HMMParameters, _normalize_tensor, baum_welch_train, run_baum_welch
from ....dfa import DFA
from ....latent import LatentTransitionPotential, apply_hmm_transition_potential


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
        # Initialize validated model parameters and symbol/state metadata.
        symbols = tuple(symbols)
        _validate_symbol_tuple(symbols)
        dtype = dtype or torch.float32
        transition_t = torch.as_tensor(transition, dtype=dtype, device=device)
        emission_t = torch.as_tensor(emission, dtype=dtype, device=device)
        initial_t = torch.as_tensor(initial, dtype=dtype, device=device)
        if normalize:
            # Keep all parameter tensors row-stochastic unless caller opts out.
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
        # Return the number of hidden states.
        return int(self.initial_probs.numel())

    @property
    def symbol_count(self) -> int:
        # Return the vocabulary size.
        return len(self.symbols)

    @property
    def transition(self) -> torch.Tensor:
        # Provide read-only-style access to transition probabilities.
        return self.transition_probs

    @property
    def emission(self) -> torch.Tensor:
        # Provide read-only-style access to emission probabilities.
        return self.emission_probs

    def transition_with_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> torch.Tensor:
        """Return transitions after optional latent-potential reweighting."""
        # Delegate potential application and keep transition semantics centralized.
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def with_transition_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]],
    ) -> "DiscreteHMM":
        """Return a new HMM with transition dynamics reweighted by *transition_potential*."""
        # Create a new immutable-style view with reweighted transitions.
        return DiscreteHMM(
            self.transition_with_potential(transition_potential),
            self.emission_probs,
            self.initial_probs,
            self.symbols,
            state_names=self.state_names,
            normalize=False,
        )

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "DiscreteHMM":
        # Return an equivalent model moved/cast to the requested backend.
        # Rebuild via constructor so invariants (shape/type assumptions) stay centralized.
        return DiscreteHMM(
            self.transition_probs.to(device=device, dtype=dtype or self.dtype),
            self.emission_probs.to(device=device, dtype=dtype or self.dtype),
            self.initial_probs.to(device=device, dtype=dtype or self.dtype),
            self.symbols,
            state_names=self.state_names,
            normalize=False,
        )

    def encode(self, sequences: Sequence[Sequence[object]]) -> tuple[torch.Tensor, torch.Tensor]:
        # Map symbol sequences to padded index tensors and per-sequence lengths.
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
        # Right-pad variable-length encoded sequences into a single batch tensor.
        padded = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)
        for idx, row in enumerate(encoded):
            padded[idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
        return padded, torch.tensor(lengths, dtype=torch.long, device=self.device)

    def _prepare_observations(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Canonicalize user inputs into one batched format shared by all inference paths.
        # observations[b, t] is the observed/emitted symbol index at timestep t for batch item b.
        # It must be in [0, symbol_count-1] and is distinct from the latent (hidden) state.
        # Returns:
        # - obs: shape [batch, seq_len]
        # - lengths_t: valid length per batch item
        # - mask: True on valid timesteps, False on right-padding positions
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
        # mask[b, t] is True only for valid timesteps of sequence b.
        mask = torch.arange(obs.shape[1], device=self.device).unsqueeze(0) < lengths_t.unsqueeze(1)
        return obs, lengths_t, mask

    def log_prob(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> torch.Tensor:
        # Return batched log-likelihoods for observation sequences.
        # forward_backward already computes numerically stable log-likelihoods.
        factors = self.forward_backward(observations, lengths, transition_potential=transition_potential)
        return factors.log_likelihood

    def sequence_probability(
        self,
        sequence: Sequence[object],
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> float:
        # Convenience wrapper that returns a scalar probability for one sequence.
        if not sequence:
            return 1.0
        # Keep scalar API for compatibility: encode one sequence and decode from log-space.
        obs, lengths = self.encode([sequence])
        return float(torch.exp(self.log_prob(obs, lengths, transition_potential=transition_potential))[0].item())

    def forward_backward(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> HMMForwardBackward:
        # Compute scaled forward/backward factors and posterior marginals.
        obs, lengths_t, mask = self._prepare_observations(observations, lengths)
        batch, seq_len = obs.shape
        state_count = self.state_count
        eps = torch.finfo(self.dtype).eps
        transition = self.transition_with_potential(transition_potential)

        # --- Forward pass (scaled) ---
        # alpha[t] stores p(z_t | x_{<=t}) up to a per-step normalization.
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
            # For padded positions, keep the previous alpha unchanged.
            alpha_t = torch.where(mask[:, t].unsqueeze(-1), current, alpha_t)
            alpha_rows.append(alpha_t)
            # Scale of padded steps is neutral so they do not affect log-likelihood.
            scale_rows.append(torch.where(mask[:, t], scale, torch.ones_like(scale)))
        alpha = torch.stack(alpha_rows, dim=1)
        scales = torch.stack(scale_rows, dim=1)

        # --- Backward pass (scaled with forward scales) ---
        beta_rows: list[torch.Tensor | None] = [None] * seq_len
        beta_t = torch.ones((batch, state_count), dtype=self.dtype, device=self.device)
        beta_rows[-1] = beta_t
        for t in range(seq_len - 2, -1, -1):
            next_emit = self.emission_probs[:, obs[:, t + 1]].transpose(0, 1)
            current = torch.matmul(transition.unsqueeze(0), (next_emit * beta_t).unsqueeze(-1)).squeeze(-1)
            current = current / scales[:, t + 1].clamp_min(eps).unsqueeze(-1)
            # Do not back-propagate through padded suffixes.
            beta_t = torch.where(mask[:, t + 1].unsqueeze(-1), current, beta_t)
            beta_rows[t] = beta_t
        beta = torch.stack([row for row in beta_rows if row is not None], dim=1)

        # Posterior state marginals p(z_t | x_{1:T}).
        gamma = alpha * beta
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(eps)
        gamma = torch.where(mask.unsqueeze(-1), gamma, torch.zeros_like(gamma))

        # Posterior transition marginals p(z_t=i, z_{t+1}=j | x_{1:T}).
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

        # log P(x) = sum_t log(scale_t) over valid timesteps only.
        log_likelihood = torch.log(scales.clamp_min(eps)).masked_fill(~mask, 0.0).sum(dim=-1)
        return HMMForwardBackward(alpha=alpha, beta=beta, gamma=gamma, xi=xi, scales=scales, log_likelihood=log_likelihood, mask=mask)

    def viterbi(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Decode the most likely hidden-state path for each sequence.
        obs, lengths_t, _mask = self._prepare_observations(observations, lengths)
        batch, seq_len = obs.shape
        # Run decoding in log-space to avoid repeated products of tiny probabilities.
        log_initial = torch.log(self.initial_probs.clamp_min(torch.finfo(self.dtype).eps))
        transition = self.transition_with_potential(transition_potential)
        log_transition = torch.log(transition.clamp_min(torch.finfo(self.dtype).eps))
        log_emission = torch.log(self.emission_probs.clamp_min(torch.finfo(self.dtype).eps))
        delta = log_initial.unsqueeze(0) + log_emission[:, obs[:, 0]].transpose(0, 1)
        backpointers = torch.zeros((batch, seq_len, self.state_count), dtype=torch.long, device=self.device)
        deltas = [delta]
        for t in range(1, seq_len):
            # scores[b, src, dst] = best log-score ending in src then transitioning to dst.
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
            # Trace back only through the valid sequence prefix for batch item b.
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
        # Generate observation samples by alternating emission and transition draws.
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        transition = self.transition_with_potential(transition_potential)
        # Start each sampled trajectory from the initial-state distribution.
        states = torch.multinomial(self.initial_probs, batch_size, replacement=True, generator=generator)
        outputs = []
        for _ in range(max_length):
            # Emit symbol from current state, then transition to next state.
            probs = self.emission_probs.index_select(0, states)
            symbol = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            outputs.append(symbol)
            next_probs = transition.index_select(0, states)
            states = torch.multinomial(next_probs, 1, generator=generator).squeeze(-1)
        return torch.stack(outputs, dim=1)

    def extract_argmax_dfa(self, accept_probability_threshold: float = 0.0) -> DFA:
        # Build a deterministic argmax approximation of this probabilistic model.
        alphabet = frozenset(self.symbols)
        states = frozenset(range(self.state_count))
        transitions = {}
        # Joint score for taking transition i->j and emitting symbol k from j.
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
        # Persist model parameters and metadata to a directory.
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"symbols": list(self.symbols), "state_names": list(self.state_names), "dtype": str(self.dtype).replace("torch.", "")}
        # Keep lightweight metadata in JSON and numeric tensors in torch format.
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
        torch.save(
            {"initial": self.initial_probs.detach().cpu(), "transition": self.transition_probs.detach().cpu(), "emission": self.emission_probs.detach().cpu()},
            path / "model.pt",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "DiscreteHMM":
        # Reconstruct a model instance from files produced by save_pretrained.
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        # Load on CPU by default, then move/cast via constructor arguments.
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
        init: "DiscreteHMM | HMMParameters | None" = None,
        random_seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "BaumWelchResult":
        # Baum-Welch implementation is extracted into baumWelchDiscretHMM.
        return run_baum_welch(
            cls,
            sequences,
            symbols,
            state_count,
            max_iter=max_iter,
            tol=tol,
            smoothing=smoothing,
            init=init,
            random_seed=random_seed,
            device=device,
            dtype=dtype,
        )


def compare_hmm_dfa(
    hmm: DiscreteHMM,
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
        hmm: Trained discrete HMM.
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


def _validate_symbol_tuple(symbols: tuple[object, ...]) -> None:
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")


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

