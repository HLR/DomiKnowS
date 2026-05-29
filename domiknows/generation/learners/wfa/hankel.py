"""Weighted Finite Automaton (WFA) and finite Hankel matrix utilities.

Provides:
- ``WeightedFiniteAutomaton``: a Torch-backed linear WFA that scores batched
  observation sequences via an initial vector, per-symbol transition matrices,
  and a final vector.
- ``ProductDecoderState``: a combined WFA × DFA state used for
  constrained decoding.
- ``hankel_matrix`` / ``constrained_hankel_matrix``: build finite Torch Hankel
  tensors H(u, v) = P(uv) with optional DFA acceptance masking.
- ``projection_summary``: compare Hankel mass before and after constraint
  projection.
- ``start_product_state`` / ``step_product_state`` / ``allowed_product_symbols``:
  utilities for synchronous WFA × DFA traversal.

Type aliases:
- ``Symbol``: any ``Hashable`` value used as an alphabet symbol.
- ``Vector`` / ``Matrix``: Torch tensors representing WFA states and matrices.
"""
from __future__ import annotations

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ...dfa import DFA, State
from ...latent import LatentTransitionPotential, apply_wfa_transition_potential

# Type aliases used throughout this module.
Symbol = Hashable
Vector = torch.Tensor
Matrix = torch.Tensor


class WeightedFiniteAutomaton:
    """An immutable linear Weighted Finite Automaton (WFA).

    Computes the score of an observation sequence ``x_1 … x_T`` as:

    .. math::
        P(x) = \\langle \\alpha, A_{x_1} \\cdots A_{x_T} \\omega \\rangle

    where ``α`` (``initial``) is the initial weight vector, each ``A_σ``
    (``transitions[σ]``) is a square transition matrix, and ``ω`` (``final``)
    is the final weight vector.

    Attributes:
        initial: Initial (row) weight vector of length *S* (state count).
        transitions: Mapping from each alphabet symbol to a square *S × S*
            transition matrix.
        final: Final weight vector of length *S*.
        symbols: Ordered tuple of alphabet symbols; must match the keys of
            ``transitions`` exactly.
    """

    def __init__(
        self,
        initial: torch.Tensor | Sequence[float],
        transitions: Mapping[Symbol, torch.Tensor | Sequence[Sequence[float]]] | torch.Tensor,
        final: torch.Tensor | Sequence[float],
        symbols: Sequence[Symbol],
        *,
        state_names: Sequence[str] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        symbols = tuple(symbols)
        _validate_symbols(symbols)
        if dtype is None:
            dtype = torch.as_tensor(initial).dtype if torch.as_tensor(initial).is_floating_point() else torch.float32
        self.initial = torch.as_tensor(initial, dtype=dtype, device=device)
        self.final = torch.as_tensor(final, dtype=dtype, device=device)
        if self.initial.dim() != 1 or self.initial.numel() < 1:
            raise ValueError("initial must not be empty")
        if self.final.shape != self.initial.shape:
            raise ValueError("initial and final vectors must have the same length")
        if isinstance(transitions, torch.Tensor):
            transition_tensor = transitions.to(device=device, dtype=dtype)
            if transition_tensor.dim() != 3 or transition_tensor.shape[0] != len(symbols):
                raise ValueError("transition tensor must have shape [symbols, states, states]")
        else:
            expected_symbols = set(symbols)
            transition_symbols = set(transitions)
            missing = expected_symbols - transition_symbols
            extra = transition_symbols - expected_symbols
            if missing:
                raise ValueError(f"transitions missing symbol(s): {sorted(missing, key=repr)!r}")
            if extra:
                raise ValueError(f"transitions include unknown symbol(s): {sorted(extra, key=repr)!r}")
            matrices = []
            for symbol in symbols:
                matrix = torch.as_tensor(transitions[symbol], dtype=dtype, device=device)
                if matrix.dim() != 2 or matrix.shape[0] != self.initial.numel():
                    raise ValueError(f"transition matrix for {symbol!r} must have {self.initial.numel()} rows")
                if matrix.shape[1] != self.initial.numel():
                    raise ValueError(f"transition matrix for {symbol!r} must be square")
                matrices.append(matrix)
            transition_tensor = torch.stack(matrices, dim=0)
        _validate_wfa_tensors(self.initial, transition_tensor, self.final)
        if state_names is None:
            state_names = tuple(f"S{i}" for i in range(self.initial.numel()))
        else:
            state_names = tuple(str(name) for name in state_names)
            if len(state_names) != self.initial.numel():
                raise ValueError("state_names length must match state count")
            if len(set(state_names)) != len(state_names):
                raise ValueError("state_names must be unique")
        self.transition_tensor = transition_tensor
        self.symbols = symbols
        self.state_names = tuple(state_names)
        self._symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}

    @property
    def transitions(self) -> Mapping[Symbol, torch.Tensor]:
        return {symbol: self.transition_tensor[idx] for idx, symbol in enumerate(self.symbols)}

    @property
    def state_count(self) -> int:
        """Number of WFA states (dimension of all weight vectors)."""
        return int(self.initial.numel())

    @property
    def device(self) -> torch.device:
        return self.initial.device

    @property
    def dtype(self) -> torch.dtype:
        return self.initial.dtype

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "WeightedFiniteAutomaton":
        return WeightedFiniteAutomaton(
            self.initial.to(device=device, dtype=dtype or self.dtype),
            self.transition_tensor.to(device=device, dtype=dtype or self.dtype),
            self.final.to(device=device, dtype=dtype or self.dtype),
            self.symbols,
            state_names=self.state_names,
        )

    def transition_with_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
        *,
        mode: str = "multiply",
    ) -> torch.Tensor:
        """Return signed WFA transitions after optional latent-potential reweighting."""
        return apply_wfa_transition_potential(self.transition_tensor, transition_potential, mode=mode)

    def with_transition_potential(
        self,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]],
        *,
        mode: str = "multiply",
    ) -> "WeightedFiniteAutomaton":
        """Return a new WFA with transition tensors reweighted by *transition_potential*."""
        return WeightedFiniteAutomaton(
            self.initial,
            self.transition_with_potential(transition_potential, mode=mode),
            self.final,
            self.symbols,
            state_names=self.state_names,
        )

    def encode(self, sequences: Sequence[Sequence[Symbol]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("sequences must not be empty")
        encoded: list[list[int]] = []
        lengths: list[int] = []
        for seq_idx, sequence in enumerate(sequences):
            row = []
            for symbol in sequence:
                if symbol not in self._symbol_index:
                    raise ValueError(f"unknown symbol {symbol!r} in sequence {seq_idx}")
                row.append(self._symbol_index[symbol])
            encoded.append(row)
            lengths.append(len(row))
        max_len = max(max(lengths), 1)
        tensor = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)
        for idx, row in enumerate(encoded):
            if row:
                tensor[idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
        return tensor, torch.tensor(lengths, dtype=torch.long, device=self.device)

    def prefix_state(
        self,
        sequence: Sequence[Symbol],
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
        transition_potential_mode: str = "multiply",
    ) -> Vector:
        """Compute the WFA state reached after consuming *sequence*.

        Starts from ``self.initial`` and multiplies by each transition matrix
        in order, i.e. ``α · A_{x_1} · A_{x_2} · … · A_{x_T}``.

        Args:
            sequence: An ordered sequence of symbols from ``self.symbols``.

        Returns:
            The resulting weight vector of length ``state_count``.

        Raises:
            ValueError: If any symbol in *sequence* is not in the alphabet.
        """
        state = self.initial
        transitions = self.transition_with_potential(transition_potential, mode=transition_potential_mode)
        for symbol in sequence:
            if symbol not in self._symbol_index:
                raise ValueError(f"unknown symbol {symbol!r}")
            state = torch.matmul(state, transitions[self._symbol_index[symbol]])
        return state

    def state_score(self, state: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Project a WFA state onto the final vector to obtain a scalar score.

        Computes the dot product ``⟨state, ω⟩`` where ``ω = self.final``.

        Args:
            state: A weight vector of length ``state_count``.

        Returns:
            The scalar score for this state.

        Raises:
            ValueError: If *state* has the wrong length.
        """
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        if state.shape[-1] != self.state_count:
            raise ValueError("state length must match WFA state_count")
        return torch.matmul(state, self.final)

    def score_batch(
        self,
        observations: torch.Tensor | Sequence[Sequence[int]],
        lengths: torch.Tensor | Sequence[int] | None = None,
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        obs = torch.as_tensor(observations, dtype=torch.long, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.dim() != 2:
            raise ValueError("observations must have shape [seq] or [batch, seq]")
        if lengths is None:
            lengths_t = torch.full((obs.shape[0],), obs.shape[1], dtype=torch.long, device=self.device)
        else:
            lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=self.device).reshape(-1)
        if lengths_t.numel() != obs.shape[0]:
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(lengths_t < 0) or torch.any(lengths_t > obs.shape[1]):
            raise ValueError("lengths must be in [0, seq_len]")
        if torch.any((obs < 0) | (obs >= len(self.symbols))):
            raise ValueError("observations contain labels outside the symbol vocabulary")
        state = self.initial.expand(obs.shape[0], -1)
        transitions = self.transition_with_potential(transition_potential, mode=transition_potential_mode)
        for t in range(obs.shape[1]):
            next_state = torch.bmm(state.unsqueeze(1), transitions.index_select(0, obs[:, t])).squeeze(1)
            state = torch.where((t < lengths_t).unsqueeze(-1), next_state, state)
        return torch.matmul(state, self.final)

    def sequence_probability(
        self,
        sequence: Sequence[Symbol],
        *,
        transition_potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None = None,
        transition_potential_mode: str = "multiply",
    ) -> float:
        """Compute the WFA score P(sequence) = ⟨α · A_{x_1} … A_{x_T}, ω⟩.

        Convenience wrapper combining :meth:`prefix_state` and
        :meth:`state_score`.
        """
        return float(
            self.state_score(
                self.prefix_state(
                    sequence,
                    transition_potential=transition_potential,
                    transition_potential_mode=transition_potential_mode,
                )
            ).item()
        )

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"symbols": list(self.symbols), "state_names": list(self.state_names), "dtype": str(self.dtype).replace("torch.", "")}
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
        torch.save(
            {
                "initial": self.initial.detach().cpu(),
                "transitions": self.transition_tensor.detach().cpu(),
                "final": self.final.detach().cpu(),
            },
            path / "model.pt",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "WeightedFiniteAutomaton":
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        weights = torch.load(path / "model.pt", map_location=device or "cpu", weights_only=True)
        return cls(weights["initial"], weights["transitions"], weights["final"], config["symbols"], state_names=config.get("state_names"), device=device, dtype=dtype)


@dataclass(frozen=True)
class ProductDecoderState:
    """Combined WFA × DFA state for synchronous constrained decoding.

    Tracks the current WFA weight vector and DFA state together so that a
    decoder can efficiently advance both automata one symbol at a time while
    respecting DFA constraints.

    Attributes:
        wfa_state: Current WFA weight vector (``α · A_{x_1} … A_{x_t}``).
        dfa_state: Current DFA state after consuming the same prefix.
        score: Pre-computed ``⟨wfa_state, ω⟩`` for fast ranking without
            re-computing the dot product at every step.  Defaults to ``0.0``.
    """
    wfa_state: Vector
    dfa_state: State
    score: float = 0.0


def hankel_matrix(
    wfa: WeightedFiniteAutomaton,
    prefixes: Sequence[Sequence[Symbol]],
    suffixes: Sequence[Sequence[Symbol]],
) -> torch.Tensor:
    """Build a finite Hankel matrix H(u, v) = P(uv).

    Rows correspond to *prefixes*, columns to *suffixes*.  Each cell contains
    the WFA score of the concatenated sequence ``prefix + suffix``.

    Args:
        wfa: Trained :class:`WeightedFiniteAutomaton`.
        prefixes: Ordered list of prefix sequences (row basis).
        suffixes: Ordered list of suffix sequences (column basis).

    Returns:
        A ``len(prefixes) × len(suffixes)`` list of lists of floats.
    """
    return torch.tensor(
        [
            [wfa.sequence_probability(tuple(prefix) + tuple(suffix)) for suffix in suffixes]
            for prefix in prefixes
        ],
        dtype=wfa.dtype,
        device=wfa.device,
    )


def constrained_hankel_matrix(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    prefixes: Sequence[Sequence[Symbol]],
    suffixes: Sequence[Sequence[Symbol]],
) -> torch.Tensor:
    """Build a DFA-masked Hankel matrix H_C(u, v) = 1[uv ∈ C] · P(uv).

    Identical to :func:`hankel_matrix` except that any sequence ``uv`` not
    accepted by *dfa* contributes ``0.0`` instead of the WFA score.  This
    projects the Hankel matrix onto the constraint language *C* defined by
    the DFA.

    Args:
        wfa: Trained :class:`WeightedFiniteAutomaton`.
        dfa: Reference :class:`~.dfa.DFA` acting as the acceptance constraint.
        prefixes: Ordered list of prefix sequences (row basis).
        suffixes: Ordered list of suffix sequences (column basis).

    Returns:
        A ``len(prefixes) × len(suffixes)`` list of lists of floats.
    """
    rows = []
    for prefix in prefixes:
        row = []
        for suffix in suffixes:
            sequence = tuple(prefix) + tuple(suffix)
            # Mask out sequences rejected by the DFA.
            row.append(wfa.sequence_probability(sequence) if dfa.accepts(sequence) else 0.0)
        rows.append(row)
    return torch.tensor(rows, dtype=wfa.dtype, device=wfa.device)


def projection_summary(
    original: torch.Tensor | Sequence[Sequence[float]],
    constrained: torch.Tensor | Sequence[Sequence[float]],
) -> dict[str, float]:
    """Summarise how much Hankel matrix mass is retained after DFA projection.

    Compares the sum of all cell values in *original* against *constrained*
    to quantify the fraction of probability mass that lies within the
    constraint language.

    Args:
        original: Unconstrained Hankel matrix (from :func:`hankel_matrix`).
        constrained: DFA-masked Hankel matrix (from
            :func:`constrained_hankel_matrix`).  Must have the same shape as
            *original*.

    Returns:
        Dictionary with keys:

        - ``original_mass``: sum of all cells in *original*.
        - ``constrained_mass`` / ``retained_mass``: sum of all cells in
          *constrained* (both keys hold the same value).
        - ``retained_fraction``: ``constrained_mass / original_mass``, or
          ``0.0`` when ``original_mass`` is zero.
        - ``original_nonzero``: count of non-zero cells in *original*.
        - ``constrained_nonzero``: count of non-zero cells in *constrained*.

    Raises:
        ValueError: If *original* and *constrained* have different shapes.
    """
    original_t = torch.as_tensor(original)
    constrained_t = torch.as_tensor(constrained, dtype=original_t.dtype, device=original_t.device)
    if original_t.shape != constrained_t.shape:
        raise ValueError("original and constrained matrices must have the same shape")
    original_mass = float(torch.sum(original_t).item())
    constrained_mass = float(torch.sum(constrained_t).item())
    return {
        "original_mass": original_mass,
        "constrained_mass": constrained_mass,
        "retained_mass": constrained_mass,  # alias for convenience
        "retained_fraction": constrained_mass / original_mass if original_mass else 0.0,
        "original_nonzero": float(torch.count_nonzero(original_t).item()),
        "constrained_nonzero": float(torch.count_nonzero(constrained_t).item()),
    }


def start_product_state(wfa: WeightedFiniteAutomaton, dfa: DFA) -> ProductDecoderState:
    """Create the initial WFA × DFA product state before any symbol is consumed.

    Both automata start at their respective initial states; the score is the
    WFA score of the empty prefix (i.e. ``⟨α, ω⟩``).
    """
    return ProductDecoderState(
        wfa_state=wfa.initial,
        dfa_state=dfa.start_state,
        score=float(wfa.state_score(wfa.initial).item()),
    )


def step_product_state(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    state: ProductDecoderState,
    symbol: Symbol,
) -> ProductDecoderState | None:
    """Advance the WFA × DFA product state by one symbol.

    The DFA is consulted first; if it has no transition for *symbol* from the
    current DFA state, ``None`` is returned immediately (the symbol is
    blocked).  Otherwise both automata advance and a new
    :class:`ProductDecoderState` is returned with an updated WFA state and
    score.

    Args:
        wfa: The weighted finite automaton.
        dfa: The constraint DFA.
        state: The current product state.
        symbol: The symbol to consume.

    Returns:
        The successor :class:`ProductDecoderState`, or ``None`` if the DFA
        blocks *symbol* from the current state.

    Raises:
        ValueError: If *symbol* is not in the WFA alphabet.
    """
    if symbol not in wfa._symbol_index:
        raise ValueError(f"unknown symbol {symbol!r}")
    # Check DFA first — if blocked, no need to advance the WFA.
    next_dfa_state = dfa.step(state.dfa_state, symbol)
    if next_dfa_state is None:
        return None
    next_wfa_state = torch.matmul(state.wfa_state, wfa.transition_tensor[wfa._symbol_index[symbol]])
    return ProductDecoderState(
        wfa_state=next_wfa_state,
        dfa_state=next_dfa_state,
        score=float(wfa.state_score(next_wfa_state).item()),
    )


def allowed_product_symbols(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    state: ProductDecoderState,
) -> set[Symbol]:
    """Return symbols that are valid for both the WFA and the DFA.

    A symbol is *allowed* when it is part of the WFA alphabet **and** the DFA
    permits a transition from the current DFA state (i.e. it is returned by
    :meth:`~.dfa.DFA.allowed_tokens`).  This is the intersection of the two
    reachability constraints.

    Args:
        wfa: The weighted finite automaton (defines the alphabet).
        dfa: The constraint DFA.
        state: The current product state.

    Returns:
        The subset of ``wfa.symbols`` that the DFA allows from the current
        state.
    """
    # Intersect the WFA alphabet with the DFA's allowed tokens at this state.
    return set(wfa.symbols) & dfa.allowed_tokens(state.dfa_state)


def _validate_symbols(symbols: tuple[Symbol, ...]) -> None:
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")


def _validate_wfa_tensors(initial: torch.Tensor, transitions: torch.Tensor, final: torch.Tensor) -> None:
    if initial.dim() != 1 or initial.numel() < 1:
        raise ValueError("initial must not be empty")
    if final.shape != initial.shape:
        raise ValueError("initial and final vectors must have the same length")
    if transitions.dim() != 3:
        raise ValueError("transitions must have shape [symbols, states, states]")
    if transitions.shape[1:] != (initial.numel(), initial.numel()):
        raise ValueError("transition matrices must be square with state_count rows")
    if not torch.isfinite(initial).all() or not torch.isfinite(transitions).all() or not torch.isfinite(final).all():
        raise ValueError("WFA parameters must be finite")


def _validate_vector_shape(initial: Vector, final: Vector, symbols: tuple[Symbol, ...]) -> None:
    """Raise ``ValueError`` if the WFA vector/symbol arguments are malformed."""
    if not initial:
        raise ValueError("initial must not be empty")
    if len(final) != len(initial):
        raise ValueError("initial and final vectors must have the same length")
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")


def _coerce_matrix(matrix: Sequence[Sequence[float]], state_count: int, symbol: Symbol) -> Matrix:
    """Convert *matrix* to an immutable tuple-of-tuples and validate its shape.

    Args:
        matrix: Raw transition matrix provided by the caller.
        state_count: Expected number of rows and columns.
        symbol: The symbol this matrix belongs to (used in error messages).

    Returns:
        An immutable ``Matrix`` (tuple of float tuples) of shape
        ``state_count × state_count``.

    Raises:
        ValueError: If the matrix is not square or has the wrong dimensions.
    """
    matrix = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(matrix) != state_count:
        raise ValueError(f"transition matrix for {symbol!r} must have {state_count} rows")
    if any(len(row) != state_count for row in matrix):
        raise ValueError(f"transition matrix for {symbol!r} must be square")
    return matrix


def _row_times_matrix(row: Sequence[float], matrix: Matrix) -> Vector:
    """Multiply a row vector by a square matrix: ``result[j] = Σ_i row[i] * matrix[i][j]``.

    Args:
        row: Length-*S* weight vector.
        matrix: *S × S* transition matrix.

    Returns:
        Resulting length-*S* vector.

    Raises:
        ValueError: If ``len(row) != len(matrix)``.
    """
    if len(row) != len(matrix):
        raise ValueError("state length must match transition matrix size")
    return tuple(
        sum(float(row[src]) * matrix[src][dst] for src in range(len(row)))
        for dst in range(len(row))
    )


def _dot(left: Vector, right: Vector) -> float:
    """Compute the dot product of two equal-length float vectors."""
    return sum(a * b for a, b in zip(left, right))


def _validate_same_matrix_shape(
    original: Sequence[Sequence[float]],
    constrained: Sequence[Sequence[float]],
) -> None:
    """Raise ``ValueError`` if *original* and *constrained* have different shapes."""
    if len(original) != len(constrained):
        raise ValueError("original and constrained matrices must have the same row count")
    for original_row, constrained_row in zip(original, constrained):
        if len(original_row) != len(constrained_row):
            raise ValueError("original and constrained matrices must have the same shape")
