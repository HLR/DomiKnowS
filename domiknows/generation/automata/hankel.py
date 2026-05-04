"""Weighted Finite Automaton (WFA) and finite Hankel matrix utilities.

Provides:
- ``WeightedFiniteAutomaton``: a linear WFA that scores observation sequences
  via an initial vector, per-symbol transition matrices, and a final vector.
- ``ProductDecoderState``: a combined WFA × DFA state used for
  constrained decoding.
- ``hankel_matrix`` / ``constrained_hankel_matrix``: build finite Hankel
  tables H(u, v) = P(uv) with optional DFA acceptance masking.
- ``projection_summary``: compare Hankel mass before and after constraint
  projection.
- ``start_product_state`` / ``step_product_state`` / ``allowed_product_symbols``:
  utilities for synchronous WFA × DFA traversal.

Type aliases:
- ``Symbol``: any ``Hashable`` value used as an alphabet symbol.
- ``Vector``: a ``tuple[float, ...]`` representing a WFA state or weight vector.
- ``Matrix``: a ``tuple[tuple[float, ...], ...]`` representing a square
  transition matrix.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from .dfa import DFA, State

# Type aliases used throughout this module.
Symbol = Hashable
Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
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

    initial: Vector
    transitions: Mapping[Symbol, Matrix]
    final: Vector
    symbols: tuple[Symbol, ...]

    def __init__(
        self,
        initial: Sequence[float],
        transitions: Mapping[Symbol, Sequence[Sequence[float]]],
        final: Sequence[float],
        symbols: Sequence[Symbol],
    ):
        """Construct a WFA, validating and normalising all inputs.

        Args:
            initial: Initial weight vector (length *S*).
            transitions: Mapping from symbol to an *S × S* matrix given as a
                sequence of rows.  Each row must have length *S*.
            final: Final weight vector (length *S*).
            symbols: Ordered alphabet; must match the keys of *transitions*.

        Raises:
            ValueError: If shapes are inconsistent, symbols are missing/extra,
                or any symbol is duplicated.
        """
        # Convert to canonical immutable types for the frozen dataclass.
        initial = tuple(float(value) for value in initial)
        final = tuple(float(value) for value in final)
        symbols = tuple(symbols)
        _validate_vector_shape(initial, final, symbols)

        # Ensure transition dict covers exactly the declared symbols.
        expected_symbols = set(symbols)
        transition_symbols = set(transitions)
        missing = expected_symbols - transition_symbols
        extra = transition_symbols - expected_symbols
        if missing:
            raise ValueError(f"transitions missing symbol(s): {sorted(missing, key=repr)!r}")
        if extra:
            raise ValueError(f"transitions include unknown symbol(s): {sorted(extra, key=repr)!r}")

        # Coerce each transition matrix to a validated immutable tuple-of-tuples.
        matrices = {}
        for symbol in symbols:
            matrices[symbol] = _coerce_matrix(transitions[symbol], len(initial), symbol)

        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "transitions", matrices)
        object.__setattr__(self, "final", final)
        object.__setattr__(self, "symbols", symbols)

    @property
    def state_count(self) -> int:
        """Number of WFA states (dimension of all weight vectors)."""
        return len(self.initial)

    def prefix_state(self, sequence: Sequence[Symbol]) -> Vector:
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
        for symbol in sequence:
            if symbol not in self.transitions:
                raise ValueError(f"unknown symbol {symbol!r}")
            # Left-multiply the current state vector by the transition matrix.
            state = _row_times_matrix(state, self.transitions[symbol])
        return state

    def state_score(self, state: Sequence[float]) -> float:
        """Project a WFA state onto the final vector to obtain a scalar score.

        Computes the dot product ``⟨state, ω⟩`` where ``ω = self.final``.

        Args:
            state: A weight vector of length ``state_count``.

        Returns:
            The scalar score for this state.

        Raises:
            ValueError: If *state* has the wrong length.
        """
        if len(state) != self.state_count:
            raise ValueError("state length must match WFA state_count")
        return _dot(tuple(float(value) for value in state), self.final)

    def sequence_probability(self, sequence: Sequence[Symbol]) -> float:
        """Compute the WFA score P(sequence) = ⟨α · A_{x_1} … A_{x_T}, ω⟩.

        Convenience wrapper combining :meth:`prefix_state` and
        :meth:`state_score`.
        """
        return self.state_score(self.prefix_state(sequence))


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
) -> list[list[float]]:
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
    return [
        [wfa.sequence_probability(tuple(prefix) + tuple(suffix)) for suffix in suffixes]
        for prefix in prefixes
    ]


def constrained_hankel_matrix(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    prefixes: Sequence[Sequence[Symbol]],
    suffixes: Sequence[Sequence[Symbol]],
) -> list[list[float]]:
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
    return rows


def projection_summary(
    original: Sequence[Sequence[float]],
    constrained: Sequence[Sequence[float]],
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
    _validate_same_matrix_shape(original, constrained)
    # Flatten both matrices for vectorised summation.
    original_values = [float(value) for row in original for value in row]
    constrained_values = [float(value) for row in constrained for value in row]
    original_mass = sum(original_values)
    constrained_mass = sum(constrained_values)
    return {
        "original_mass": original_mass,
        "constrained_mass": constrained_mass,
        "retained_mass": constrained_mass,  # alias for convenience
        "retained_fraction": constrained_mass / original_mass if original_mass else 0.0,
        "original_nonzero": float(sum(1 for value in original_values if value != 0.0)),
        "constrained_nonzero": float(sum(1 for value in constrained_values if value != 0.0)),
    }


def start_product_state(wfa: WeightedFiniteAutomaton, dfa: DFA) -> ProductDecoderState:
    """Create the initial WFA × DFA product state before any symbol is consumed.

    Both automata start at their respective initial states; the score is the
    WFA score of the empty prefix (i.e. ``⟨α, ω⟩``).
    """
    return ProductDecoderState(
        wfa_state=wfa.initial,
        dfa_state=dfa.start_state,
        score=wfa.state_score(wfa.initial),
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
    if symbol not in wfa.transitions:
        raise ValueError(f"unknown symbol {symbol!r}")
    # Check DFA first — if blocked, no need to advance the WFA.
    next_dfa_state = dfa.step(state.dfa_state, symbol)
    if next_dfa_state is None:
        return None
    next_wfa_state = _row_times_matrix(state.wfa_state, wfa.transitions[symbol])
    return ProductDecoderState(
        wfa_state=next_wfa_state,
        dfa_state=next_dfa_state,
        score=wfa.state_score(next_wfa_state),
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
