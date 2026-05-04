"""Spectral learning of Weighted Finite Automata (WFA) from Hankel matrices.

Implements the spectral / Hankel-SVD learning algorithm for signed WFAs:

1. Evaluate the target distribution over a finite prefix × suffix basis to
   build the Hankel matrix ``H`` (and per-symbol shifted matrices ``H_\u03c3``).
2. Compute a rank-*k* truncated SVD of ``H``.
3. Recover WFA parameters ``(α, {A_\u03c3}, \u03c9)`` from the factor matrices using the
   standard basis-change equations.

Provides:
- ``SpectralBasis``: immutable container for a prefix/suffix basis.
- ``SpectralLearningResult``: structured output of a learning run.
- ``build_spectral_basis``: convenience factory for length-bounded bases.
- ``spectral_learn_from_oracle``: learn from a callable ``P(sequence)``.
- ``spectral_learn_from_samples``: learn from a corpus of sample sequences.

Requires PyTorch for SVD computation (``torch.linalg.svd``).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from itertools import product

import torch

from .hankel import WeightedFiniteAutomaton

# Type aliases used throughout this module.
Symbol = Hashable
# A callable that maps a sequence of symbols to its probability/score.
SequenceProbability = Callable[[tuple[Symbol, ...]], float]


@dataclass(frozen=True)
class SpectralBasis:
    """An immutable prefix/suffix basis for Hankel matrix construction.

    The basis defines the rows (prefixes) and columns (suffixes) of the finite
    Hankel matrix used in spectral learning.  Both the empty sequence ``()``
    and all symbol sequences up to a chosen length are typically included.

    Attributes:
        prefixes: Ordered tuple of prefix sequences (row basis).
        suffixes: Ordered tuple of suffix sequences (column basis).
        symbols: Ordered alphabet shared by all sequences in this basis.
    """
    prefixes: tuple[tuple[Symbol, ...], ...]
    suffixes: tuple[tuple[Symbol, ...], ...]
    symbols: tuple[Symbol, ...]

    def __init__(
        self,
        prefixes: Sequence[Sequence[Symbol]],
        suffixes: Sequence[Sequence[Symbol]],
        symbols: Sequence[Symbol],
    ):
        """Construct and validate a :class:`SpectralBasis`.

        Args:
            prefixes: Row basis sequences; must be non-empty and use only
                symbols from *symbols*.
            suffixes: Column basis sequences; same constraints as *prefixes*.
            symbols: Ordered alphabet; must be non-empty and duplicate-free.

        Raises:
            ValueError: If any argument is empty, contains duplicates, or
                references unknown symbols.
        """
        symbols = tuple(symbols)
        _validate_symbols(symbols)
        prefixes = tuple(tuple(prefix) for prefix in prefixes)
        suffixes = tuple(tuple(suffix) for suffix in suffixes)
        if not prefixes:
            raise ValueError("prefixes must not be empty")
        if not suffixes:
            raise ValueError("suffixes must not be empty")
        _validate_sequences(prefixes, symbols, "prefix")
        _validate_sequences(suffixes, symbols, "suffix")
        object.__setattr__(self, "prefixes", prefixes)
        object.__setattr__(self, "suffixes", suffixes)
        object.__setattr__(self, "symbols", symbols)


@dataclass(frozen=True)
class SpectralLearningResult:
    """Structured output of a spectral WFA learning run.

    Attributes:
        model: The recovered :class:`~.hankel.WeightedFiniteAutomaton`.
        basis: The :class:`SpectralBasis` used to build the Hankel matrices.
        rank: The truncation rank *k* used in the SVD.
        singular_values: All singular values of the full Hankel matrix, in
            descending order.  Values beyond *rank* were discarded.
        diagnostics: Quality metrics computed after learning; see
            :func:`_diagnostics` for the full list of keys.
    """
    model: WeightedFiniteAutomaton
    basis: SpectralBasis
    rank: int
    singular_values: tuple[float, ...]
    diagnostics: dict[str, float]


def build_spectral_basis(
    symbols: Sequence[Symbol],
    max_prefix_len: int,
    max_suffix_len: int,
    *,
    include_empty: bool = True,
) -> SpectralBasis:
    """Build a simple length-bounded prefix/suffix basis.

    Enumerates all sequences of length 0 (if *include_empty* is ``True``) up
    to *max_prefix_len* for rows and up to *max_suffix_len* for columns.

    Args:
        symbols: Ordered alphabet.
        max_prefix_len: Maximum prefix length (inclusive, ≥ 0).
        max_suffix_len: Maximum suffix length (inclusive, ≥ 0).
        include_empty: When ``True`` (default), include the empty sequence
            ``()`` in both prefix and suffix sets.  This is required for
            spectral learning since the Hankel entry ``H[ε, ε] = P(ε)``
            is used to recover the initial and final vectors.

    Returns:
        A :class:`SpectralBasis` ready for use with the learning functions.

    Raises:
        ValueError: If *symbols* is invalid or either length bound is negative.
    """
    symbols = tuple(symbols)
    _validate_symbols(symbols)
    if max_prefix_len < 0:
        raise ValueError("max_prefix_len must be non-negative")
    if max_suffix_len < 0:
        raise ValueError("max_suffix_len must be non-negative")
    prefixes = _all_sequences_up_to(symbols, max_prefix_len, include_empty=include_empty)
    suffixes = _all_sequences_up_to(symbols, max_suffix_len, include_empty=include_empty)
    return SpectralBasis(prefixes=prefixes, suffixes=suffixes, symbols=symbols)


def spectral_learn_from_oracle(
    probability_fn: SequenceProbability,
    symbols: Sequence[Symbol],
    rank: int,
    *,
    basis: SpectralBasis | None = None,
    max_prefix_len: int = 2,
    max_suffix_len: int = 2,
    singular_tolerance: float = 1e-10,
) -> SpectralLearningResult:
    """Learn a signed WFA from a callable that returns P(sequence).

    Uses the exact probability oracle to fill the Hankel matrix, then applies
    the spectral / Hankel-SVD algorithm to recover a rank-*k* WFA.

    Args:
        probability_fn: A callable ``f(sequence) -> float`` that returns the
            target probability (or score) for any symbol sequence.
        symbols: Ordered alphabet.
        rank: Desired WFA state count (SVD truncation rank).
        basis: Optional pre-built :class:`SpectralBasis`.  If ``None``, one
            is built from *max_prefix_len* and *max_suffix_len*.
        max_prefix_len: Maximum prefix length when auto-building the basis.
        max_suffix_len: Maximum suffix length when auto-building the basis.
        singular_tolerance: Singular values below this threshold are
            considered numerically zero when checking the numerical rank.

    Returns:
        A :class:`SpectralLearningResult` containing the learned model and
        associated diagnostics.

    Raises:
        ValueError: If *rank* exceeds the matrix dimensions or numerical rank,
            or if any basis/symbol constraints are violated.
    """
    symbols = tuple(symbols)
    _validate_symbols(symbols)
    if basis is None:
        basis = build_spectral_basis(symbols, max_prefix_len, max_suffix_len)
    _validate_basis_for_learning(basis, symbols)

    # Wrap the oracle so it validates its inputs before delegating.
    def probability(sequence: tuple[Symbol, ...]) -> float:
        _validate_sequences((sequence,), symbols, "sequence")
        return float(probability_fn(sequence))

    return _spectral_learn(probability, basis, rank, singular_tolerance)


def spectral_learn_from_samples(
    sequences: Sequence[Sequence[Symbol]],
    symbols: Sequence[Symbol],
    rank: int,
    *,
    basis: SpectralBasis | None = None,
    max_prefix_len: int = 2,
    max_suffix_len: int = 2,
    smoothing: float = 0.0,
    singular_tolerance: float = 1e-10,
) -> SpectralLearningResult:
    """Estimate string probabilities from samples, then learn a signed WFA.

    Builds an empirical probability estimate from *sequences* (optionally with
    additive smoothing over all sequences in the basis support), then delegates
    to :func:`spectral_learn_from_oracle` using that estimate as the oracle.

    Args:
        sequences: Training corpus; a collection of symbol sequences.
        symbols: Ordered alphabet.
        rank: Desired WFA state count (SVD truncation rank).
        basis: Optional pre-built :class:`SpectralBasis`.
        max_prefix_len: Maximum prefix length when auto-building the basis.
        max_suffix_len: Maximum suffix length when auto-building the basis.
        smoothing: Additive (Laplace-style) smoothing count added to every
            sequence in the basis support.  ``0.0`` uses raw empirical counts.
        singular_tolerance: Passed through to the SVD rank check.

    Returns:
        A :class:`SpectralLearningResult` containing the learned model and
        diagnostics.

    Raises:
        ValueError: If *sequences* is empty, *smoothing* is negative, or any
            symbol/rank constraint is violated.
    """
    symbols = tuple(symbols)
    _validate_symbols(symbols)
    if not sequences:
        raise ValueError("sequences must not be empty")
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")

    encoded = tuple(tuple(sequence) for sequence in sequences)
    _validate_sequences(encoded, symbols, "sequence")
    if basis is None:
        basis = build_spectral_basis(symbols, max_prefix_len, max_suffix_len)
    _validate_basis_for_learning(basis, symbols)

    # Determine the smoothing denominator based on the full basis support.
    support = _query_support(basis)
    counts = Counter(encoded)
    denominator = float(len(encoded)) + float(smoothing) * len(support)

    def probability(sequence: tuple[Symbol, ...]) -> float:
        _validate_sequences((sequence,), symbols, "sequence")
        if smoothing:
            # Laplace-smoothed estimate: (count + smoothing) / (N + smoothing * |support|)
            return (float(counts.get(sequence, 0)) + float(smoothing)) / denominator
        # Unsmoothed empirical frequency.
        return float(counts.get(sequence, 0)) / float(len(encoded))

    return _spectral_learn(probability, basis, rank, singular_tolerance)


def _spectral_learn(
    probability: SequenceProbability,
    basis: SpectralBasis,
    rank: int,
    singular_tolerance: float,
) -> SpectralLearningResult:
    """Core Hankel-SVD spectral learning algorithm.

    Steps:
    1. Build the ``|prefixes| x |suffixes|`` Hankel matrix ``H`` and the
       per-symbol shifted matrices ``H_\u03c3``.
    2. Compute the full SVD of ``H``; truncate to rank *k*.
    3. Recover WFA parameters using the basis-change equations:

       - ``\u03b1  = h[\u03b5, :] · V_k · \u03a3_k^{-1/2}``
       - ``\u03c9  = \u03a3_k^{-1/2} · U_k\u1d40 · h[:, \u03b5]``
       - ``A_\u03c3 = \u03a3_k^{-1/2} · U_k\u1d40 · H_\u03c3 · V_k · \u03a3_k^{-1/2}``

    Args:
        probability: Validated probability oracle.
        basis: :class:`SpectralBasis` with the empty sequence in both sets.
        rank: SVD truncation rank.
        singular_tolerance: Threshold below which singular values are
            considered numerically zero.

    Returns:
        A fully populated :class:`SpectralLearningResult`.

    Raises:
        ValueError: If *rank* is out of range.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if singular_tolerance < 0.0:
        raise ValueError("singular_tolerance must be non-negative")

    # --- Step 1: Build the main Hankel matrix ---
    h = _hankel_tensor(probability, basis.prefixes, basis.suffixes)
    max_rank = min(h.shape)
    if rank > max_rank:
        raise ValueError(f"rank cannot exceed min Hankel dimension {max_rank}")

    # --- Step 2: Truncated SVD ---
    u, singular_values, vh = torch.linalg.svd(h, full_matrices=False)
    numerical_rank = int(torch.sum(singular_values > float(singular_tolerance)).item())
    if rank > numerical_rank:
        raise ValueError(f"rank cannot exceed numerical Hankel rank {numerical_rank}")

    # Truncate to rank k and compute the inverse square-root scale matrix.
    u_r = u[:, :rank]          # left singular vectors  (|prefixes| x k)
    values_r = singular_values[:rank]  # top-k singular values
    v_r = vh[:rank, :].transpose(0, 1)  # right singular vectors (|suffixes| x k)
    inv_sqrt = torch.diag(torch.rsqrt(values_r))  # diagonal \u03a3_k^{-1/2}

    # --- Step 3: Extract WFA parameters ---
    # Locate the empty-sequence rows/columns in the basis.
    epsilon_prefix_idx = basis.prefixes.index(())
    epsilon_suffix_idx = basis.suffixes.index(())
    h_epsilon_s = h[epsilon_prefix_idx, :]   # row vector for \u03b5 prefix
    h_p_epsilon = h[:, epsilon_suffix_idx]   # column vector for \u03b5 suffix

    # Recover initial (α) and final (ω) vectors.
    initial = h_epsilon_s @ v_r @ inv_sqrt
    final = inv_sqrt @ u_r.transpose(0, 1) @ h_p_epsilon

    # Recover one transition matrix per symbol from the shifted Hankel matrices.
    transitions = {}
    for symbol in basis.symbols:
        h_symbol = _hankel_tensor(
            probability,
            basis.prefixes,
            basis.suffixes,
            middle=(symbol,),
        )
        transition = inv_sqrt @ u_r.transpose(0, 1) @ h_symbol @ v_r @ inv_sqrt
        transitions[symbol] = _matrix_to_lists(transition)

    model = WeightedFiniteAutomaton(
        initial=_vector_to_list(initial),
        transitions=transitions,
        final=_vector_to_list(final),
        symbols=basis.symbols,
    )
    diagnostics = _diagnostics(model, h, singular_values, rank, basis)
    return SpectralLearningResult(
        model=model,
        basis=basis,
        rank=rank,
        singular_values=tuple(float(value) for value in singular_values.tolist()),
        diagnostics=diagnostics,
    )


def _diagnostics(
    model: WeightedFiniteAutomaton,
    original_hankel: torch.Tensor,
    singular_values: torch.Tensor,
    rank: int,
    basis: SpectralBasis,
) -> dict[str, float]:
    """Compute post-learning quality metrics for a recovered WFA.

    Args:
        model: The learned :class:`~.hankel.WeightedFiniteAutomaton`.
        original_hankel: The ``|prefixes| x |suffixes|`` Hankel tensor used
            to fit the model.
        singular_values: Full singular value vector from ``torch.linalg.svd``.
        rank: Truncation rank used during learning.
        basis: The :class:`SpectralBasis` (used to reconstruct the Hankel
            matrix from the model for error computation).

    Returns:
        Dictionary with keys:

        - ``negative_score_count``: number of basis sequences with score < 0.
        - ``min_score`` / ``max_score``: range of scores over the basis.
        - ``reconstruction_error``: Frobenius norm of ``H - H_model``.
        - ``relative_reconstruction_error``: ``reconstruction_error / ||H||``.
        - ``retained_singular_mass``: sum of the top-*k* singular values.
        - ``retained_singular_fraction``: fraction of total singular mass
          retained by the rank-*k* approximation.
    """
    # Reconstruct the Hankel matrix from the learned model for error analysis.
    reconstructed = torch.tensor(
        [
            [model.sequence_probability(prefix + suffix) for suffix in basis.suffixes]
            for prefix in basis.prefixes
        ],
        dtype=torch.float64,
    )
    diff = original_hankel - reconstructed
    original_norm = float(torch.linalg.norm(original_hankel).item())
    reconstruction_error = float(torch.linalg.norm(diff).item())
    # Evaluate the model on every sequence in the basis support.
    scores = [model.sequence_probability(sequence) for sequence in sorted(_query_support(basis), key=repr)]
    singular_total = float(torch.sum(singular_values).item())
    retained = float(torch.sum(singular_values[:rank]).item())
    return {
        "negative_score_count": float(sum(1 for score in scores if score < 0.0)),
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "reconstruction_error": reconstruction_error,
        "relative_reconstruction_error": reconstruction_error / original_norm if original_norm else 0.0,
        "retained_singular_mass": retained,
        "retained_singular_fraction": retained / singular_total if singular_total else 0.0,
    }


def _hankel_tensor(
    probability: SequenceProbability,
    prefixes: Sequence[tuple[Symbol, ...]],
    suffixes: Sequence[tuple[Symbol, ...]],
    *,
    middle: tuple[Symbol, ...] = (),
) -> torch.Tensor:
    """Build a ``float64`` Hankel tensor of shape ``(|prefixes|, |suffixes|)``.

    Each cell ``[i, j]`` holds ``probability(prefixes[i] + middle + suffixes[j])``.
    When *middle* is empty this gives the standard Hankel matrix ``H``;
    when *middle* is a single symbol ``(\u03c3,)`` this gives the shifted matrix
    ``H_\u03c3`` used to recover the transition matrix for symbol *\u03c3*.

    Args:
        probability: Callable probability oracle.
        prefixes: Row basis sequences.
        suffixes: Column basis sequences.
        middle: Optional symbol tuple inserted between prefix and suffix.

    Returns:
        A 2-D ``torch.Tensor`` of dtype ``float64``.
    """
    return torch.tensor(
        [
            [float(probability(prefix + middle + suffix)) for suffix in suffixes]
            for prefix in prefixes
        ],
        dtype=torch.float64,
    )


def _query_support(basis: SpectralBasis) -> set[tuple[Symbol, ...]]:
    """Return the set of all sequences queried when building the Hankel matrices.

    Includes ``prefix + suffix`` (main Hankel) and ``prefix + (\u03c3,) + suffix``
    (shifted Hankel for every symbol) for all prefix/suffix pairs, as well as
    bare prefixes and suffixes.  Used to enumerate the smoothing domain and
    to compute diagnostic scores.
    """
    support = set()
    for prefix in basis.prefixes:
        for suffix in basis.suffixes:
            support.add(prefix + suffix)  # main Hankel entry
            for symbol in basis.symbols:
                support.add(prefix + (symbol,) + suffix)  # shifted Hankel entry
    # Also include standalone prefixes and suffixes.
    for suffix in basis.suffixes:
        support.add(suffix)
    for prefix in basis.prefixes:
        support.add(prefix)
    return support


def _validate_basis_for_learning(basis: SpectralBasis, symbols: tuple[Symbol, ...]) -> None:
    """Raise if *basis* is incompatible with spectral learning requirements.

    Spectral learning requires both the empty prefix and empty suffix to be
    present so that the initial and final vectors can be extracted from the
    ``H[\u03b5, :]`` row and ``H[:, \u03b5]`` column respectively.
    """
    if not isinstance(basis, SpectralBasis):
        raise TypeError("basis must be a SpectralBasis")
    if basis.symbols != symbols:
        raise ValueError("basis symbols must match symbols")
    if () not in basis.prefixes:
        raise ValueError("basis prefixes must include the empty sequence")
    if () not in basis.suffixes:
        raise ValueError("basis suffixes must include the empty sequence")


def _validate_symbols(symbols: tuple[Symbol, ...]) -> None:
    """Raise ``ValueError`` if *symbols* is empty or contains duplicates."""
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")


def _validate_sequences(sequences: Sequence[tuple[Symbol, ...]], symbols: tuple[Symbol, ...], label: str) -> None:
    """Raise ``ValueError`` if any sequence contains a symbol outside *symbols*.

    Args:
        sequences: Sequences to check.
        symbols: Valid alphabet.
        label: Human-readable name for the sequence role (e.g. ``"prefix"``);
            used in error messages.
    """
    symbol_set = set(symbols)
    for seq_idx, sequence in enumerate(sequences):
        for symbol in sequence:
            if symbol not in symbol_set:
                raise ValueError(f"unknown symbol {symbol!r} in {label} {seq_idx}")


def _all_sequences_up_to(
    symbols: tuple[Symbol, ...],
    max_len: int,
    *,
    include_empty: bool,
) -> tuple[tuple[Symbol, ...], ...]:
    """Enumerate all symbol sequences of length 0 (optional) through *max_len*.

    Args:
        symbols: The alphabet to draw from.
        max_len: Maximum sequence length (inclusive).
        include_empty: When ``True``, prepend the empty tuple ``()``.

    Returns:
        An immutable tuple of tuples, ordered by increasing length then by
        ``itertools.product`` order within each length.
    """
    output = [()] if include_empty else []
    for length in range(1, max_len + 1):
        output.extend(tuple(sequence) for sequence in product(symbols, repeat=length))
    return tuple(output)


def _vector_to_list(vector: torch.Tensor) -> list[float]:
    """Convert a 1-D ``torch.Tensor`` to a plain Python ``list[float]``."""
    return [float(value) for value in vector.tolist()]


def _matrix_to_lists(matrix: torch.Tensor) -> list[list[float]]:
    """Convert a 2-D ``torch.Tensor`` to a ``list[list[float]]``."""
    return [[float(value) for value in row] for row in matrix.tolist()]
