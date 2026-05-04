"""Hidden Markov Model (HMM) training and inference utilities.

Provides:
- ``ProbabilisticAutomaton``: a discrete HMM that can score observation
  sequences and be extracted into a deterministic finite automaton (DFA).
- ``HMMParameters`` / ``BaumWelchResult``: lightweight data containers.
- ``baum_welch_train``: dependency-free Baum-Welch EM training with scaled
  forward-backward to avoid floating-point underflow.
- ``compare_hmm_dfa``: evaluation helper that compares HMM acceptance against
  a reference DFA over a corpus of sequences.
- ``all_sequences``: enumerates all symbol sequences up to a given length.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Iterable, Sequence

from .dfa import DFA


@dataclass(frozen=True)
class ProbabilisticAutomaton:
    """An immutable discrete Hidden Markov Model (HMM).

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
    model: ProbabilisticAutomaton
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
    init: ProbabilisticAutomaton | HMMParameters | None = None,
    random_seed: int = 0,
) -> BaumWelchResult:
    """Train a discrete HMM with Baum-Welch expectation maximization.

    This is intentionally dependency-free and aimed at small research/test
    examples. It uses scaled forward-backward to avoid probability underflow.
    """

    # Validate inputs and encode symbol strings to integer indices for speed.
    encoded_sequences, symbols = _validate_and_encode_sequences(sequences, symbols, state_count)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    if init is None:
        # Randomly initialise parameters when no warm-start is provided.
        initial = _random_stochastic_vector(state_count, Random(random_seed))
        rng = Random(random_seed + 1)
        transition = _random_stochastic_matrix(state_count, state_count, rng)
        emission = _random_stochastic_matrix(state_count, len(symbols), rng)
    else:
        # Coerce and normalise the provided warm-start parameters.
        params = _coerce_init(init, symbols, state_count)
        transition = [list(row) for row in params.transition]
        emission = [list(row) for row in params.emission]
        initial = list(params.initial)

    log_likelihoods: list[float] = []
    converged = False

    for iteration in range(max_iter):
        # --- E-step accumulators ---
        init_counts = [0.0 for _ in range(state_count)]
        trans_counts = [[0.0 for _ in range(state_count)] for _ in range(state_count)]
        emit_counts = [[0.0 for _ in range(len(symbols))] for _ in range(state_count)]
        total_log_likelihood = 0.0

        for obs in encoded_sequences:
            # Run the scaled forward-backward algorithm for this sequence.
            alpha, scales, log_likelihood = _forward_scaled(obs, initial, transition, emission)
            beta = _backward_scaled(obs, transition, emission, scales)
            # Compute state-occupation (gamma) and transition (xi) posteriors.
            gamma, xi = _expectations(obs, alpha, beta, transition, emission)
            total_log_likelihood += log_likelihood

            # Accumulate expected counts for the M-step.
            for state in range(state_count):
                init_counts[state] += gamma[0][state]  # initial state usage
            for t, symbol_idx in enumerate(obs):
                for state in range(state_count):
                    emit_counts[state][symbol_idx] += gamma[t][state]  # emission usage
            for t in range(len(obs) - 1):
                for src in range(state_count):
                    for dst in range(state_count):
                        trans_counts[src][dst] += xi[t][src][dst]  # transition usage

        # --- M-step: re-estimate parameters from accumulated counts ---
        initial = _normalize_row(init_counts, smoothing)
        transition = [_normalize_row(row, smoothing) for row in trans_counts]
        emission = [_normalize_row(row, smoothing) for row in emit_counts]
        log_likelihoods.append(total_log_likelihood)

        # Check convergence: stop if the log-likelihood gain is negligible.
        if len(log_likelihoods) > 1 and log_likelihoods[-1] - log_likelihoods[-2] < tol:
            converged = True
            break

    model = ProbabilisticAutomaton(
        transition=transition,
        emission=emission,
        initial=initial,
        symbols=symbols,
    )
    return BaumWelchResult(
        model=model,
        log_likelihoods=tuple(log_likelihoods),
        iterations=iteration + 1,
        converged=converged,
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
    init: ProbabilisticAutomaton | HMMParameters,
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
    if isinstance(init, ProbabilisticAutomaton):
        if init.symbols != symbols:
            raise ValueError("init symbols must match symbols")
        params = HMMParameters(init.transition, init.emission, init.initial)
    elif isinstance(init, HMMParameters):
        params = init
    else:
        raise TypeError("init must be ProbabilisticAutomaton, HMMParameters, or None")

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
