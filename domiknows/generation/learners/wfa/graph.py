"""Spectral-learning utilities for graph-constrained sequence models.

This module implements a compact weighted automaton learned from finite Hankel
blocks, with optional graph and DFA constraints that zero invalid queries.

It also includes helper utilities used by HMM/Spectral initialization and
sequence-validity checks under transition/emission masks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from ..hmm.constraints import combine_masks, project_matrix_rows, validate_mask
from ..hmm.dynamic import DynamicConstraintContext, transition_energy_matrix


@dataclass
class GraphSpectralFitResult:
    """Diagnostics from graph-constrained Hankel/SVD learning."""

    rank: int
    singular_values: torch.Tensor
    hankel: torch.Tensor
    constrained_query_count: int
    total_query_count: int
    diagnostics: dict[str, float] = field(default_factory=dict)


class GraphSpectralAutomaton:
    """Signed WFA learned from graph-constrained finite Hankel blocks.

    The model scores a sequence as ``initial @ M_a @ ... @ final``. Graph
    constraints affect learning by zeroing Hankel entries whose prefix/suffix
    concatenation has no legal hidden path under supplied HMM-style masks, or
    is rejected by an optional DFA.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        graph_adapter=None,
        transition_mask=None,
        emission_mask=None,
        dfa=None,
        smoothing: float = 0.0,
        device=None,
        dtype: torch.dtype = torch.float64,
        operator_transform: Callable[[DynamicConstraintContext, Any, torch.Tensor], Any] | None = None,
        operator_energy: Callable[[DynamicConstraintContext, Any], Any] | None = None,
        energy_weight: float = 1.0,
        dynamic_metadata: Mapping[str, Any] | None = None,
    ):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if energy_weight < 0:
            raise ValueError("energy_weight must be non-negative")
        self.symbols = tuple(symbols) if symbols is not None else None
        self.symbol_to_id: dict[Any, int] = {}
        self.id_to_symbol: tuple[Any, ...] = ()
        self.graph_adapter = graph_adapter
        self._explicit_transition_mask = transition_mask
        self._explicit_emission_mask = emission_mask
        self.dfa = dfa
        self.smoothing = smoothing
        self.device = device
        self.dtype = dtype
        self.operator_transform = operator_transform
        self.operator_energy = operator_energy
        self.energy_weight = energy_weight
        self.dynamic_metadata = dict(dynamic_metadata or {})

        self.initial: torch.Tensor | None = None
        self.final: torch.Tensor | None = None
        self.operators: dict[Any, torch.Tensor] = {}
        self.transition_mask_: torch.Tensor | None = None
        self.emission_mask_: torch.Tensor | None = None
        self.prefixes: tuple[tuple[Any, ...], ...] = ()
        self.suffixes: tuple[tuple[Any, ...], ...] = ()
        self.fit_result_: GraphSpectralFitResult | None = None
        self._encoded_counts: Counter[tuple[int, ...]] = Counter()
        self._sample_total = 0

    def fit(
        self,
        sequences: Sequence[Sequence[Any]],
        prefixes: Sequence[Sequence[Any]],
        suffixes: Sequence[Sequence[Any]],
        *,
        rank: int,
        graph_adapter=None,
        dfa=None,
    ) -> "GraphSpectralAutomaton":
        """Fit a finite-rank signed WFA from constrained Hankel blocks."""
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if not sequences:
            raise ValueError("sequences must not be empty")
        self.graph_adapter = graph_adapter or self.graph_adapter
        self.dfa = dfa or self.dfa

        self.prefixes = _normalize_basis(prefixes, name="prefixes")
        self.suffixes = _normalize_basis(suffixes, name="suffixes")
        if () not in self.prefixes:
            raise ValueError("prefixes must include the empty prefix ()")
        if () not in self.suffixes:
            raise ValueError("suffixes must include the empty suffix ()")

        all_sequences = [tuple(sequence) for sequence in sequences]
        self._build_symbol_table(all_sequences, self.prefixes, self.suffixes)
        encoded_sequences = [self._encode_sequence(sequence) for sequence in all_sequences]
        self._build_masks()

        self._encoded_counts = Counter(encoded_sequences)
        self._sample_total = sum(self._encoded_counts.values())
        probability = self._empirical_probability_oracle(encoded_sequences)
        # Base Hankel block H(.,.) used for truncated SVD factorization.
        hankel, valid_queries, total_queries = self._build_hankel(probability, middle=())
        max_rank = min(hankel.shape)
        if rank > max_rank:
            raise ValueError(f"rank cannot exceed min Hankel dimension {max_rank}")

        U, singular_values, Vh = torch.linalg.svd(hankel, full_matrices=False)
        numerical_rank = int((singular_values > torch.finfo(self.dtype).eps).sum().item())
        if numerical_rank == 0:
            raise ValueError("constrained Hankel matrix has numerical rank 0")
        if rank > numerical_rank:
            raise ValueError(f"rank cannot exceed numerical Hankel rank {numerical_rank}")

        U_r = U[:, :rank]
        S_r = singular_values[:rank]
        V_r = Vh[:rank, :].T
        inv_sqrt = torch.diag(torch.rsqrt(S_r.clamp_min(torch.finfo(self.dtype).tiny)))

        epsilon_prefix_index = self.prefixes.index(())
        epsilon_suffix_index = self.suffixes.index(())
        h_epsilon_s = hankel[epsilon_prefix_index, :]
        h_p_epsilon = hankel[:, epsilon_suffix_index]

        # Standard finite-rank WFA recovery from truncated Hankel factors.
        self.initial = h_epsilon_s @ V_r @ inv_sqrt
        self.final = inv_sqrt @ U_r.T @ h_p_epsilon
        self.operators = {}
        for symbol in self.id_to_symbol:
            # Shifted Hankel block H(., symbol, .) gives the symbol operator.
            shifted, _, _ = self._build_hankel(probability, middle=(symbol,))
            self.operators[symbol] = inv_sqrt @ U_r.T @ shifted @ V_r @ inv_sqrt

        reconstructed = self.reconstruct_hankel(dynamic=False)
        error = torch.linalg.norm(hankel - reconstructed).item()
        retained_mass = float(S_r.sum().item() / singular_values.sum().clamp_min(torch.finfo(self.dtype).tiny).item())
        self.fit_result_ = GraphSpectralFitResult(
            rank=rank,
            singular_values=singular_values,
            hankel=hankel,
            constrained_query_count=valid_queries,
            total_query_count=total_queries,
            diagnostics={
                "reconstruction_error": float(error),
                "retained_singular_mass": retained_mass,
                "zeroed_query_count": float(total_queries - valid_queries),
            },
        )
        return self

    def score(self, sequence: Sequence[Any], *, enforce_constraints: bool = False) -> float:
        """Return scalar sequence score under the fitted signed WFA.

        Spectral fitting uses graph/DFA legality to filter Hankel queries, but
        the low-rank signed WFA reconstruction itself is not a hard constraint
        mechanism.  Set ``enforce_constraints=True`` when the score is being
        used as a constrained inference value; invalid completed strings then
        receive score ``0.0`` instead of the unconstrained signed WFA score.
        """
        self._require_fitted()
        if enforce_constraints and not self.is_sequence_allowed(sequence):
            return 0.0
        state = self.prefix_state(sequence)
        return float((state @ self.final).item())

    def hard_score(self, sequence: Sequence[Any]) -> float:
        """Return score with graph/DFA legality filtering enabled."""

        return self.score(sequence, enforce_constraints=True)

    def prefix_state(self, sequence: Sequence[Any]) -> torch.Tensor:
        """Traverse operators for a prefix and return the resulting state row."""
        self._require_fitted()
        state = self.initial.clone()
        prefix: list[Any] = []
        full_sequence = tuple(sequence)
        for step, symbol in enumerate(sequence):
            if symbol not in self.operators:
                raise ValueError(f"unknown symbol {symbol!r}")
            context = self._context(step=step, prefix=tuple(prefix), belief=state, sequence=full_sequence)
            state = state @ self.operator_for_context(symbol, context)
            prefix.append(symbol)
        return state

    def operator(self, symbol: Any) -> torch.Tensor:
        """Return the static learned operator for one symbol."""
        self._require_fitted()
        if symbol not in self.operators:
            raise ValueError(f"unknown symbol {symbol!r}")
        return self.operators[symbol]

    def operator_for_context(self, symbol: Any, context: DynamicConstraintContext) -> torch.Tensor:
        """Return the effective operator for *symbol* under *context*."""

        base = self.operator(symbol)
        operator = base
        if self.operator_transform is not None:
            # Optional hard transform hook (context-aware operator override).
            transformed = self.operator_transform(context, symbol, base)
            if transformed is not None:
                operator = self._validate_operator(transformed, name="operator_transform")
        if self.operator_energy is not None:
            # Optional soft penalty hook: multiply by exp(-weight * energy).
            energy = self.operator_energy(context, symbol)
            if energy is not None:
                penalty = transition_energy_matrix(
                    energy,
                    shape=tuple(operator.shape),
                    dtype=operator.dtype,
                    device=operator.device,
                )
                operator = operator * torch.exp(-self.energy_weight * penalty)
        return self._validate_operator(operator, name="dynamic operator")

    def allowed_symbols(self, state_or_prefix: torch.Tensor | Sequence[Any] | None = None) -> tuple[Any, ...]:
        """Return symbols allowed globally or from a specific prefix."""
        if state_or_prefix is None or isinstance(state_or_prefix, torch.Tensor):
            return self.id_to_symbol
        prefix = tuple(state_or_prefix)
        allowed = []
        for symbol in self.id_to_symbol:
            if self._sequence_allowed(prefix + (symbol,)):
                allowed.append(symbol)
        return tuple(allowed)

    def is_sequence_allowed(self, sequence: Sequence[Any]) -> bool:
        """Return whether a completed string satisfies active graph/DFA filters."""

        self._require_fitted()
        raw = tuple(sequence)
        encoded = self._encode_sequence(raw)
        return self._encoded_sequence_allowed(encoded, raw)

    def build_hankel(self, middle: Sequence[Any] = ()) -> torch.Tensor:
        """Build empirical constrained Hankel block for an optional middle token."""
        self._require_basis()
        probability = self._empirical_probability_oracle()
        return self._build_hankel(probability, middle=tuple(middle))[0]

    def reconstruct_hankel(self, *, dynamic: bool = False) -> torch.Tensor:
        """Reconstruct the fitted Hankel block using static or dynamic traversal."""

        self._require_basis()
        self._require_fitted()
        reconstructed = torch.zeros((len(self.prefixes), len(self.suffixes)), dtype=self.dtype, device=self.device)
        for i, prefix in enumerate(self.prefixes):
            prefix_state = self.prefix_state(prefix) if dynamic else self._static_prefix_state(prefix)
            for j, suffix in enumerate(self.suffixes):
                if dynamic:
                    # Dynamic mode recomputes full traversal with context hooks.
                    state = self.prefix_state(prefix + suffix)
                else:
                    state = prefix_state.clone()
                    for symbol in suffix:
                        state = state @ self.operator(symbol)
                reconstructed[i, j] = state @ self.final
        return reconstructed

    def to_torch_learner(self, *, trainable: bool = True, pad_size: int = 4, label_to_token_id=None, random_seed: int | None = None):
        """Return a PMD-compatible Torch head initialized from this fitted WFA."""
        from .graph_head import GraphSpectralGenerationHead

        return GraphSpectralGenerationHead.from_graph_spectral(
            self,
            trainable=trainable,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            random_seed=random_seed,
        )

    def _build_symbol_table(
        self,
        sequences: Sequence[tuple[Any, ...]],
        prefixes: Sequence[tuple[Any, ...]],
        suffixes: Sequence[tuple[Any, ...]],
    ) -> None:
        """Infer symbol vocabulary from data and basis (unless explicitly provided)."""
        symbols = list(self.symbols) if self.symbols is not None else []
        seen = set(symbols)
        for sequence in list(sequences) + list(prefixes) + list(suffixes):
            for symbol in sequence:
                if symbol not in seen:
                    symbols.append(symbol)
                    seen.add(symbol)
        if not symbols:
            raise ValueError("symbols must not be empty")
        self.id_to_symbol = tuple(symbols)
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(self.id_to_symbol)}
        if self.graph_adapter is not None:
            self.graph_adapter.set_symbols(self.id_to_symbol)

    def _encode_sequence(self, sequence: Sequence[Any]) -> tuple[int, ...]:
        """Map a symbol sequence to integer ids using the learned table."""
        encoded = []
        for symbol in sequence:
            if symbol not in self.symbol_to_id:
                raise ValueError(f"unknown symbol {symbol!r}")
            encoded.append(self.symbol_to_id[symbol])
        return tuple(encoded)

    def _build_masks(self) -> None:
        """Combine adapter/external masks for legality filtering of Hankel queries."""
        emission_shape = None
        graph_transition = None
        graph_emission = None
        if self.graph_adapter is not None:
            graph_transition = self.graph_adapter.allowed_transition_mask()
            graph_emission = self.graph_adapter.emission_type_mask()
            emission_shape = tuple(graph_emission.shape)
        explicit_transition = None
        if self._explicit_transition_mask is not None:
            explicit_transition = torch.as_tensor(self._explicit_transition_mask, dtype=self.dtype, device=self.device)
        if self._explicit_emission_mask is not None:
            explicit_emission = torch.as_tensor(self._explicit_emission_mask, dtype=self.dtype, device=self.device)
            emission_shape = tuple(explicit_emission.shape)
        elif emission_shape is None and explicit_transition is not None:
            emission_shape = (explicit_transition.shape[0], len(self.id_to_symbol))
        elif emission_shape is None:
            # No masks configured: all strings are legal unless DFA rejects them.
            self.transition_mask_ = None
            self.emission_mask_ = None
            return

        state_count, symbol_count = emission_shape
        if symbol_count != len(self.id_to_symbol):
            raise ValueError(f"emission mask symbol dimension {symbol_count} does not match {len(self.id_to_symbol)} symbols")
        transition_shape = (state_count, state_count)
        self.transition_mask_ = combine_masks(
            (graph_transition, self._explicit_transition_mask),
            transition_shape,
            name="transition_mask",
            device=self.device,
            dtype=self.dtype,
        )
        self.emission_mask_ = combine_masks(
            (graph_emission, self._explicit_emission_mask),
            emission_shape,
            name="emission_mask",
            device=self.device,
            dtype=self.dtype,
        )

    def _empirical_probability_oracle(self, encoded_sequences: list[tuple[int, ...]] | None = None):
        """Return a callable empirical probability oracle over encoded strings."""
        counts = Counter(encoded_sequences) if encoded_sequences is not None else self._encoded_counts
        total = sum(counts.values()) if encoded_sequences is not None else self._sample_total

        def probability(encoded: tuple[int, ...]) -> torch.Tensor:
            if total == 0:
                return torch.tensor(0.0, dtype=self.dtype, device=self.device)
            count = counts.get(encoded, 0)
            if count == 0 and self.smoothing > 0:
                # Optional additive smoothing for unseen strings.
                return torch.tensor(self.smoothing / (total + self.smoothing), dtype=self.dtype, device=self.device)
            return torch.tensor(count / total, dtype=self.dtype, device=self.device)

        return probability

    def _build_hankel(self, probability, *, middle: tuple[Any, ...]) -> tuple[torch.Tensor, int, int]:
        """Assemble a Hankel block and track valid vs total constrained queries."""
        middle_encoded = tuple(self.symbol_to_id[symbol] for symbol in middle)
        hankel = torch.zeros((len(self.prefixes), len(self.suffixes)), dtype=self.dtype, device=self.device)
        valid_queries = 0
        total_queries = 0
        for i, prefix in enumerate(self.prefixes):
            prefix_encoded = tuple(self.symbol_to_id[symbol] for symbol in prefix)
            for j, suffix in enumerate(self.suffixes):
                suffix_encoded = tuple(self.symbol_to_id[symbol] for symbol in suffix)
                encoded = prefix_encoded + middle_encoded + suffix_encoded
                raw = prefix + middle + suffix
                total_queries += 1
                if self._encoded_sequence_allowed(encoded, raw):
                    valid_queries += 1
                    hankel[i, j] = probability(encoded)
        return hankel, valid_queries, total_queries

    def _sequence_allowed(self, sequence: tuple[Any, ...]) -> bool:
        """Check legality for a raw symbol sequence under DFA/masks."""
        encoded = tuple(self.symbol_to_id[symbol] for symbol in sequence)
        return self._encoded_sequence_allowed(encoded, sequence)

    def _encoded_sequence_allowed(self, encoded: tuple[int, ...], raw: tuple[Any, ...]) -> bool:
        """Check legality for an encoded sequence under active constraints."""
        if self.dfa is not None and not self.dfa.accepts(raw):
            return False
        if self.transition_mask_ is None or self.emission_mask_ is None:
            return True
        return sequence_has_legal_path(encoded, self.transition_mask_, self.emission_mask_)

    def _static_prefix_state(self, sequence: Sequence[Any]) -> torch.Tensor:
        """Traverse prefix with static operators only (no dynamic hooks)."""
        state = self.initial.clone()
        for symbol in sequence:
            state = state @ self.operator(symbol)
        return state

    def _context(
        self,
        *,
        step: int,
        prefix: tuple[Any, ...],
        belief: torch.Tensor,
        sequence: tuple[Any, ...] | None,
    ) -> DynamicConstraintContext:
        """Build dynamic hook context with defensive copies and metadata."""
        return DynamicConstraintContext(
            step=step,
            prefix=prefix,
            belief=belief.detach().clone(),
            sequence=sequence,
            metadata={
                "symbols": self.id_to_symbol,
                "rank": None if self.initial is None else int(self.initial.numel()),
                "graph_adapter": self.graph_adapter,
                **self.dynamic_metadata,
            },
        )

    def _validate_operator(self, operator, *, name: str) -> torch.Tensor:
        """Validate operator shape/finiteness for robust dynamic overrides."""
        tensor = torch.as_tensor(operator, dtype=self.dtype, device=self.device)
        rank = None if self.initial is None else int(self.initial.numel())
        expected = None if rank is None else (rank, rank)
        if expected is not None and tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 matrix")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    def _require_fitted(self) -> None:
        if self.initial is None or self.final is None or not self.operators:
            raise RuntimeError("GraphSpectralAutomaton must be fit before this operation")

    def _require_basis(self) -> None:
        if not self.prefixes or not self.suffixes:
            raise RuntimeError("GraphSpectralAutomaton must be fit before building Hankel blocks")


def sequence_has_legal_path(
    encoded_sequence: Sequence[int],
    transition_mask: torch.Tensor,
    emission_mask: torch.Tensor,
) -> bool:
    """Return true when some hidden path can emit *encoded_sequence*."""

    if not encoded_sequence:
        return True
    transition = (transition_mask > 0).to(dtype=torch.bool)
    emission = (emission_mask > 0).to(dtype=torch.bool)
    if max(encoded_sequence) >= emission.shape[1] or min(encoded_sequence) < 0:
        return False
    possible = emission[:, encoded_sequence[0]]
    for symbol_id in encoded_sequence[1:]:
        # Propagate reachable states through allowed transitions.
        next_possible = (possible.to(dtype=torch.float64) @ transition.to(dtype=torch.float64)) > 0
        possible = next_possible & emission[:, symbol_id]
        if not possible.any():
            return False
    return bool(possible.any().item())


def masked_empirical_initialization(
    encoded_sequences: list[list[int]],
    *,
    n_hidden_states: int,
    symbol_count: int,
    transition_mask: torch.Tensor,
    emission_mask: torch.Tensor,
    smoothing: float = 1e-6,
    random_seed: int = 0,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a deterministic empirical initialization under the masks."""

    if n_hidden_states < 1:
        raise ValueError("n_hidden_states must be at least 1")
    if symbol_count < 1:
        raise ValueError("symbol_count must be at least 1")

    transition_mask = validate_mask(transition_mask, (n_hidden_states, n_hidden_states), name="transition_mask", device=device, dtype=dtype)
    emission_mask = validate_mask(emission_mask, (n_hidden_states, symbol_count), name="emission_mask", device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)

    initial = torch.rand(n_hidden_states, generator=generator, dtype=dtype, device=device) + smoothing
    transition = torch.rand((n_hidden_states, n_hidden_states), generator=generator, dtype=dtype, device=device) * smoothing
    emission = torch.rand((n_hidden_states, symbol_count), generator=generator, dtype=dtype, device=device) * smoothing

    unigram_counts = Counter()
    bigram_counts = Counter()
    first_counts = Counter()
    for sequence in encoded_sequences:
        if not sequence:
            continue
        first_counts[sequence[0]] += 1
        unigram_counts.update(sequence)
        bigram_counts.update(zip(sequence, sequence[1:]))

    for state in range(n_hidden_states):
        # Bias each state toward frequently observed compatible symbols.
        compatible = torch.nonzero(emission_mask[state] > 0, as_tuple=False).flatten().tolist()
        if not compatible:
            continue
        preferred = max(compatible, key=lambda symbol: unigram_counts.get(symbol, 0))
        emission[state, preferred] += unigram_counts.get(preferred, 0) + smoothing
        for symbol in compatible:
            emission[state, symbol] += smoothing + unigram_counts.get(symbol, 0) / max(1, len(compatible))
            initial[state] += first_counts.get(symbol, 0) / max(1, len(compatible))

    emitters_by_symbol: dict[int, list[int]] = {}
    for symbol in range(symbol_count):
        emitters = torch.nonzero(emission_mask[:, symbol] > 0, as_tuple=False).flatten().tolist()
        emitters_by_symbol[symbol] = emitters or list(range(n_hidden_states))

    for (left_symbol, right_symbol), count in bigram_counts.items():
        # Distribute observed bigram mass across compatible latent emitters.
        left_states = emitters_by_symbol[left_symbol]
        right_states = emitters_by_symbol[right_symbol]
        share = count / max(1, len(left_states) * len(right_states))
        for left_state in left_states:
            for right_state in right_states:
                transition[left_state, right_state] += share

    initial = initial / initial.sum().clamp_min(torch.finfo(dtype).tiny)
    # Final projection guarantees exact support compliance with masks.
    transition = project_matrix_rows(transition, transition_mask, smoothing=smoothing)
    emission = project_matrix_rows(emission, emission_mask, smoothing=smoothing)
    return initial, transition, emission


def _normalize_basis(values: Sequence[Sequence[Any]], *, name: str) -> tuple[tuple[Any, ...], ...]:
    """Normalize prefix/suffix basis to hashable tuples and validate uniqueness."""
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = tuple(tuple(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized
