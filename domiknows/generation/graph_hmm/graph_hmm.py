"""Torch-first DomiKnowS-aware discrete HMM learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .constraints import combine_masks, normalize_matrix_rows, project_distribution, project_matrix, project_matrix_rows, validate_mask
from .dynamic import DynamicConstraintContext, FactorizedStateSpace, apply_transition_energy, transition_energy_matrix
from .graph_adapter import DomiKnowSGraphAdapter
from .spectral import masked_empirical_initialization


@dataclass
class HMMFitResult:
    log_likelihoods: list[float]
    iterations: int
    converged: bool


@dataclass
class ViterbiResult:
    state_ids: tuple[int, ...]
    states: tuple[str, ...]
    score: float


class DomiKnowSAwareHMM:
    """A discrete HMM whose probabilities are projected through graph masks."""

    def __init__(
        self,
        graph=None,
        *,
        concepts: Iterable[Any] | None = None,
        relations: Iterable[Any] | None = None,
        constraints: Iterable[Any] | None = None,
        n_hidden_states: int,
        transition_mask=None,
        emission_mask=None,
        constraint_weight: float = 1.0,
        smoothing: float = 1e-6,
        symbols: Iterable[Any] | None = None,
        state_names: Iterable[str] | None = None,
        device=None,
        dtype: torch.dtype = torch.float64,
        random_seed: int = 0,
        dynamic_transition: Callable[[DynamicConstraintContext], Any] | None = None,
        transition_energy: Callable[[DynamicConstraintContext], Any] | None = None,
        energy_weight: float = 1.0,
        state_space: FactorizedStateSpace | None = None,
        dynamic_metadata: Mapping[str, Any] | None = None,
    ):
        if n_hidden_states < 1:
            raise ValueError("n_hidden_states must be at least 1")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if constraint_weight < 0:
            raise ValueError("constraint_weight must be non-negative")
        if energy_weight < 0:
            raise ValueError("energy_weight must be non-negative")

        self.graph = graph
        self.n_hidden_states = n_hidden_states
        self.constraint_weight = constraint_weight
        self.smoothing = smoothing
        self.device = device
        self.dtype = dtype
        self.random_seed = random_seed
        self.dynamic_transition = dynamic_transition
        self.transition_energy = transition_energy
        self.energy_weight = energy_weight
        self.state_space = state_space
        self.dynamic_metadata = dict(dynamic_metadata or {})
        if state_space is not None and len(state_space) != n_hidden_states:
            raise ValueError("state_space size must match n_hidden_states")
        self.symbols = tuple(symbols) if symbols is not None else None
        if state_names is None and state_space is not None:
            state_names = state_space.state_names
        self.state_names = tuple(state_names) if state_names is not None else tuple(f"S{i}" for i in range(n_hidden_states))
        if len(self.state_names) != n_hidden_states:
            raise ValueError("state_names length must match n_hidden_states")
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be unique")

        self._explicit_transition_mask = transition_mask
        self._explicit_emission_mask = emission_mask
        self.adapter = DomiKnowSGraphAdapter(
            graph,
            concepts=concepts,
            relations=relations,
            constraints=constraints,
            n_hidden_states=n_hidden_states,
            state_names=self.state_names,
            state_space=self.state_space,
            symbols=self.symbols,
            device=device,
            dtype=dtype,
        )

        self.initial_: torch.Tensor | None = None
        self.transition_: torch.Tensor | None = None
        self.emission_: torch.Tensor | None = None
        self.transition_mask_: torch.Tensor | None = None
        self.emission_mask_: torch.Tensor | None = None
        self.symbol_to_id: dict[Any, int] = {}
        self.id_to_symbol: tuple[Any, ...] = ()
        self.fit_result_: HMMFitResult | None = None
        self.constraint_report = self.adapter.report

    def fit(
        self,
        sequences: Sequence[Sequence[Any]],
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        init: str | dict[str, Any] | None = None,
    ) -> "DomiKnowSAwareHMM":
        encoded = self._prepare_training_sequences(sequences)
        self._build_masks(symbol_count=len(self.id_to_symbol))
        self._validate_training_observations(encoded)
        self._initialize_parameters(encoded, init=init)

        log_likelihoods: list[float] = []
        converged = False
        for iteration in range(max_iter):
            pi_counts = torch.zeros(self.n_hidden_states, dtype=self.dtype, device=self.device)
            transition_counts = torch.zeros_like(self.transition_)
            emission_counts = torch.zeros_like(self.emission_)
            total_log_likelihood = 0.0

            for sequence in encoded:
                factors = self._forward_backward_encoded(sequence)
                if factors is None:
                    raise ValueError("training sequence has zero probability under the current graph masks")
                alpha, beta, gamma, xi, log_likelihood = factors
                pi_counts += gamma[0]
                if xi.numel():
                    transition_counts += xi.sum(dim=0)
                for t, symbol_id in enumerate(sequence):
                    emission_counts[:, symbol_id] += gamma[t]
                total_log_likelihood += float(log_likelihood)

            self.initial_ = project_distribution(pi_counts + self.smoothing, torch.ones_like(pi_counts), smoothing=self.smoothing)
            self.transition_ = project_matrix_rows(
                transition_counts + self.smoothing * self.transition_mask_,
                self.transition_mask_,
                smoothing=self.smoothing,
            )
            self.emission_ = project_matrix_rows(
                emission_counts + self.smoothing * self.emission_mask_,
                self.emission_mask_,
                smoothing=self.smoothing,
            )
            log_likelihoods.append(total_log_likelihood)
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                converged = True
                break

        self.fit_result_ = HMMFitResult(log_likelihoods, len(log_likelihoods), converged)
        return self

    def score(self, sequences):
        """Return log-likelihoods for one sequence or a list of sequences."""

        self._require_fitted()
        single = _is_single_sequence(sequences)
        sequence_list = [sequences] if single else list(sequences)
        scores = []
        for sequence in sequence_list:
            encoded = self._encode_sequence(sequence, allow_unknown=False)
            factors = self._forward_backward_encoded(encoded)
            scores.append(float("-inf") if factors is None else float(factors[-1]))
        return scores[0] if single else scores

    def viterbi(self, sequence: Sequence[Any]) -> ViterbiResult:
        self._require_fitted()
        encoded = self._encode_sequence(sequence, allow_unknown=False)
        if not encoded:
            raise ValueError("sequence must not be empty")
        _, emission = self._projected_dynamics()
        tiny = torch.finfo(self.dtype).tiny
        log_initial = torch.log(self.initial_.clamp_min(tiny))
        log_emission = torch.where(
            self.emission_mask_ > 0,
            torch.log(emission.clamp_min(tiny)),
            torch.full_like(self.emission_, float("-inf")),
        )

        delta = log_initial + log_emission[:, encoded[0]]
        backpointers: list[torch.Tensor] = []
        prefix = [self.id_to_symbol[encoded[0]]]
        raw_sequence = tuple(self.id_to_symbol[idx] for idx in encoded)
        for step, symbol_id in enumerate(encoded[1:]):
            belief = _belief_from_log_scores(delta)
            transition = self._transition_for_context(
                step=step,
                prefix=tuple(prefix),
                belief=belief,
                sequence=raw_sequence,
            )
            log_transition = torch.where(
                transition > 0,
                torch.log(transition.clamp_min(tiny)),
                torch.full_like(transition, float("-inf")),
            )
            scores = delta[:, None] + log_transition
            best_scores, best_prev = scores.max(dim=0)
            delta = best_scores + log_emission[:, symbol_id]
            backpointers.append(best_prev)
            prefix.append(self.id_to_symbol[symbol_id])
        best_score, best_state = delta.max(dim=0)
        if not torch.isfinite(best_score):
            return ViterbiResult((), (), float("-inf"))

        states = [int(best_state.item())]
        for backpointer in reversed(backpointers):
            states.append(int(backpointer[states[-1]].item()))
        states.reverse()
        names = tuple(self.state_names[state] for state in states)
        return ViterbiResult(tuple(states), names, float(best_score.item()))

    def sample(self, length: int, *, generator: torch.Generator | None = None) -> list[Any]:
        self._require_fitted()
        if length < 1:
            raise ValueError("length must be at least 1")
        _, emission = self._projected_dynamics()
        state = int(torch.multinomial(self.initial_, 1, generator=generator).item())
        sequence: list[Any] = []
        for step in range(length):
            symbol = int(torch.multinomial(emission[state], 1, generator=generator).item())
            sequence.append(self.id_to_symbol[symbol])
            if step < length - 1:
                belief = torch.zeros(self.n_hidden_states, dtype=self.dtype, device=self.device)
                belief[state] = 1.0
                transition = self._transition_for_context(
                    step=step,
                    prefix=tuple(sequence),
                    belief=belief,
                    sequence=None,
                )
                if transition[state].sum() <= 0:
                    raise RuntimeError(f"no dynamically allowed outgoing transition from state {state} at step {step}")
                state = int(torch.multinomial(transition[state], 1, generator=generator).item())
        return sequence

    def to_constraint_dfa(self):
        """Export an approximate hard DFA over observable symbols.

        DFA states are hidden states plus a dead sink. A transition emits a
        symbol and moves to the most likely next hidden state that can emit it.
        """

        self._require_fitted()
        from domiknows.generation.automata import DFA

        states = set(range(self.n_hidden_states)) | {"start", "dead"}
        alphabet = set(self.id_to_symbol)
        transitions: dict[tuple[Any, Any], Any] = {}
        transition, emission = self._projected_dynamics()

        for symbol_id, symbol in enumerate(self.id_to_symbol):
            start_scores = self.initial_ * emission[:, symbol_id] * self.emission_mask_[:, symbol_id]
            transitions[("start", symbol)] = _argmax_state_or_dead(start_scores)
            transitions[("dead", symbol)] = "dead"
            for state in range(self.n_hidden_states):
                scores = transition[state] * emission[:, symbol_id] * self.emission_mask_[:, symbol_id]
                transitions[(state, symbol)] = _argmax_state_or_dead(scores)

        return DFA(
            states=frozenset(states),
            alphabet=frozenset(alphabet),
            transitions=transitions,
            start_state="start",
            accepting_states=frozenset(range(self.n_hidden_states)) | {"start"},
            dead_states=frozenset({"dead"}),
        )

    def to_torch_learner(self, *, trainable: bool = True, pad_size: int = 4, label_to_token_id=None):
        """Return a PMD-compatible Torch head initialized from this fitted HMM."""
        from .torch_learners import GraphHMMGenerationHead

        return GraphHMMGenerationHead.from_graph_hmm(
            self,
            trainable=trainable,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
        )

    def _prepare_training_sequences(self, sequences: Sequence[Sequence[Any]]) -> list[list[int]]:
        if not sequences:
            raise ValueError("training data must not be empty")
        symbols = list(self.symbols) if self.symbols is not None else []
        seen = set(symbols)
        for sequence in sequences:
            if not sequence:
                raise ValueError("empty sequences are not supported")
            for symbol in sequence:
                if symbol not in seen:
                    symbols.append(symbol)
                    seen.add(symbol)
        if not symbols:
            raise ValueError("symbols must not be empty")
        self.id_to_symbol = tuple(symbols)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.id_to_symbol)}
        self.adapter.set_symbols(self.id_to_symbol)
        return [self._encode_sequence(sequence, allow_unknown=False) for sequence in sequences]

    def _encode_sequence(self, sequence: Sequence[Any], *, allow_unknown: bool) -> list[int]:
        encoded: list[int] = []
        for symbol in sequence:
            if symbol not in self.symbol_to_id:
                if allow_unknown:
                    continue
                raise ValueError(f"unknown symbol {symbol!r}")
            encoded.append(self.symbol_to_id[symbol])
        if not encoded:
            raise ValueError("sequence must not be empty")
        return encoded

    def _build_masks(self, *, symbol_count: int) -> None:
        transition_shape = (self.n_hidden_states, self.n_hidden_states)
        emission_shape = (self.n_hidden_states, symbol_count)
        graph_transition = self.adapter.allowed_transition_mask()
        graph_emission = self.adapter.emission_type_mask()
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
        if (self.transition_mask_.sum(dim=1) == 0).any():
            self.constraint_report.add_unsupported("one or more transition mask rows are all zero; projection will use fallback rows")
        if (self.emission_mask_.sum(dim=1) == 0).any():
            self.constraint_report.add_unsupported("one or more emission mask rows are all zero; projection will use fallback rows")

    def _validate_training_observations(self, encoded: list[list[int]]) -> None:
        for symbol_id, symbol in enumerate(self.id_to_symbol):
            if (self.emission_mask_[:, symbol_id] > 0).any():
                continue
            used = any(symbol_id in sequence for sequence in encoded)
            if used:
                raise ValueError(f"symbol {symbol!r} is forbidden for every hidden state by emission_mask")

    def _initialize_parameters(self, encoded: list[list[int]], *, init: str | dict[str, Any] | None) -> None:
        if isinstance(init, dict):
            initial = torch.as_tensor(init["initial"], dtype=self.dtype, device=self.device)
            transition = validate_mask(init["transition"], (self.n_hidden_states, self.n_hidden_states), name="initial transition", device=self.device, dtype=self.dtype)
            emission = validate_mask(init["emission"], (self.n_hidden_states, len(self.id_to_symbol)), name="initial emission", device=self.device, dtype=self.dtype)
            self.initial_ = project_distribution(initial, torch.ones_like(initial), smoothing=self.smoothing)
            self.transition_ = project_matrix_rows(transition, self.transition_mask_, smoothing=self.smoothing)
            self.emission_ = project_matrix_rows(emission, self.emission_mask_, smoothing=self.smoothing)
            return
        if init == "spectral":
            self.initial_, self.transition_, self.emission_ = masked_empirical_initialization(
                encoded,
                n_hidden_states=self.n_hidden_states,
                symbol_count=len(self.id_to_symbol),
                transition_mask=self.transition_mask_,
                emission_mask=self.emission_mask_,
                smoothing=self.smoothing,
                random_seed=self.random_seed,
                device=self.device,
                dtype=self.dtype,
            )
            return
        if init not in (None, "random"):
            raise ValueError("init must be None, 'random', 'spectral', or a parameter dict")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_seed)
        initial = torch.rand(self.n_hidden_states, generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        transition = torch.rand((self.n_hidden_states, self.n_hidden_states), generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        emission = torch.rand((self.n_hidden_states, len(self.id_to_symbol)), generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        self.initial_ = project_distribution(initial, torch.ones_like(initial), smoothing=self.smoothing)
        self.transition_ = project_matrix_rows(transition, self.transition_mask_, smoothing=self.smoothing)
        self.emission_ = project_matrix_rows(emission, self.emission_mask_, smoothing=self.smoothing)

    def _forward_backward_encoded(self, sequence: list[int]):
        _, emission = self._projected_dynamics()
        length = len(sequence)
        alpha = torch.zeros((length, self.n_hidden_states), dtype=self.dtype, device=self.device)
        scales = torch.zeros(length, dtype=self.dtype, device=self.device)
        transition_sequence: list[torch.Tensor] = []
        raw_sequence = tuple(self.id_to_symbol[idx] for idx in sequence)

        alpha[0] = self.initial_ * emission[:, sequence[0]] * self.emission_mask_[:, sequence[0]]
        scales[0] = alpha[0].sum()
        if scales[0] <= 0:
            return None
        alpha[0] = alpha[0] / scales[0]

        for t in range(1, length):
            transition_t = self._transition_for_context(
                step=t - 1,
                prefix=raw_sequence[:t],
                belief=alpha[t - 1],
                sequence=raw_sequence,
            )
            transition_sequence.append(transition_t)
            alpha[t] = (alpha[t - 1] @ transition_t) * emission[:, sequence[t]] * self.emission_mask_[:, sequence[t]]
            scales[t] = alpha[t].sum()
            if scales[t] <= 0:
                return None
            alpha[t] = alpha[t] / scales[t]

        beta = torch.zeros_like(alpha)
        beta[-1] = 1.0
        for t in range(length - 2, -1, -1):
            beta[t] = transition_sequence[t] @ (emission[:, sequence[t + 1]] * self.emission_mask_[:, sequence[t + 1]] * beta[t + 1])
            beta[t] = beta[t] / scales[t + 1].clamp_min(torch.finfo(self.dtype).tiny)

        gamma = alpha * beta
        gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(torch.finfo(self.dtype).tiny)

        xi = torch.zeros((max(0, length - 1), self.n_hidden_states, self.n_hidden_states), dtype=self.dtype, device=self.device)
        for t in range(length - 1):
            next_factor = emission[:, sequence[t + 1]] * self.emission_mask_[:, sequence[t + 1]] * beta[t + 1]
            xi_t = alpha[t][:, None] * transition_sequence[t] * next_factor[None, :]
            total = xi_t.sum()
            if total > 0:
                xi[t] = xi_t / total
        log_likelihood = torch.log(scales.clamp_min(torch.finfo(self.dtype).tiny)).sum()
        return alpha, beta, gamma, xi, log_likelihood

    def _projected_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        transition = project_matrix(self.transition_, self.transition_mask_, smoothing=self.smoothing)
        emission = project_matrix(self.emission_, self.emission_mask_, smoothing=self.smoothing)
        return transition, emission

    def _transition_for_context(
        self,
        *,
        step: int,
        prefix: tuple[Any, ...],
        belief: torch.Tensor | None,
        sequence: tuple[Any, ...] | None,
    ) -> torch.Tensor:
        transition, _ = self._projected_dynamics()
        if self.dynamic_transition is None and self.transition_energy is None:
            return transition

        context = DynamicConstraintContext(
            step=step,
            prefix=prefix,
            belief=None if belief is None else belief.detach().clone(),
            sequence=sequence,
            metadata={
                "state_names": self.state_names,
                "symbols": self.id_to_symbol,
                "state_space": self.state_space,
                **self.dynamic_metadata,
            },
        )
        weighted = transition
        if self.dynamic_transition is not None:
            dynamic = self.dynamic_transition(context)
            if dynamic is not None:
                factor = validate_mask(
                    dynamic,
                    (self.n_hidden_states, self.n_hidden_states),
                    name="dynamic_transition",
                    device=self.device,
                    dtype=self.dtype,
                )
                weighted = weighted * factor
        if self.transition_energy is not None:
            energy = self.transition_energy(context)
            if energy is not None:
                weighted = apply_transition_energy(
                    weighted,
                    transition_energy_matrix(
                        energy,
                        shape=(self.n_hidden_states, self.n_hidden_states),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    weight=self.energy_weight,
                )
        return normalize_matrix_rows(weighted * self.transition_mask_)

    def _require_fitted(self) -> None:
        if self.initial_ is None or self.transition_ is None or self.emission_ is None:
            raise RuntimeError("DomiKnowSAwareHMM must be fit before this operation")


def _is_single_sequence(value) -> bool:
    if isinstance(value, (str, bytes)):
        return True
    if not isinstance(value, Sequence):
        return True
    if not value:
        return False
    first = value[0]
    return not isinstance(first, Sequence) or isinstance(first, (str, bytes))


def _argmax_state_or_dead(scores: torch.Tensor):
    if scores.numel() == 0 or scores.max() <= 0:
        return "dead"
    return int(scores.argmax().item())


def _belief_from_log_scores(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    if not finite.any():
        return torch.zeros_like(scores)
    shifted = torch.where(finite, scores, torch.full_like(scores, float("-inf")))
    probs = torch.exp(shifted - shifted[finite].max())
    total = probs.sum()
    if total <= 0:
        return torch.zeros_like(scores)
    return probs / total
