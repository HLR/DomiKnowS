"""Torch-first DomiKnowS-aware discrete HMM learner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch

from .constraints import ConstraintApplicationReport, project_distribution, project_matrix, project_matrix_rows, validate_mask
from .dynamic import DynamicConstraintContext, FactorizedStateSpace, FiniteStateDynamicConstraint, apply_transition_energy, transition_energy_matrix
from ...wfa.graph import masked_empirical_initialization


@dataclass
class HMMFitResult:
    """Summary of EM optimization progress."""
    log_likelihoods: list[float]
    iterations: int
    converged: bool


@dataclass
class ViterbiResult:
    """Decoded latent path and associated log score."""
    state_ids: tuple[int, ...]
    states: tuple[str, ...]
    score: float


class DomiKnowSAwareHMM:
    """A discrete HMM whose probabilities are projected through graph masks.

     This learner is DFA-first and supports two construction modes:

     1. (default) pass a generation ``bundle`` plus a DFA or graph context so
         the model compiles DFA support and maps it to HMM masks, or
     2. (optional) pass explicit transition/emission masks directly.
    """

    @classmethod
    def from_generation_constraints(cls, bundle, **kwargs) -> "DomiKnowSAwareHMM":
        """Build an initialized HMM from a generation bundle.

        This is the main user-facing entry point. The bundle is expected to
        carry the graph used to derive DFA support.
        """

        return cls.from_bundle(bundle, **kwargs)

    @classmethod
    def from_bundle(cls, bundle, **kwargs) -> "DomiKnowSAwareHMM":
        """Build an HMM from a generation bundle using its embedded graph.

        The bundle must expose ``bundle.graph`` for DFA-first compilation.
        """

        graph = getattr(bundle, "graph", None)
        if graph is None:
            raise ValueError("bundle must expose graph for DFA-first HMM construction")
        return cls(bundle=bundle, **kwargs)

    @classmethod
    def from_dfa(cls, bundle, dfa, **kwargs) -> "DomiKnowSAwareHMM":
        """Build an HMM from a preconstructed DFA and a generation bundle."""

        return cls(bundle=bundle, dfa=dfa, **kwargs)

    @classmethod
    def from_preconstructed_dfa(cls, bundle, preconstructed_dfa, **kwargs) -> "DomiKnowSAwareHMM":
        """Backward-compatible alias for :meth:`from_dfa`."""

        return cls.from_dfa(bundle, preconstructed_dfa, **kwargs)

    def __init__(
        self,
        *,
        n_hidden_states: int | None = None,
        bundle=None,
        dfa=None,
        preconstructed_dfa=None,
        transition_mask=None,
        emission_mask=None,
        include_other: bool = False,
        eos_token: str | None = None,
        state_name_fn: Callable[[Any, str, Any], str] | None = None,
        on_unsupported: str = "error",
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
        """Initialize a graph-aware HMM with DFA-first or explicit-mask setup.

        Parameters
        ----------
        n_hidden_states:
            Hidden-state count. If omitted, inferred from compiled DFA support or
            explicit mask shapes.
        bundle:
            Generation bundle with vocabulary and graph context used by the
            DFA->HMM compilation path.
        dfa:
            Caller-supplied DFA to use directly instead of compiling from graph.
        preconstructed_dfa:
            Backward-compatible alias for ``dfa``.
        transition_mask:
            Optional explicit transition support matrix for explicit-mask mode.
        emission_mask:
            Optional explicit emission support matrix for explicit-mask mode.
        include_other:
            Include vocabulary "other" token while compiling DFA support.
        eos_token:
            Override EOS token used in DFA->HMM edge-state construction.
        state_name_fn:
            Optional state-naming callback for DFA-edge-backed HMM states.
        on_unsupported:
            Handling policy for unsupported logical constraints during graph->DFA
            compilation.
        constraint_weight:
            Soft weighting exponent applied to positive mask entries.
        smoothing:
            Non-negative additive smoothing for initialization/projection.
        symbols:
            Optional observable symbol order. In DFA-first mode defaults to
            compiled symbols.
        state_names:
            Optional state labels. In DFA-first mode defaults to compiled
            edge-state names.
        device:
            Torch device for all model tensors.
        dtype:
            Torch dtype for model tensors (defaults to float64).
        random_seed:
            Seed used by random/spectral initialization routines.
        dynamic_transition:
            Optional callback returning per-step hard transition compatibility.
        transition_energy:
            Optional callback returning per-step soft transition energy matrix.
        energy_weight:
            Non-negative scale for soft transition energy.
        state_space:
            Optional factorized hidden-state space; length must match
            ``n_hidden_states``.
        dynamic_metadata:
            Extra metadata passed to dynamic callbacks and DFA-export helpers.
        """
        # Step 1: Validate scalar hyperparameters before any state is created.
        if n_hidden_states is not None and n_hidden_states < 1:
            raise ValueError("n_hidden_states must be at least 1")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if constraint_weight < 0:
            raise ValueError("constraint_weight must be non-negative")
        if energy_weight < 0:
            raise ValueError("energy_weight must be non-negative")

        graph = getattr(bundle, "graph", None) if bundle is not None else None
        if dfa is None:
            dfa = preconstructed_dfa

        # Step 2: Determine whether caller selected explicit-mask mode.
        has_explicit_transition = transition_mask is not None
        has_explicit_emission = emission_mask is not None
        has_full_explicit_masks = has_explicit_transition and has_explicit_emission
        if has_explicit_transition != has_explicit_emission:
            raise ValueError(
                "explicit-mask mode requires both transition_mask and emission_mask; "
                "otherwise use DFA-first mode with bundle + (dfa or graph)"
            )

        # Step 3: In explicit-mask mode, infer hidden-state count when omitted.
        if has_full_explicit_masks and n_hidden_states is None:
            transition_shape = torch.as_tensor(transition_mask).shape
            emission_shape = torch.as_tensor(emission_mask).shape
            if len(transition_shape) != 2 or transition_shape[0] != transition_shape[1]:
                raise ValueError("transition_mask must be a square 2D matrix")
            if len(emission_shape) != 2 or emission_shape[0] != transition_shape[0]:
                raise ValueError(
                    "emission_mask must be a 2D matrix with the same row count as transition_mask"
                )
            n_hidden_states = int(transition_shape[0])

        # Step 4: DFA-first default: compile masks unless full explicit masks were supplied.
        compilation = None
        if not has_full_explicit_masks and bundle is not None:
            # Import locally to avoid circular imports at module load time.
            from .constraint_compiler import compile_dfa_to_hmm_support, compile_generation_constraints_to_hmm_support

            if dfa is not None:
                compilation = compile_dfa_to_hmm_support(
                    dfa,
                    bundle,
                    symbols=symbols,
                    eos_token=eos_token,
                    include_other=include_other,
                    state_name_fn=state_name_fn,
                    device=device,
                    dtype=dtype,
                )
            elif graph is not None:
                compilation = compile_generation_constraints_to_hmm_support(
                    graph,
                    bundle,
                    symbols=symbols,
                    eos_token=eos_token,
                    include_other=include_other,
                    state_name_fn=state_name_fn,
                    on_unsupported=on_unsupported,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    "DFA-first mode with bundle requires either dfa or graph"
                )
            if compilation is not None:
                transition_mask = compilation.transition_mask
                emission_mask = compilation.emission_mask
                if symbols is None:
                    symbols = compilation.symbols
                compiled_state_names = tuple(state.name for state in compilation.states)
                if state_names is None:
                    state_names = compiled_state_names
                if n_hidden_states is None:
                    n_hidden_states = len(compilation.states)
                elif n_hidden_states != len(compilation.states):
                    raise ValueError(
                        "n_hidden_states does not match DFA-compiled state count: "
                        f"{n_hidden_states} != {len(compilation.states)}"
                    )

        if transition_mask is None or emission_mask is None:
            raise ValueError(
                "unable to build HMM support: use DFA-first mode by passing "
                "bundle + (dfa or graph), or pass explicit transition_mask and emission_mask"
            )

        if n_hidden_states is None:
            raise ValueError(
                "n_hidden_states is required unless provided implicitly by DFA compilation "
                "or explicit-mask shape inference"
            )

        # Step 5: Store primary model configuration and callback handles.
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
        self.constraint_hmm_compilation = compilation
        self._bundle = bundle
        self._dfa = dfa
        self._dfa_compile_options = {
            "symbols": symbols,
            "eos_token": eos_token,
            "include_other": include_other,
            "state_name_fn": state_name_fn,
            "on_unsupported": on_unsupported,
        }
        # Step 6: Validate optional factorized state-space cardinality.
        if state_space is not None and len(state_space) != n_hidden_states:
            raise ValueError("state_space size must match n_hidden_states")

        # Step 7: Build symbol and state naming metadata used for diagnostics,
        # exports, and readable decoding results.
        self.symbols = tuple(symbols) if symbols is not None else None
        if state_names is None and state_space is not None:
            state_names = state_space.state_names
        self.state_names = tuple(state_names) if state_names is not None else tuple(f"S{i}" for i in range(n_hidden_states))
        if len(self.state_names) != n_hidden_states:
            raise ValueError("state_names length must match n_hidden_states")
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be unique")

        # Step 8: Keep resolved masks; they are the authoritative support.
        self._explicit_transition_mask = transition_mask
        self._explicit_emission_mask = emission_mask

        # Step 9: Initialize learned tensors and runtime bookkeeping.
        # These become non-None after fit() completes.
        self.initial_: torch.Tensor | None = None
        self.transition_: torch.Tensor | None = None
        self.emission_: torch.Tensor | None = None
        self.transition_mask_: torch.Tensor | None = None
        self.emission_mask_: torch.Tensor | None = None
        self.symbol_to_id: dict[Any, int] = {}
        self.id_to_symbol: tuple[Any, ...] = ()
        self.fit_result_: HMMFitResult | None = None

        # Step 10: Collect mask-application diagnostics for caller inspection.
        self.constraint_report = ConstraintApplicationReport()
        if compilation is not None:
            self.constraint_report.add_applied(
                f"compiled {len(compilation.states)} DFA edge state(s) into HMM support"
            )

    def get_generation_dfa(self, *, rebuild: bool = False):
        """Return the generation DFA associated with this model.

        Priority:
        1. DFA captured in existing compilation artifacts.
        2. Caller-provided preconstructed DFA.
        3. Rebuild from stored bundle + graph context.

        Raises:
            ValueError: when no generation-DFA context is available.
        """

        if not rebuild and self.constraint_hmm_compilation is not None:
            return self.constraint_hmm_compilation.dfa

        if self._dfa is not None and not rebuild:
            return self._dfa

        if self._bundle is None:
            raise ValueError(
                "generation DFA is unavailable: initialize with bundle + (dfa or graph), "
                "or call to_constraint_dfa(...) for support DFA export"
            )

        from .constraint_compiler import compile_dfa_to_hmm_support, compile_generation_constraints_to_hmm_support

        if self._dfa is not None:
            compilation = compile_dfa_to_hmm_support(
                self._dfa,
                self._bundle,
                symbols=self._dfa_compile_options["symbols"],
                eos_token=self._dfa_compile_options["eos_token"],
                include_other=self._dfa_compile_options["include_other"],
                state_name_fn=self._dfa_compile_options["state_name_fn"],
                device=self.device,
                dtype=self.dtype,
            )
        elif self.graph is not None:
            compilation = compile_generation_constraints_to_hmm_support(
                self.graph,
                self._bundle,
                symbols=self._dfa_compile_options["symbols"],
                eos_token=self._dfa_compile_options["eos_token"],
                include_other=self._dfa_compile_options["include_other"],
                state_name_fn=self._dfa_compile_options["state_name_fn"],
                on_unsupported=self._dfa_compile_options["on_unsupported"],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(
                "generation DFA is unavailable: bundle is set but no graph/dfa is available"
            )

        self.constraint_hmm_compilation = compilation
        return compilation.dfa

    def fit(
        self,
        sequences: Sequence[Sequence[Any]],
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        init: str | dict[str, Any] | None = None,
    ) -> "DomiKnowSAwareHMM":
        """Fit parameters with constrained Baum-Welch under static/dynamic masks."""
        # Step 1: Build a stable symbol vocabulary and convert sequences to ids.
        encoded = self._prepare_training_sequences(sequences)
        # Step 2: Validate/prepare explicit transition and emission masks.
        self._build_masks(symbol_count=len(self.id_to_symbol))
        # Step 3: Fail fast if training data contains globally forbidden symbols.
        self._validate_training_observations(encoded)
        # Step 4: Initialize parameters (explicit tensors, spectral, or random).
        self._initialize_parameters(encoded, init=init)

        log_likelihoods: list[float] = []
        converged = False
        for iteration in range(max_iter):
            # Step 5 (E-step accumulator init): expected sufficient statistics
            # for initial state, transitions, and emissions over this iteration.
            pi_counts = torch.zeros(self.n_hidden_states, dtype=self.dtype, device=self.device)
            transition_counts = torch.zeros_like(self.transition_)
            emission_counts = torch.zeros_like(self.emission_)
            total_log_likelihood = 0.0

            for sequence in encoded:
                # Step 5a: Run constrained forward-backward on one sequence.
                # It already applies dynamic constraints per time step.
                factors = self._forward_backward_encoded(sequence)
                if factors is None:
                    raise ValueError("training sequence has zero probability under the current graph masks")
                alpha, beta, gamma, xi, log_likelihood = factors
                # Step 5b: Aggregate expected initial-state counts.
                pi_counts += gamma[0]
                if xi.numel():
                    # Step 5c: Aggregate expected transition counts.
                    transition_counts += xi.sum(dim=0)
                for t, symbol_id in enumerate(sequence):
                    # Step 5d: Aggregate expected emission counts.
                    emission_counts[:, symbol_id] += gamma[t]
                total_log_likelihood += float(log_likelihood)

            # Step 6 (M-step): re-estimate parameters from expected counts,
            # then project to legal support so forbidden entries remain zero.
            self.initial_ = project_distribution(pi_counts + self.smoothing, torch.ones_like(pi_counts), smoothing=self.smoothing)
            # Re-project M-step estimates so forbidden support remains exactly zero.
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
            # Step 7: Stop if absolute log-likelihood improvement is small.
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                converged = True
                break

        self.fit_result_ = HMMFitResult(log_likelihoods, len(log_likelihoods), converged)
        return self

    def score(self, sequences):
        """Return log-likelihoods for one sequence or a list of sequences."""

        self._require_fitted()
        # Step 1: Normalize input to either one sequence or a batch.
        single = _is_single_sequence(sequences)
        sequence_list = [sequences] if single else list(sequences)
        scores = []
        for sequence in sequence_list:
            # Step 2: Encode sequence into vocabulary ids.
            encoded = self._encode_sequence(sequence, allow_unknown=False)
            # Step 3: Compute sequence log-likelihood via forward-backward.
            factors = self._forward_backward_encoded(encoded)
            # Step 4: Return -inf for impossible sequences at inference time.
            scores.append(float("-inf") if factors is None else float(factors[-1]))
        return scores[0] if single else scores

    def viterbi(self, sequence: Sequence[Any]) -> ViterbiResult:
        """Decode the most likely hidden-state sequence under constraints."""
        self._require_fitted()
        # Step 1: Validate and encode the observable sequence.
        encoded = self._encode_sequence(sequence, allow_unknown=False)
        if not encoded:
            raise ValueError("sequence must not be empty")
        # Step 2: Prepare log-space initial/emission scores for stability.
        _, emission = self._projected_dynamics()
        tiny = torch.finfo(self.dtype).tiny
        log_initial = torch.log(self.initial_.clamp_min(tiny))
        log_emission = torch.where(
            self.emission_mask_ > 0,
            torch.log(emission.clamp_min(tiny)),
            torch.full_like(self.emission_, float("-inf")),
        )

        # Step 3: Initialize DP with first observation.
        delta = log_initial + log_emission[:, encoded[0]]
        # Backpointers store argmax previous state for each next state/time.
        backpointers: list[torch.Tensor] = []
        prefix = [self.id_to_symbol[encoded[0]]]
        raw_sequence = tuple(self.id_to_symbol[idx] for idx in encoded)
        for step, symbol_id in enumerate(encoded[1:]):
            # Step 4a: Convert DP log-scores to a belief proxy for callbacks.
            belief = _belief_from_log_scores(delta)
            transition = self._transition_for_context(
                step=step,
                prefix=tuple(prefix),
                belief=belief,
                sequence=raw_sequence,
            )
            # Step 4b: Score all predecessor->successor transitions.
            log_transition = torch.where(
                transition > 0,
                torch.log(transition.clamp_min(tiny)),
                torch.full_like(transition, float("-inf")),
            )
            scores = delta[:, None] + log_transition
            best_scores, best_prev = scores.max(dim=0)
            # Step 4c: Keep the best predecessor and add current emission score.
            delta = best_scores + log_emission[:, symbol_id]
            backpointers.append(best_prev)
            prefix.append(self.id_to_symbol[symbol_id])
        best_score, best_state = delta.max(dim=0)
        if not torch.isfinite(best_score):
            # No legal hidden path emits the whole sequence.
            return ViterbiResult((), (), float("-inf"))

        # Step 5: Backtrack through stored argmax pointers.
        states = [int(best_state.item())]
        for backpointer in reversed(backpointers):
            states.append(int(backpointer[states[-1]].item()))
        states.reverse()
        names = tuple(self.state_names[state] for state in states)
        return ViterbiResult(tuple(states), names, float(best_score.item()))

    def sample(self, length: int, *, generator: torch.Generator | None = None) -> list[Any]:
        """Sample an observable sequence while enforcing all active constraints."""
        self._require_fitted()
        if length < 1:
            raise ValueError("length must be at least 1")
        # Step 1: Use currently projected dynamics as the sampling model.
        _, emission = self._projected_dynamics()
        # Step 2: Draw initial hidden state from the fitted initial distribution.
        state = int(torch.multinomial(self.initial_, 1, generator=generator).item())
        sequence: list[Any] = []
        for step in range(length):
            # Step 3: Emit one symbol from the current state's emission row.
            symbol = int(torch.multinomial(emission[state], 1, generator=generator).item())
            sequence.append(self.id_to_symbol[symbol])
            if step < length - 1:
                # Step 4: Build one-hot belief because sampled hidden state is known.
                belief = torch.zeros(self.n_hidden_states, dtype=self.dtype, device=self.device)
                belief[state] = 1.0
                transition = self._transition_for_context(
                    step=step,
                    prefix=tuple(sequence),
                    belief=belief,
                    sequence=None,
                )
                if transition[state].sum() <= 0:
                    # Dynamic constraints can invalidate all outgoing edges from a row.
                    raise RuntimeError(f"no dynamically allowed outgoing transition from state {state} at step {step}")
                # Step 5: Sample next hidden state from filtered transition row.
                state = int(torch.multinomial(transition[state], 1, generator=generator).item())
        return sequence

    def to_constraint_dfa(
        self,
        *,
        finite_state_dynamic: FiniteStateDynamicConstraint | None = None,
        on_unsupported_dynamic: Literal["error", "static"] = "error",
        support_threshold: float = 0.0,
    ):
        """Export a hard-support DFA over observable symbols.

        The exported language contains exactly the observable strings that have
        at least one positive-probability hidden-state path under the current
        projected initial, transition, and emission supports. DFA states are
        reachable sets of HMM states, optionally paired with a caller-supplied
        finite dynamic-constraint state.
        """

        self._require_fitted()
        if on_unsupported_dynamic not in {"error", "static"}:
            raise ValueError("on_unsupported_dynamic must be 'error' or 'static'")
        if support_threshold < 0:
            raise ValueError("support_threshold must be non-negative")
        if finite_state_dynamic is None and on_unsupported_dynamic == "error":
            if self.dynamic_transition is not None:
                raise ValueError(
                    "dynamic_transition cannot be exported exactly as a DFA without "
                    "finite_state_dynamic; pass finite_state_dynamic=... or "
                    "on_unsupported_dynamic='static' to intentionally ignore dynamic callbacks"
                )
            if self.transition_energy is not None:
                raise ValueError(
                    "transition_energy is a soft scoring bias, not a hard support constraint; "
                    "pass finite_state_dynamic=... after converting it to a hard finite-state "
                    "mask, or on_unsupported_dynamic='static' to intentionally ignore it"
                )

        from ....dfa import DFA

        # Step 1: Convert fitted probabilities to hard support using threshold.
        transition, emission = self._projected_dynamics()
        transition_support = transition > support_threshold
        emission_support = (emission > support_threshold) & (self.emission_mask_ > 0)
        initial_support = self.initial_ > support_threshold

        alphabet = tuple(self.id_to_symbol)
        start = "start"
        dead = frozenset()
        states: set[Any] = {start, dead}
        transitions: dict[tuple[Any, Any], Any] = {}
        metadata = {
            "state_names": self.state_names,
            "symbols": self.id_to_symbol,
            "state_space": self.state_space,
            "support_threshold": support_threshold,
            **self.dynamic_metadata,
        }

        def ensure_hashable(value: Any, *, name: str) -> Any:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be hashable for DFA export, got {value!r}") from exc
            return value

        if finite_state_dynamic is not None:
            ensure_hashable(finite_state_dynamic.start_state, name="finite_state_dynamic.start_state")

        def dynamic_mask(dynamic_state: Any, reachable_states: frozenset[int]) -> torch.Tensor:
            if finite_state_dynamic is None:
                return transition_support
            # Caller hook supplies additional finite-state transition legality.
            mask = validate_mask(
                finite_state_dynamic.transition_mask(dynamic_state, reachable_states, metadata),
                (self.n_hidden_states, self.n_hidden_states),
                name="finite_state_dynamic.transition_mask",
                device=self.device,
                dtype=self.dtype,
            )
            return transition_support & (mask > support_threshold)

        def advance_dynamic(dynamic_state: Any, symbol: Any, reachable_states: frozenset[int]) -> Any:
            if finite_state_dynamic is None:
                return None
            next_dynamic = finite_state_dynamic.advance(dynamic_state, symbol, reachable_states, metadata)
            return ensure_hashable(next_dynamic, name="finite_state_dynamic.advance(...)")

        def next_reachable(current: Any, symbol_id: int) -> frozenset[int]:
            # Step 2: Compute hidden states reachable after reading one symbol.
            if current == start:
                # DFA start has no previous hidden state, so use initial support.
                reachable = initial_support & emission_support[:, symbol_id]
            else:
                current_reachable = current[0] if finite_state_dynamic is not None else current
                current_dynamic = current[1] if finite_state_dynamic is not None else None
                # Expand one HMM step from the current reachable-set frontier.
                current_mask = torch.zeros(self.n_hidden_states, dtype=torch.bool, device=self.device)
                for state in current_reachable:
                    current_mask[int(state)] = True
                support = dynamic_mask(current_dynamic, current_reachable)
                reachable = (current_mask.to(dtype=self.dtype) @ support.to(dtype=self.dtype)) > 0
                reachable = reachable & emission_support[:, symbol_id]
            ids = torch.nonzero(reachable, as_tuple=False).flatten().tolist()
            return frozenset(int(state) for state in ids)

        def next_dfa_state(current: Any, symbol_id: int, symbol: Any) -> Any:
            # Step 3: Turn hidden-state reachability into a DFA state object.
            reachable = next_reachable(current, symbol_id)
            if not reachable:
                return dead
            if finite_state_dynamic is None:
                return reachable
            if current == start:
                current_dynamic = finite_state_dynamic.start_state
            else:
                current_dynamic = current[1]
            return (reachable, advance_dynamic(current_dynamic, symbol, reachable))

        def is_accepting_dfa_state(state: Any) -> bool:
            if state == start or state == dead:
                return False
            if finite_state_dynamic is None:
                return True
            return finite_state_dynamic.accepts(state[1])

        queue = deque([start])
        accepting: set[Any] = set()
        # Step 4: BFS over reachable DFA states to materialize all transitions.
        while queue:
            state = queue.popleft()
            for symbol_id, symbol in enumerate(alphabet):
                next_state = next_dfa_state(state, symbol_id, symbol)
                transitions[(state, symbol)] = next_state
                if next_state not in states:
                    states.add(next_state)
                    if is_accepting_dfa_state(next_state):
                        accepting.add(next_state)
                    queue.append(next_state)

        for symbol in alphabet:
            # Step 5: Dead state loops to itself on all symbols.
            transitions[(dead, symbol)] = dead

        return DFA(
            states=frozenset(states),
            alphabet=frozenset(alphabet),
            transitions=transitions,
            start_state=start,
            accepting_states=frozenset(accepting),
            dead_states=frozenset({dead}),
        )

    def to_torch_learner(self, *, trainable: bool = True, pad_size: int = 4, label_to_token_id=None, random_seed: int | None = None):
        """Return a PMD-compatible Torch head initialized from this fitted HMM."""
        # Import lazily so this module can be used without torch-head extras.
        from .graphAwareHMMLearner import GraphHMMGenerationHead

        return GraphHMMGenerationHead.from_graph_hmm(
            self,
            trainable=trainable,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            random_seed=random_seed,
        )

    def _prepare_training_sequences(self, sequences: Sequence[Sequence[Any]]) -> list[list[int]]:
        """Build a stable symbol vocabulary and encode all training sequences."""
        if not sequences:
            raise ValueError("training data must not be empty")
        # Step 1: Start from optional user-provided symbol order.
        # Step 2: Append unseen symbols in first-observed order from training data.
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
        # Step 3: Persist symbol<->id mappings used by all subsequent methods.
        self.id_to_symbol = tuple(symbols)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.id_to_symbol)}
        return [self._encode_sequence(sequence, allow_unknown=False) for sequence in sequences]

    def _encode_sequence(self, sequence: Sequence[Any], *, allow_unknown: bool) -> list[int]:
        """Map symbols to integer ids; optionally skip unknown symbols."""
        encoded: list[int] = []
        for symbol in sequence:
            if symbol not in self.symbol_to_id:
                if allow_unknown:
                    # In permissive mode we silently drop unknown symbols.
                    continue
                raise ValueError(f"unknown symbol {symbol!r}")
            encoded.append(self.symbol_to_id[symbol])
        if not encoded:
            raise ValueError("sequence must not be empty")
        return encoded

    def _build_masks(self, *, symbol_count: int) -> None:
        """Validate explicit masks and apply optional soft constraint weighting."""
        transition_shape = (self.n_hidden_states, self.n_hidden_states)
        emission_shape = (self.n_hidden_states, symbol_count)
        # Step 1: Enforce explicit mask-only construction path.
        if self._explicit_transition_mask is None or self._explicit_emission_mask is None:
            raise ValueError(
                "transition_mask and emission_mask are required; "
                "DFA-first mode expects bundle with dfa or graph; "
                "explicit-mask mode expects both transition_mask and emission_mask"
            )

        # Step 2: Validate and materialize explicit masks to target shapes/dtype.
        self.transition_mask_ = validate_mask(
            self._explicit_transition_mask,
            transition_shape,
            name="transition_mask",
            device=self.device,
            dtype=self.dtype,
        )
        self.emission_mask_ = validate_mask(
            self._explicit_emission_mask,
            emission_shape,
            name="emission_mask",
            device=self.device,
            dtype=self.dtype,
        )

        if self.constraint_weight != 1.0:
            # Step 3: Optionally reweight positive mask entries (soft preference).
            self.transition_mask_ = _apply_constraint_weight(self.transition_mask_, self.constraint_weight)
            self.emission_mask_ = _apply_constraint_weight(self.emission_mask_, self.constraint_weight)
            self.constraint_report.add_applied(
                f"applied soft explicit-mask weighting with constraint_weight={self.constraint_weight}"
            )
        # Step 4: Record unsupported rows for diagnostics.
        if (self.transition_mask_.sum(dim=1) == 0).any():
            self.constraint_report.add_unsupported("one or more transition mask rows are all zero; projected rows will remain zero")
        if (self.emission_mask_.sum(dim=1) == 0).any():
            self.constraint_report.add_unsupported("one or more emission mask rows are all zero; projected rows will remain zero")

    def _validate_training_observations(self, encoded: list[list[int]]) -> None:
        """Fail early when observed symbols are globally forbidden by emission masks."""
        # If a symbol appears in data but no state is allowed to emit it,
        # EM cannot assign any positive likelihood to those sequences.
        for symbol_id, symbol in enumerate(self.id_to_symbol):
            if (self.emission_mask_[:, symbol_id] > 0).any():
                continue
            used = any(symbol_id in sequence for sequence in encoded)
            if used:
                raise ValueError(f"symbol {symbol!r} is forbidden for every hidden state by emission_mask")

    def _initialize_parameters(self, encoded: list[list[int]], *, init: str | dict[str, Any] | None) -> None:
        """Initialize parameters from user values, spectral init, or seeded random init."""
        if isinstance(init, dict):
            # Step 1A: Use caller-provided tensors, then project to legal support.
            initial = torch.as_tensor(init["initial"], dtype=self.dtype, device=self.device)
            transition = validate_mask(init["transition"], (self.n_hidden_states, self.n_hidden_states), name="initial transition", device=self.device, dtype=self.dtype)
            emission = validate_mask(init["emission"], (self.n_hidden_states, len(self.id_to_symbol)), name="initial emission", device=self.device, dtype=self.dtype)
            self.initial_ = project_distribution(initial, torch.ones_like(initial), smoothing=self.smoothing)
            self.transition_ = project_matrix_rows(transition, self.transition_mask_, smoothing=self.smoothing)
            self.emission_ = project_matrix_rows(emission, self.emission_mask_, smoothing=self.smoothing)
            return
        if init == "spectral":
            # Step 1B: Use masked empirical initialization from observed data.
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

        # Step 1C: Default to reproducible random init, then projection.
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.random_seed)
        initial = torch.rand(self.n_hidden_states, generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        transition = torch.rand((self.n_hidden_states, self.n_hidden_states), generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        emission = torch.rand((self.n_hidden_states, len(self.id_to_symbol)), generator=generator, dtype=self.dtype, device=self.device) + self.smoothing
        self.initial_ = project_distribution(initial, torch.ones_like(initial), smoothing=self.smoothing)
        self.transition_ = project_matrix_rows(transition, self.transition_mask_, smoothing=self.smoothing)
        self.emission_ = project_matrix_rows(emission, self.emission_mask_, smoothing=self.smoothing)

    def _forward_backward_encoded(self, sequence: list[int]):
        """Run scaled forward-backward for one encoded sequence.

        Returns normalized alpha/beta, posterior gamma/xi, and sequence log-likelihood.
        Returns None when constraints make the sequence impossible.
        """
        # Step 1: Retrieve projected dynamics and allocate scaled DP buffers.
        _, emission = self._projected_dynamics()
        length = len(sequence)
        # Scaling factors prevent underflow on long sequences.
        alpha = torch.zeros((length, self.n_hidden_states), dtype=self.dtype, device=self.device)
        scales = torch.zeros(length, dtype=self.dtype, device=self.device)
        transition_sequence: list[torch.Tensor] = []
        raw_sequence = tuple(self.id_to_symbol[idx] for idx in sequence)

        # Step 2: Forward init at t=0 using initial * emission support.
        alpha[0] = self.initial_ * emission[:, sequence[0]] * self.emission_mask_[:, sequence[0]]
        scales[0] = alpha[0].sum()
        if scales[0] <= 0:
            # No legal starting state can emit the first symbol.
            return None
        alpha[0] = alpha[0] / scales[0]

        # Step 3: Forward recursion with per-step dynamic transition matrices.
        for t in range(1, length):
            # Each step may use a different matrix due to dynamic hooks/energy.
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
                # Dynamic/static constraints block all paths at this step.
                return None
            alpha[t] = alpha[t] / scales[t]

        # Step 4: Backward recursion over the stored per-step matrices.
        beta = torch.zeros_like(alpha)
        beta[-1] = 1.0
        for t in range(length - 2, -1, -1):
            beta[t] = transition_sequence[t] @ (emission[:, sequence[t + 1]] * self.emission_mask_[:, sequence[t + 1]] * beta[t + 1])
            beta[t] = beta[t] / scales[t + 1].clamp_min(torch.finfo(self.dtype).tiny)

        # Step 5: Posterior marginals per state/time.
        gamma = alpha * beta
        gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(torch.finfo(self.dtype).tiny)

        # Step 6: Posterior pairwise marginals over transitions.
        xi = torch.zeros((max(0, length - 1), self.n_hidden_states, self.n_hidden_states), dtype=self.dtype, device=self.device)
        for t in range(length - 1):
            # Xi(t, i, j) is the posterior of transition i->j at step t.
            next_factor = emission[:, sequence[t + 1]] * self.emission_mask_[:, sequence[t + 1]] * beta[t + 1]
            xi_t = alpha[t][:, None] * transition_sequence[t] * next_factor[None, :]
            total = xi_t.sum()
            if total > 0:
                xi[t] = xi_t / total
        # Step 7: Sequence log-likelihood is sum(log(scale_t)).
        log_likelihood = torch.log(scales.clamp_min(torch.finfo(self.dtype).tiny)).sum()
        return alpha, beta, gamma, xi, log_likelihood

    def _projected_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transition/emission matrices projected onto legal support."""
        # Projection enforces hard zeros and row normalization consistently.
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
        """Build the per-step transition matrix under dynamic hard/soft constraints."""
        # Step 1: Start from the static projected transition matrix.
        transition, _ = self._projected_dynamics()
        # Fast path: no dynamic hooks, return static projected transition.
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
        # Step 2: Build callback context for this decoding/training step.
        weighted = transition
        effective_mask = self.transition_mask_
        if self.dynamic_transition is not None:
            # Step 3: Apply hard dynamic compatibility (can zero transitions).
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
                # Dynamic hard zeros must remain hard zeros.  A dynamic
                # compatibility matrix may use positive non-binary weights, so
                # only its positive support participates in the projection mask.
                effective_mask = effective_mask * (factor > 0).to(dtype=self.dtype)
        if self.transition_energy is not None:
            # Step 4: Apply optional soft energy reweighting.
            energy = self.transition_energy(context)
            if energy is not None:
                # Soft energy downweights transitions via exp(-weight * energy).
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
        # Step 5: Project onto combined static + dynamic hard support. Using only
        # the static mask here would let project_distribution recover a
        # dynamically all-zero row from the static support, reintroducing
        # transitions that the runtime hook explicitly blocked.
        return project_matrix_rows(weighted, effective_mask, smoothing=self.smoothing)

    def _require_fitted(self) -> None:
        """Guard operations that require learned parameters."""
        # A model is considered fitted only when all parameter tensors exist.
        if self.initial_ is None or self.transition_ is None or self.emission_ is None:
            raise RuntimeError("DomiKnowSAwareHMM must be fit before this operation")


def _is_single_sequence(value) -> bool:
    """Heuristic to distinguish one sequence from a batch of sequences."""
    # Strings are treated as single sequence elements, not batches.
    if isinstance(value, (str, bytes)):
        return True
    if not isinstance(value, Sequence):
        return True
    if not value:
        return False
    first = value[0]
    return not isinstance(first, Sequence) or isinstance(first, (str, bytes))


def _apply_constraint_weight(mask: torch.Tensor, weight: float) -> torch.Tensor:
    """Apply soft weighting to positive mask entries while preserving hard zeros.

    - Entries equal to 0 stay 0 (forbidden support remains forbidden).
    - Positive entries are transformed as ``entry ** weight``.
      This lets non-binary masks express softer/stronger preferences.
    """
    positive = mask > 0
    if not positive.any():
        return mask
    weighted = torch.zeros_like(mask)
    weighted[positive] = torch.pow(mask[positive], weight)
    return weighted


def _belief_from_log_scores(scores: torch.Tensor) -> torch.Tensor:
    """Convert unnormalized log-scores into a probability belief vector."""
    finite = torch.isfinite(scores)
    if not finite.any():
        return torch.zeros_like(scores)
    # Subtract max finite log-score before exp for numerical stability.
    shifted = torch.where(finite, scores, torch.full_like(scores, float("-inf")))
    probs = torch.exp(shifted - shifted[finite].max())
    total = probs.sum()
    if total <= 0:
        return torch.zeros_like(scores)
    return probs / total
