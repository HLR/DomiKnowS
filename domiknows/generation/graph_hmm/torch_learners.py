"""Torch learner heads that expose graph-aware sequence models to PMD pipelines.

This module provides two compact-label neural heads:
- an HMM-based head that projects parameters through static/dynamic constraints,
- a spectral (signed WFA) head recovered from finite-rank Hankel learning.

Both heads expose log-probabilities suitable for differentiable training while
keeping graph/constraint semantics explicit.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Mapping

import torch

from .constraints import combine_masks, project_matrix_rows, validate_mask
from .dynamic import DynamicConstraintContext, FactorizedStateSpace, apply_transition_energy, transition_energy_matrix
from .graph_adapter import DomiKnowSGraphAdapter


class GraphHMMGenerationHead(torch.nn.Module):
    """Compact-label HMM head that projects parameters through graph constraints."""

    def __init__(
        self,
        *,
        graph=None,
        concepts: Sequence[Any] | None = None,
        relations: Sequence[Any] | None = None,
        constraints: Sequence[Any] | None = None,
        n_hidden_states: int,
        label_count: int,
        symbols: Sequence[Any] | None = None,
        state_names: Sequence[str] | None = None,
        transition_mask=None,
        emission_mask=None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int = 0,
        initial=None,
        transition=None,
        emission=None,
        dynamic_transition: Callable[[DynamicConstraintContext], Any] | None = None,
        transition_energy: Callable[[DynamicConstraintContext], Any] | None = None,
        energy_weight: float = 1.0,
        state_space: FactorizedStateSpace | None = None,
        dynamic_metadata: Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize a compact-label HMM head with graph-constrained support."""
        super().__init__()
        if n_hidden_states < 1:
            raise ValueError("n_hidden_states must be at least 1")
        if label_count < 1:
            raise ValueError("label_count must be at least 1")
        if pad_size < 1:
            raise ValueError("pad_size must be at least 1")
        if energy_weight < 0:
            raise ValueError("energy_weight must be non-negative")

        self.n_hidden_states = int(n_hidden_states)
        self.label_count = int(label_count)
        self.pad_size = int(pad_size)
        self.state_space = state_space
        self.state_names = tuple(state_names) if state_names is not None else (
            tuple(state_space.state_names) if state_space is not None else tuple(f"S{i}" for i in range(self.n_hidden_states))
        )
        if len(self.state_names) != self.n_hidden_states:
            raise ValueError("state_names length must match n_hidden_states")
        self.symbols = tuple(symbols) if symbols is not None else tuple(range(self.label_count))
        if len(self.symbols) != self.label_count:
            raise ValueError("symbols length must match label_count")
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)
        self.dynamic_transition = dynamic_transition
        self.transition_energy = transition_energy
        self.energy_weight = float(energy_weight)
        self.dynamic_metadata = dict(dynamic_metadata or {})

        adapter = DomiKnowSGraphAdapter(
            graph,
            concepts=concepts,
            relations=relations,
            constraints=constraints,
            n_hidden_states=self.n_hidden_states,
            state_names=self.state_names,
            state_space=state_space,
            symbols=self.symbols,
            dtype=dtype,
        )
        graph_transition = adapter.allowed_transition_mask()
        graph_emission = adapter.emission_type_mask()
        self.constraint_report = adapter.report

        transition_mask_t = combine_masks(
            (graph_transition, transition_mask),
            (self.n_hidden_states, self.n_hidden_states),
            name="transition_mask",
            dtype=dtype,
        ).to(dtype=torch.float32)
        emission_mask_t = combine_masks(
            (graph_emission, emission_mask),
            (self.n_hidden_states, self.label_count),
            name="emission_mask",
            dtype=dtype,
        ).to(dtype=torch.float32)
        self.register_buffer("transition_mask", transition_mask_t)
        self.register_buffer("emission_mask", emission_mask_t)

        if initial is None or transition is None or emission is None:
            initial_t, transition_t, emission_t = _random_hmm_parameters(
                self.n_hidden_states,
                self.label_count,
                random_seed,
                self.transition_mask,
                self.emission_mask,
            )
        else:
            initial_t = _normalize_vector(torch.as_tensor(initial, dtype=torch.float32))
            transition_t = project_matrix_rows(
                torch.as_tensor(transition, dtype=torch.float32),
                self.transition_mask,
            ).to(dtype=torch.float32)
            emission_t = project_matrix_rows(
                torch.as_tensor(emission, dtype=torch.float32),
                self.emission_mask,
            ).to(dtype=torch.float32)
            _validate_hmm_shapes(initial_t, transition_t, emission_t, self.n_hidden_states, self.label_count)

        self.initial_logits = torch.nn.Parameter(_safe_log(initial_t), requires_grad=trainable)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition_t), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission_t), requires_grad=trainable)

    @classmethod
    def from_graph_hmm(
        cls,
        learner,
        *,
        trainable: bool = True,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
    ) -> "GraphHMMGenerationHead":
        """Create a PMD head initialized from a fitted ``DomiKnowSAwareHMM``."""
        learner._require_fitted()
        return cls(
            graph=learner.graph,
            n_hidden_states=learner.n_hidden_states,
            label_count=len(learner.id_to_symbol),
            symbols=learner.id_to_symbol,
            state_names=learner.state_names,
            transition_mask=learner.transition_mask_,
            emission_mask=learner.emission_mask_,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            trainable=trainable,
            initial=learner.initial_,
            transition=learner.transition_,
            emission=learner.emission_,
            dynamic_transition=learner.dynamic_transition,
            transition_energy=learner.transition_energy,
            energy_weight=learner.energy_weight,
            state_space=learner.state_space,
            dynamic_metadata=learner.dynamic_metadata,
            dtype=learner.dtype,
        )

    @property
    def initial_probs(self) -> torch.Tensor:
        """Return normalized initial state probabilities."""
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        """Return transition probabilities reprojected onto legal support."""
        return project_matrix_rows(torch.exp(self.transition_logits), self.transition_mask).to(self.transition_logits.dtype)

    @property
    def emission_probs(self) -> torch.Tensor:
        """Return emission probabilities reprojected onto legal support."""
        return project_matrix_rows(torch.exp(self.emission_logits), self.emission_mask).to(self.emission_logits.dtype)

    def token_id_for_label(self, label: int) -> int:
        """Resolve tokenizer id for a label; fail for non 1-to-1 mappings."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def trainable_parameter_names(self) -> list[str]:
        """List parameter names currently participating in gradient updates."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def sequence_log_probs(self, target_labels: torch.Tensor | Sequence[int], *, lengths=None) -> torch.Tensor:
        """Compute per-step log-probabilities for one batch of label sequences."""
        labels, _lengths_t, step_mask, squeeze = _target_label_batch(target_labels, self.pad_size, lengths=lengths)
        batch, seq_len = labels.shape
        state = self.initial_probs.expand(batch, -1)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        outputs = []
        prefixes: list[list[int]] = [[] for _ in range(batch)]
        for step in range(seq_len):
            # Predict next-label distribution from current latent belief.
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            # Posterior over hidden states after observing current label.
            posterior = state * emission[:, labels[:, step]].transpose(0, 1)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_states = []
            for batch_index in range(batch):
                transition = self._transition_for_prefix(
                    step=step,
                    prefix=tuple(prefixes[batch_index] + [int(labels[batch_index, step].item())]),
                    belief=posterior[batch_index],
                )
                next_states.append(torch.matmul(posterior[batch_index], transition))
                prefixes[batch_index].append(int(labels[batch_index, step].item()))
            state = torch.stack(next_states, dim=0)
        result = torch.stack(outputs, dim=1)
        # Zero padded timesteps so downstream losses can ignore them safely.
        result = result * step_mask.to(dtype=result.dtype).unsqueeze(-1)
        return result[0] if squeeze else result

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return next-label logits for a decoded prefix of token ids."""
        prefix_labels = _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)
        state = self.initial_probs
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        for step, label in enumerate(prefix_labels):
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            transition = self._transition_for_prefix(step=step, prefix=tuple(prefix_labels[: step + 1]), belief=posterior)
            state = torch.matmul(posterior, transition)
        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """PMD module interface: returns sequence log-probabilities."""
        return self.sequence_log_probs(target_labels)

    def _transition_for_prefix(self, *, step: int, prefix: tuple[int, ...], belief: torch.Tensor | None) -> torch.Tensor:
        """Build per-step transition matrix under optional dynamic constraints."""
        transition = self.transition_probs
        if self.dynamic_transition is None and self.transition_energy is None:
            return transition
        context = DynamicConstraintContext(
            step=step,
            prefix=tuple(self.symbols[label] for label in prefix),
            belief=None if belief is None else belief.detach().clone(),
            sequence=None,
            metadata={
                "state_names": self.state_names,
                "symbols": self.symbols,
                "state_space": self.state_space,
                **self.dynamic_metadata,
            },
        )
        weighted = transition
        effective_mask = self.transition_mask.to(device=weighted.device, dtype=weighted.dtype)
        if self.dynamic_transition is not None:
            # Hard multiplicative transition factor from user callback.
            dynamic = self.dynamic_transition(context)
            if dynamic is not None:
                factor = validate_mask(
                    dynamic,
                    (self.n_hidden_states, self.n_hidden_states),
                    name="dynamic_transition",
                    device=weighted.device,
                    dtype=weighted.dtype,
                )
                weighted = weighted * factor
                # Preserve dynamic hard zeros through the final row
                # normalization.  Positive factor values are compatibility
                # weights; zero values are the hard forbidden support.
                effective_mask = effective_mask * (factor > 0).to(dtype=weighted.dtype)
        if self.transition_energy is not None:
            # Soft penalty: multiply by exp(-weight * energy).
            energy = self.transition_energy(context)
            if energy is not None:
                weighted = apply_transition_energy(
                    weighted,
                    transition_energy_matrix(
                        energy,
                        shape=(self.n_hidden_states, self.n_hidden_states),
                        dtype=weighted.dtype,
                        device=weighted.device,
                    ),
                    weight=self.energy_weight,
                )
        return project_matrix_rows(weighted, effective_mask)


class GraphSpectralGenerationHead(torch.nn.Module):
    """Compact-label signed WFA head initialized from graph spectral learning."""

    def __init__(
        self,
        *,
        label_count: int,
        state_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int = 0,
        initial=None,
        final=None,
        operators: Sequence[Any] | Mapping[Any, Any] | None = None,
        symbols: Sequence[Any] | None = None,
    ):
        """Initialize signed-WFA Torch head for compact-label decoding/training."""
        super().__init__()
        if label_count < 1 or state_count < 1:
            raise ValueError("label_count and state_count must be at least 1")
        self.label_count = int(label_count)
        self.state_count = int(state_count)
        self.pad_size = int(pad_size)
        self.symbols = tuple(symbols) if symbols is not None else tuple(range(self.label_count))
        if len(self.symbols) != self.label_count:
            raise ValueError("symbols length must match label_count")
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)
        generator = torch.Generator().manual_seed(int(random_seed))
        if initial is None:
            # Bias state 0 so random init starts with a mild anchor.
            initial_t = torch.randn(self.state_count, generator=generator) * 0.1
            initial_t[0] += 1.0
        else:
            initial_t = torch.as_tensor(initial, dtype=torch.float32)
        if final is None:
            final_t = torch.randn(self.state_count, generator=generator) * 0.1
        else:
            final_t = torch.as_tensor(final, dtype=torch.float32)
        if operators is None:
            operators_t = torch.randn(self.label_count, self.state_count, self.state_count, generator=generator) * 0.1
        else:
            if isinstance(operators, Mapping):
                # Preserve explicit symbol ordering when operators are keyed.
                operators_t = torch.stack([torch.as_tensor(operators[symbol], dtype=torch.float32) for symbol in self.symbols], dim=0)
            else:
                operators_t = torch.as_tensor(operators, dtype=torch.float32)
        _validate_wfa_shapes(initial_t, final_t, operators_t, self.label_count, self.state_count)
        self.initial = torch.nn.Parameter(initial_t, requires_grad=trainable)
        self.final = torch.nn.Parameter(final_t, requires_grad=trainable)
        self.operators = torch.nn.Parameter(operators_t, requires_grad=trainable)

    @classmethod
    def from_graph_spectral(
        cls,
        automaton,
        *,
        trainable: bool = True,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
    ) -> "GraphSpectralGenerationHead":
        automaton._require_fitted()
        return cls(
            label_count=len(automaton.id_to_symbol),
            state_count=int(automaton.initial.numel()),
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            trainable=trainable,
            initial=automaton.initial,
            final=automaton.final,
            operators=automaton.operators,
            symbols=automaton.id_to_symbol,
        )

    def token_id_for_label(self, label: int) -> int:
        """Resolve tokenizer id for a label; fail for non 1-to-1 mappings."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def trainable_parameter_names(self) -> list[str]:
        """List parameter names currently participating in gradient updates."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def sequence_log_probs(self, target_labels: torch.Tensor | Sequence[int], *, lengths=None) -> torch.Tensor:
        """Compute per-step log-probabilities for target label sequences."""
        labels, _lengths_t, step_mask, squeeze = _target_label_batch(target_labels, self.pad_size, lengths=lengths)
        outputs = []
        for row in labels:
            state = self.initial
            row_outputs = []
            for label_t in row:
                # Score each possible next label before advancing with target label.
                logits = torch.stack([state @ self.operators[label] @ self.final for label in range(self.label_count)])
                row_outputs.append(torch.log_softmax(logits, dim=-1))
                state = state @ self.operators[int(label_t.item())]
            outputs.append(torch.stack(row_outputs, dim=0))
        result = torch.stack(outputs, dim=0)
        # Zero padded timesteps so downstream losses can ignore them safely.
        result = result * step_mask.to(dtype=result.dtype).unsqueeze(-1)
        return result[0] if squeeze else result

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return unnormalized next-label scores for a token-id prefix."""
        prefix_labels = _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)
        state = self.initial
        for label in prefix_labels:
            state = state @ self.operators[label]
        return torch.stack([state @ self.operators[label] @ self.final for label in range(self.label_count)])

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """PMD module interface: returns sequence log-probabilities."""
        return self.sequence_log_probs(target_labels)


def _random_hmm_parameters(
    state_count: int,
    label_count: int,
    random_seed: int,
    transition_mask: torch.Tensor,
    emission_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random positive HMM parameters and project through legal masks."""
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.rand(state_count, generator=generator) + 0.1
    transition = torch.rand(state_count, state_count, generator=generator) + 0.1
    emission = torch.rand(state_count, label_count, generator=generator) + 0.1
    return (
        _normalize_vector(initial),
        project_matrix_rows(transition, transition_mask).to(dtype=torch.float32),
        project_matrix_rows(emission, emission_mask).to(dtype=torch.float32),
    )


def _safe_log(tensor: torch.Tensor) -> torch.Tensor:
    """Numerically safe log with dtype-aware epsilon floor."""
    return torch.log(tensor.clamp_min(torch.finfo(tensor.dtype).eps))


def _normalize_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a non-negative 1D vector with uniform fallback on zero sum."""
    if tensor.ndim != 1:
        raise ValueError("initial must be a rank-1 tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("initial must contain only finite values")
    tensor = tensor.clamp_min(0)
    total = tensor.sum()
    if total <= 0:
        return torch.full_like(tensor, 1.0 / tensor.numel())
    return tensor / total


def _validate_hmm_shapes(initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor, state_count: int, label_count: int) -> None:
    """Validate HMM parameter tensor shapes."""
    if tuple(initial.shape) != (state_count,):
        raise ValueError("initial shape must be [state_count]")
    if tuple(transition.shape) != (state_count, state_count):
        raise ValueError("transition shape must be [state_count, state_count]")
    if tuple(emission.shape) != (state_count, label_count):
        raise ValueError("emission shape must be [state_count, label_count]")


def _validate_wfa_shapes(initial: torch.Tensor, final: torch.Tensor, operators: torch.Tensor, label_count: int, state_count: int) -> None:
    """Validate signed-WFA parameter shapes and finiteness."""
    if tuple(initial.shape) != (state_count,):
        raise ValueError("initial shape must be [state_count]")
    if tuple(final.shape) != (state_count,):
        raise ValueError("final shape must be [state_count]")
    if tuple(operators.shape) != (label_count, state_count, state_count):
        raise ValueError("operators shape must be [label_count, state_count, state_count]")
    for name, tensor in (("initial", initial), ("final", final), ("operators", operators)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")


def _target_label_batch(target_labels: torch.Tensor | Sequence[int], pad_size: int, *, lengths=None):
    """Coerce labels to a 2D long batch and pad/truncate to ``pad_size``."""
    labels = torch.as_tensor(target_labels, dtype=torch.long)
    squeeze = labels.ndim == 1
    if squeeze:
        labels = labels.unsqueeze(0)
    if labels.ndim != 2:
        raise ValueError("target_labels must be rank 1 or 2")
    if labels.shape[1] > pad_size:
        labels = labels[:, :pad_size]
    if labels.shape[1] < pad_size:
        pad = torch.zeros((labels.shape[0], pad_size - labels.shape[1]), dtype=torch.long, device=labels.device)
        labels = torch.cat([labels, pad], dim=1)
    if (labels < 0).any():
        raise ValueError("target_labels must be non-negative")
    lengths_t = (
        torch.as_tensor(lengths, dtype=torch.long, device=labels.device)
        if lengths is not None
        else torch.full((labels.shape[0],), labels.shape[1], dtype=torch.long, device=labels.device)
    )
    if lengths_t.ndim != 1 or lengths_t.shape[0] != labels.shape[0]:
        raise ValueError("lengths must be a rank-1 tensor/sequence with one value per batch item")
    if (lengths_t < 0).any():
        raise ValueError("lengths must be non-negative")
    lengths_t = lengths_t.clamp(max=labels.shape[1])
    # step_mask[b, t] == True iff timestep t is valid for batch item b.
    step_mask = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)
    return labels, lengths_t, step_mask, squeeze


def _coerce_label_to_token_id(label_to_token_id: Sequence[int | None] | None, label_count: int) -> tuple[int | None, ...]:
    """Normalize optional label->token mapping and validate length."""
    if label_to_token_id is None:
        return tuple(range(label_count))
    values = tuple(None if value is None else int(value) for value in label_to_token_id)
    if len(values) != label_count:
        raise ValueError("label_to_token_id length must match label_count")
    return values


def _invert_label_to_token_id(label_to_token_id: Sequence[int | None]) -> dict[int, int]:
    """Build token->label lookup for decoding from tokenizer space."""
    return {int(token_id): label for label, token_id in enumerate(label_to_token_id) if token_id is not None}


def _labels_from_input_ids(input_ids: torch.Tensor | Sequence[int], token_id_to_label: Mapping[int, int], label_count: int) -> list[int]:
    """Map token ids to compact labels, validating label range."""
    ids = torch.as_tensor(input_ids, dtype=torch.long).flatten().tolist()
    labels = []
    for token_id in ids:
        label = token_id_to_label.get(int(token_id), int(token_id))
        labels.append(_validate_label(label, label_count))
    return labels


def _validate_label(label: int, label_count: int) -> int:
    """Ensure label index is in ``[0, label_count)``."""
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range")
    return label
