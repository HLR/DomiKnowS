"""Torch generation heads backed by HMMs and weighted finite automata.

These modules expose the small interface used by DomiKnowS ``ModuleLearner``
in generation graphs: ``forward(_contains, instruction_tokens, target_labels)``
returns log-probabilities shaped ``[seq_len, label_count]`` for the
``generated_token`` concept.  They also expose ``next_label_logits`` so the
same trained head can be decoded with the label-level DFA decoder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .automata import DFA, ProbabilisticAutomaton, WeightedFiniteAutomaton


class HMMGenerationHead(torch.nn.Module):
    """Compact-label generation head backed by a discrete HMM.

    The HMM is parameterized with trainable logits when ``trainable=True`` and
    frozen tensors otherwise.  Emissions are labels in the compact
    ``GenerationEncoder`` vocabulary, so the output can be attached directly to
    ``token[generated_token]`` via ``ModuleLearner``.
    """

    def __init__(
        self,
        model: ProbabilisticAutomaton | None = None,
        *,
        label_count: int | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = False,
        random_seed: int = 0,
    ):
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _resolve_label_count(model, label_count)
        self.state_count = _resolve_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        if model is None:
            initial, transition, emission = _random_hmm_parameters(
                self.state_count,
                self.label_count,
                random_seed,
            )
        else:
            initial = torch.tensor(model.initial, dtype=torch.float32)
            transition = torch.tensor(model.transition, dtype=torch.float32)
            emission = torch.tensor(model.emission, dtype=torch.float32)
            _validate_hmm_shapes(initial, transition, emission, self.state_count, self.label_count)

        self.initial_logits = torch.nn.Parameter(_safe_log(initial), requires_grad=trainable)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission), requires_grad=trainable)

    @property
    def initial_probs(self) -> torch.Tensor:
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        return torch.softmax(self.emission_logits, dim=-1)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _next_logits_from_prefix_labels(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        state = self.initial_probs
        transition = self.transition_probs
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps

        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)

        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return next-step logits over compact generation labels."""
        return self._next_logits_from_prefix_labels(self._labels_from_input_ids(input_ids))

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        labels = _target_labels(target_labels, self.pad_size, device=self.initial_logits.device)
        generated = []
        prefix: list[int] = []
        for step in range(self.pad_size):
            generated.append(self._next_logits_from_prefix_labels(prefix))
            prefix.append(int(labels[step].item()))
        return torch.log_softmax(torch.stack(generated, dim=0), dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


class SpectralWFAGenerationHead(torch.nn.Module):
    """Compact-label generation head backed by a signed weighted automaton.

    WFA sequence scores may be signed, so next-symbol scores are treated as
    logits rather than probabilities.  This keeps spectral outputs source
    compatible with DomiKnowS concept losses without forcing stochastic
    normalization.
    """

    def __init__(
        self,
        model: WeightedFiniteAutomaton | None = None,
        *,
        label_count: int | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = False,
        random_seed: int = 0,
    ):
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _resolve_wfa_label_count(model, label_count)
        self.state_count = _resolve_wfa_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        if model is None:
            initial, transitions, final = _random_wfa_parameters(
                self.state_count,
                self.label_count,
                random_seed,
            )
        else:
            initial = torch.tensor(model.initial, dtype=torch.float32)
            final = torch.tensor(model.final, dtype=torch.float32)
            transitions = torch.stack(
                [torch.tensor(model.transitions[symbol], dtype=torch.float32) for symbol in model.symbols],
                dim=0,
            )
            _validate_wfa_shapes(initial, transitions, final, self.state_count, self.label_count)

        self.initial = torch.nn.Parameter(initial, requires_grad=trainable)
        self.transitions = torch.nn.Parameter(transitions, requires_grad=trainable)
        self.final = torch.nn.Parameter(final, requires_grad=trainable)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _prefix_state(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        state = self.initial
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            state = torch.matmul(state, self.transitions[label])
        return state

    def _next_logits_from_prefix_labels(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        state = self._prefix_state(prefix_labels)
        next_states = torch.einsum("s,lsd->ld", state, self.transitions)
        return torch.matmul(next_states, self.final)

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return next-step signed WFA scores over compact labels."""
        return self._next_logits_from_prefix_labels(self._labels_from_input_ids(input_ids))

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        labels = _target_labels(target_labels, self.pad_size, device=self.initial.device)
        generated = []
        prefix: list[int] = []
        for step in range(self.pad_size):
            generated.append(self._next_logits_from_prefix_labels(prefix))
            prefix.append(int(labels[step].item()))
        return torch.log_softmax(torch.stack(generated, dim=0), dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


def hmm_sequence_nll(
    head: HMMGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of a target label sequence under an HMM head."""
    if not isinstance(head, HMMGenerationHead):
        raise TypeError("hmm_sequence_nll expects an HMMGenerationHead")
    labels = _target_labels(target_labels, head.pad_size, device=head.initial_logits.device)
    log_probs = head(None, torch.empty((1, 0), dtype=torch.long, device=labels.device), labels)
    return F.nll_loss(log_probs, labels, reduction=reduction)


def wfa_sequence_energy_loss(
    head: SpectralWFAGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Energy-style supervised loss for a WFA head.

    Signed WFA next-symbol scores are interpreted as logits and optimized with
    cross-entropy against the target compact labels.
    """
    if not isinstance(head, SpectralWFAGenerationHead):
        raise TypeError("wfa_sequence_energy_loss expects a SpectralWFAGenerationHead")
    labels = _target_labels(target_labels, head.pad_size, device=head.initial.device)
    log_probs = head(None, torch.empty((1, 0), dtype=torch.long, device=labels.device), labels)
    return F.nll_loss(log_probs, labels, reduction=reduction)


def allowed_mass_loss(
    probs: torch.Tensor,
    dfa: DFA,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Softly encourage probability mass on labels allowed by a DFA.

    The DFA state is advanced along the greedy label path for each sequence.
    The loss remains differentiable with respect to the probability mass placed
    on labels that are valid at those visited states.  This is an auxiliary
    training signal; hard correctness still comes from DFA decoding.
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if probs.dim() == 2:
        batched = probs.unsqueeze(0)
        squeeze = True
    elif probs.dim() == 3:
        batched = probs
        squeeze = False
    else:
        raise ValueError("probs must have shape [seq_len, labels] or [batch, seq_len, labels]")

    losses = []
    eps = torch.finfo(batched.dtype).eps
    label_count = batched.shape[-1]
    for batch_idx in range(batched.shape[0]):
        state = dfa.start_state
        step_losses = []
        for step_idx in range(batched.shape[1]):
            allowed = sorted(
                int(label)
                for label in dfa.allowed_tokens(state, remaining_steps=batched.shape[1] - step_idx)
                if 0 <= int(label) < label_count
            )
            if allowed:
                allowed_index = torch.tensor(allowed, dtype=torch.long, device=batched.device)
                mass = batched[batch_idx, step_idx].index_select(0, allowed_index).sum()
                step_losses.append(-torch.log(mass.clamp_min(eps)))
            label = int(torch.argmax(batched[batch_idx, step_idx]).item())
            next_state = dfa.step(state, label)
            if next_state is not None:
                state = next_state
        if step_losses:
            losses.append(torch.stack(step_losses).mean())
        else:
            losses.append(batched.new_zeros(()))

    result = torch.stack(losses)
    if squeeze and reduction == "none":
        return result[0]
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    return result.mean()


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _resolve_label_count(model: ProbabilisticAutomaton | None, label_count: int | None) -> int:
    if model is not None:
        inferred = len(model.symbols)
        if label_count is not None and int(label_count) != inferred:
            raise ValueError("label_count does not match HMM symbol count")
        return inferred
    if label_count is None:
        raise ValueError("label_count is required when no HMM model is supplied")
    return _positive_int(label_count, "label_count")


def _resolve_state_count(model: ProbabilisticAutomaton | None, state_count: int | None) -> int:
    if model is not None:
        inferred = model.state_count
        if state_count is not None and int(state_count) != inferred:
            raise ValueError("state_count does not match HMM state count")
        return inferred
    if state_count is None:
        raise ValueError("state_count is required when no HMM model is supplied")
    return _positive_int(state_count, "state_count")


def _resolve_wfa_label_count(model: WeightedFiniteAutomaton | None, label_count: int | None) -> int:
    if model is not None:
        inferred = len(model.symbols)
        if label_count is not None and int(label_count) != inferred:
            raise ValueError("label_count does not match WFA symbol count")
        return inferred
    if label_count is None:
        raise ValueError("label_count is required when no WFA model is supplied")
    return _positive_int(label_count, "label_count")


def _resolve_wfa_state_count(model: WeightedFiniteAutomaton | None, state_count: int | None) -> int:
    if model is not None:
        inferred = model.state_count
        if state_count is not None and int(state_count) != inferred:
            raise ValueError("state_count does not match WFA state count")
        return inferred
    if state_count is None:
        raise ValueError("state_count is required when no WFA model is supplied")
    return _positive_int(state_count, "state_count")


def _coerce_label_to_token_id(
    label_to_token_id: Sequence[int | None] | None,
    label_count: int,
) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)


def _invert_label_to_token_id(label_to_token_id: Sequence[int | None]) -> Mapping[int, int]:
    return {int(token_id): label for label, token_id in enumerate(label_to_token_id) if token_id is not None}


def _labels_from_input_ids(
    input_ids: torch.Tensor | Sequence[int],
    token_id_to_label: Mapping[int, int],
    label_count: int,
) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        flat = input_ids.detach().reshape(-1).tolist()
    else:
        flat = list(input_ids)
    labels = []
    for token_id in flat:
        label = token_id_to_label.get(int(token_id))
        if label is not None and 0 <= label < label_count:
            labels.append(int(label))
    return labels


def _target_labels(
    target_labels: torch.Tensor | Sequence[int],
    pad_size: int,
    *,
    device: torch.device,
    eos_label: int = 0,
) -> torch.Tensor:
    if isinstance(target_labels, torch.Tensor):
        labels = target_labels.detach().long().reshape(-1).to(device)
    else:
        labels = torch.tensor(list(target_labels), dtype=torch.long, device=device)
    if labels.numel() >= pad_size:
        return labels[:pad_size]
    padding = torch.full((pad_size - labels.numel(),), int(eos_label), dtype=torch.long, device=device)
    return torch.cat([labels, padding], dim=0)


def _validate_label(label: int, label_count: int) -> int:
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range for {label_count} labels")
    return label


def _safe_log(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values.float().clamp_min(torch.finfo(torch.float32).eps))


def _random_hmm_parameters(state_count: int, label_count: int, random_seed: int):
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.rand(state_count, generator=generator) + 0.1
    transition = torch.rand(state_count, state_count, generator=generator) + 0.1
    emission = torch.rand(state_count, label_count, generator=generator) + 0.1
    initial = initial / initial.sum()
    transition = transition / transition.sum(dim=-1, keepdim=True)
    emission = emission / emission.sum(dim=-1, keepdim=True)
    return initial, transition, emission


def _random_wfa_parameters(state_count: int, label_count: int, random_seed: int):
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.randn(state_count, generator=generator) * 0.1
    transitions = torch.randn(label_count, state_count, state_count, generator=generator) * 0.1
    final = torch.randn(state_count, generator=generator) * 0.1
    return initial, transitions, final


def _validate_hmm_shapes(initial, transition, emission, state_count: int, label_count: int) -> None:
    if initial.shape != (state_count,):
        raise ValueError("HMM initial vector has the wrong shape")
    if transition.shape != (state_count, state_count):
        raise ValueError("HMM transition matrix has the wrong shape")
    if emission.shape != (state_count, label_count):
        raise ValueError("HMM emission matrix has the wrong shape")


def _validate_wfa_shapes(initial, transitions, final, state_count: int, label_count: int) -> None:
    if initial.shape != (state_count,):
        raise ValueError("WFA initial vector has the wrong shape")
    if transitions.shape != (label_count, state_count, state_count):
        raise ValueError("WFA transition tensor has the wrong shape")
    if final.shape != (state_count,):
        raise ValueError("WFA final vector has the wrong shape")
