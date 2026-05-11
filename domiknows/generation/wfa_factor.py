"""Spectral WFA factor-graph helpers for DomiKnowS generation graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .automata import WeightedFiniteAutomaton
from .constraints import GenerationConstraint
from .encoder import GenerationBundle, GenerationGraphContext
from .vocabulary import TokenVocabulary


@dataclass
class SpectralWFAFactorGraphBundle(GenerationBundle):
    """Generation graph bundle with explicit spectral-WFA state structure."""

    wfa_state: object
    is_next_rel: object
    current_token: object
    next_token: object
    state_names: tuple[str, ...]
    wfa_transition_pair: object | None = None
    transition_pair_names: tuple[str, ...] = ()
    include_transition_pairs: bool = False


class SpectralWFAFactorGraphContext(GenerationGraphContext):
    """Generation graph context with helpers for WFA state predicates."""

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        generated_token,
        is_before_rel,
        first_token,
        second_token,
        wfa_state,
        is_next_rel,
        current_token,
        next_token,
        state_names: Sequence[str],
        wfa_transition_pair=None,
        transition_pair_names: Sequence[str] = (),
    ):
        super().__init__(vocabulary, generated_token, is_before_rel, first_token, second_token)
        self.wfa_state = wfa_state
        self.is_next_rel = is_next_rel
        self.current_token = current_token
        self.next_token = next_token
        self.state_names = tuple(state_names)
        self.wfa_transition_pair = wfa_transition_pair
        self.transition_pair_names = tuple(transition_pair_names)

    def wfa_state_value(self, state: str | int, variable: str, path=None):
        """Return a DomiKnowS predicate asserting a token has a WFA state feature."""
        state_concept = getattr(self.wfa_state, str(self.state_index(state)))
        if path is None:
            return state_concept(variable)
        return state_concept(variable, path=path)

    def transition_pair_value(self, from_state: str | int, to_state: str | int, variable: str, path=None):
        """Return a predicate asserting an adjacent relation has a WFA state pair."""
        if self.wfa_transition_pair is None:
            raise ValueError("wfa_transition_pair is only available with include_transition_pairs=True")
        pair_concept = getattr(self.wfa_transition_pair, str(self.transition_pair_index(from_state, to_state)))
        if path is None:
            return pair_concept(variable)
        return pair_concept(variable, path=path)

    def state_index(self, state: str | int) -> int:
        """Resolve a state name or index to an integer enum value."""
        if isinstance(state, int):
            index = int(state)
        else:
            try:
                index = self.state_names.index(str(state))
            except ValueError as exc:
                raise KeyError(f"unknown WFA state {state!r}") from exc
        if index < 0 or index >= len(self.state_names):
            raise IndexError(f"WFA state index {index} is out of range")
        return index

    def transition_pair_index(self, from_state: str | int, to_state: str | int) -> int:
        """Resolve a ``from -> to`` state pair to the flattened transition enum index."""
        return self.state_index(from_state) * len(self.state_names) + self.state_index(to_state)


class SpectralWFAFactorGraphEncoder:
    """Build a generation graph with explicit spectral-WFA state factors."""

    def __init__(
        self,
        vocab: Sequence[str],
        eos_token: str,
        *,
        state_names: Sequence[str] | None = None,
        state_count: int | None = None,
        graph_name: str = "wfa_factor_generation",
        tokenizer: object | None = None,
        clear_graph: bool = True,
        include_transition_pairs: bool = False,
    ):
        self.vocabulary = TokenVocabulary(vocab, eos_token=eos_token, tokenizer=tokenizer)
        self.state_names = _resolve_state_names(state_names, state_count)
        self.graph_name = graph_name
        self.clear_graph = clear_graph
        self.include_transition_pairs = bool(include_transition_pairs)

    def build_graph(
        self,
        constraints: Sequence[GenerationConstraint] = (),
    ) -> tuple[object, SpectralWFAFactorGraphBundle]:
        """Construct the opt-in WFA factor graph and compile generation constraints."""
        from domiknows.graph import Concept, EnumConcept, Graph, Relation

        if self.clear_graph:
            Graph.clear()
            Concept.clear()
            Relation.clear()

        with Graph(self.graph_name) as graph:
            text = Concept(name="text")
            token = Concept(name="token")
            contains, = text.contains(token)

            is_before_rel = Concept(name="is_before_rel")
            first_token, second_token = is_before_rel.has_a(arg1=token, arg2=token)

            is_next_rel = Concept(name="is_next_rel")
            current_token, next_token = is_next_rel.has_a(arg1=token, arg2=token)

            generated_token = token(
                name="generated_token",
                ConceptClass=EnumConcept,
                values=[str(i) for i in range(self.vocabulary.label_count)],
            )
            wfa_state = token(
                name="wfa_state",
                ConceptClass=EnumConcept,
                values=[str(i) for i in range(len(self.state_names))],
            )

            transition_pair = None
            transition_pair_names: tuple[str, ...] = ()
            if self.include_transition_pairs:
                transition_pair_names = tuple(
                    f"{from_state}->{to_state}"
                    for from_state in self.state_names
                    for to_state in self.state_names
                )
                transition_pair = is_next_rel(
                    name="wfa_transition_pair",
                    ConceptClass=EnumConcept,
                    values=[str(i) for i in range(len(transition_pair_names))],
                )

            context = SpectralWFAFactorGraphContext(
                self.vocabulary,
                generated_token,
                is_before_rel,
                first_token,
                second_token,
                wfa_state,
                is_next_rel,
                current_token,
                next_token,
                self.state_names,
                wfa_transition_pair=transition_pair,
                transition_pair_names=transition_pair_names,
            )
            for constraint in constraints:
                if constraint.supports_domiknows:
                    constraint.apply_domiknows(context)

        bundle = SpectralWFAFactorGraphBundle(
            text=text,
            token=token,
            contains=contains,
            generated_token=generated_token,
            is_before_rel=is_before_rel,
            first_token=first_token,
            second_token=second_token,
            context=context,
            constraints=tuple(constraints),
            vocabulary=self.vocabulary,
            wfa_state=wfa_state,
            is_next_rel=is_next_rel,
            current_token=current_token,
            next_token=next_token,
            state_names=tuple(self.state_names),
            wfa_transition_pair=transition_pair,
            transition_pair_names=transition_pair_names,
            include_transition_pairs=self.include_transition_pairs,
        )
        return graph, bundle


class SpectralWFAFactorGraphHead(torch.nn.Module):
    """Differentiable signed WFA exposing graph-visible normalized projections."""

    def __init__(
        self,
        model: WeightedFiniteAutomaton | None = None,
        *,
        label_count: int | None = None,
        state_names: Sequence[str] | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int = 0,
    ):
        super().__init__()
        self.label_count = _resolve_label_count(model, label_count)
        self.state_names = _resolve_state_names(state_names, state_count or _model_state_count(model))
        self.state_count = len(self.state_names)
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = {
            int(token_id): label for label, token_id in enumerate(self.label_to_token_id) if token_id is not None
        }

        if model is None:
            initial, transitions, final = _random_wfa_parameters(self.state_count, self.label_count, random_seed)
        else:
            initial = torch.as_tensor(model.initial, dtype=torch.float32)
            final = torch.as_tensor(model.final, dtype=torch.float32)
            transitions = torch.stack(
                [torch.as_tensor(model.transitions[symbol], dtype=torch.float32) for symbol in model.symbols],
                dim=0,
            )
            _validate_wfa_shapes(initial, transitions, final, self.state_count, self.label_count)

        self.initial = torch.nn.Parameter(initial, requires_grad=trainable)
        self.transitions = torch.nn.Parameter(transitions, requires_grad=trainable)
        self.final = torch.nn.Parameter(final, requires_grad=trainable)

    def generated_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for generated tokens."""
        return _WFAFactorGeneratedProjection(self)

    def state_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for WFA state features."""
        return _WFAFactorStateProjection(self)

    def transition_pair_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for adjacent WFA factors."""
        return _WFAFactorTransitionPairProjection(self)

    def token_id_for_label(self, label: int) -> int:
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return next-step signed WFA scores over compact generation labels."""
        labels = self._labels_from_input_ids(input_ids)
        state = self._prefix_state(labels)
        next_states = torch.einsum("s,lsd->ld", state, self.transitions)
        return torch.matmul(next_states, self.final)

    def generated_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return normalized teacher-forced label scores shaped ``[T, label_count]``."""
        return self.sequence_log_probs(target_labels)

    def state_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return normalized WFA prefix-state projections shaped ``[T, state_count]``."""
        states = self.prefix_states(target_labels)
        return torch.log_softmax(states, dim=-1)

    def transition_pair_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return normalized adjacent transition-pair projections shaped ``[T - 1, S ** 2]``."""
        pair_scores = self.transition_pair_scores(target_labels).reshape(-1, self.state_count * self.state_count)
        return torch.log_softmax(pair_scores, dim=-1)

    def sequence_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return normalized teacher-forced next-label scores shaped ``[T, label_count]``."""
        labels = self._target_labels(target_labels)
        state = self.initial
        outputs = []
        for label in labels:
            next_states = torch.einsum("s,lsd->ld", state, self.transitions)
            outputs.append(torch.matmul(next_states, self.final))
            state = torch.matmul(state, self.transitions[int(label.item())])
        return torch.log_softmax(torch.stack(outputs, dim=0), dim=-1)

    def prefix_states(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return signed WFA prefix states after each observed label."""
        labels = self._target_labels(target_labels)
        states = []
        state = self.initial
        for label in labels:
            state = torch.matmul(state, self.transitions[int(label.item())])
            states.append(state)
        return torch.stack(states, dim=0)

    def transition_pair_scores(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return signed adjacent pair scores shaped ``[T - 1, state_count, state_count]``."""
        labels = self._target_labels(target_labels)
        if labels.numel() < 2:
            return torch.empty(
                (0, self.state_count, self.state_count),
                dtype=self.transitions.dtype,
                device=self.transitions.device,
            )
        prefix_states = self.prefix_states(labels)
        pairs = []
        for step in range(labels.numel() - 1):
            transition = self.transitions[int(labels[step + 1].item())]
            pairs.append(prefix_states[step].unsqueeze(1) * transition)
        return torch.stack(pairs, dim=0)

    def production_wfa(self) -> WeightedFiniteAutomaton:
        """Return a Torch-backed WFA view of current factor-head parameters."""
        return WeightedFiniteAutomaton(
            self.initial,
            self.transitions,
            self.final,
            tuple(range(self.label_count)),
            state_names=self.state_names,
        )

    def sequence_energy_loss(self, target_labels: torch.Tensor | Sequence[int], reduction: str = "mean") -> torch.Tensor:
        """Return cross-entropy over signed WFA next-label scores."""
        labels = self._target_labels(target_labels)
        return F.nll_loss(self.sequence_log_probs(labels), labels, reduction=reduction)

    def consistency_loss(self, target_labels: torch.Tensor | Sequence[int], reduction: str = "mean") -> torch.Tensor:
        """Return a differentiable diagnostic tying state and pair projections."""
        states = self.state_log_probs(target_labels).exp()
        pair = self.transition_pair_log_probs(target_labels).exp().reshape(-1, self.state_count, self.state_count)
        losses = [states.new_zeros((1,))]
        if pair.numel():
            losses.append((pair.sum(dim=2) - states[:-1]).pow(2).mean(dim=-1))
            losses.append((pair.sum(dim=1) - states[1:]).pow(2).mean(dim=-1))
        flat = torch.cat([loss.reshape(-1) for loss in losses], dim=0)
        if reduction == "none":
            return flat
        if reduction == "sum":
            return flat.sum()
        if reduction == "mean":
            return flat.mean()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        return self.generated_log_probs(target_labels)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of trainable WFA parameters."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _prefix_state(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        state = self.initial
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            state = torch.matmul(state, self.transitions[label])
        return state

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        if isinstance(input_ids, torch.Tensor):
            raw_ids = input_ids.detach().reshape(-1).tolist()
        else:
            raw_ids = list(input_ids)
        return [self._token_id_to_label[int(token_id)] for token_id in raw_ids if int(token_id) in self._token_id_to_label]

    def _target_labels(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if isinstance(target_labels, torch.Tensor):
            labels = target_labels.detach().long().reshape(-1).to(self.initial.device)
        else:
            labels = torch.tensor(list(target_labels), dtype=torch.long, device=self.initial.device)
        if labels.numel() >= self.pad_size:
            labels = labels[: self.pad_size]
        else:
            padding = torch.zeros(self.pad_size - labels.numel(), dtype=torch.long, device=labels.device)
            labels = torch.cat([labels, padding], dim=0)
        if labels.numel() == 0:
            raise ValueError("target_labels must contain at least one label")
        if torch.any((labels < 0) | (labels >= self.label_count)):
            raise ValueError("target_labels contain labels outside the compact vocabulary")
        return labels


class _WFAFactorGeneratedProjection(torch.nn.Module):
    def __init__(self, head: SpectralWFAFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.generated_log_probs(target_labels)

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return self.head.next_label_logits(input_ids)

    def token_id_for_label(self, label: int) -> int:
        return self.head.token_id_for_label(label)

    @property
    def pad_size(self) -> int:
        return self.head.pad_size


class _WFAFactorStateProjection(torch.nn.Module):
    def __init__(self, head: SpectralWFAFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.state_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return self.head.pad_size


class _WFAFactorTransitionPairProjection(torch.nn.Module):
    def __init__(self, head: SpectralWFAFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.transition_pair_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return max(0, self.head.pad_size - 1)


def wfa_factor_sequence_energy_loss(
    head: SpectralWFAFactorGraphHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Energy-style supervised loss for a spectral-WFA factor graph head."""
    if not isinstance(head, SpectralWFAFactorGraphHead):
        raise TypeError("wfa_factor_sequence_energy_loss expects a SpectralWFAFactorGraphHead")
    return head.sequence_energy_loss(target_labels, reduction=reduction)


def wfa_factor_consistency_loss(
    head: SpectralWFAFactorGraphHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Diagnostic loss tying graph-visible WFA state and transition projections."""
    if not isinstance(head, SpectralWFAFactorGraphHead):
        raise TypeError("wfa_factor_consistency_loss expects a SpectralWFAFactorGraphHead")
    return head.consistency_loss(target_labels, reduction=reduction)


def apply_wfa_factor_consistency_constraints(bundle: SpectralWFAFactorGraphBundle) -> None:
    """Add weak PMD-visible rules for graph-exposed WFA transition-pair factors."""
    if not bundle.include_transition_pairs:
        raise ValueError("WFA factor consistency constraints require include_transition_pairs=True")
    from domiknows.graph.logicalConstrain import andL, ifL

    ctx = bundle.context
    for from_state in bundle.state_names:
        for to_state in bundle.state_names:
            ifL(
                ctx.is_next_rel("next"),
                ifL(
                    ctx.transition_pair_value(from_state, to_state, "next"),
                    andL(
                        ctx.wfa_state_value(from_state, "x", path=("next", ctx.current_token)),
                        ctx.wfa_state_value(to_state, "y", path=("next", ctx.next_token)),
                    ),
                ),
            )


def _resolve_state_names(state_names: Sequence[str] | None, state_count: int | None) -> tuple[str, ...]:
    if state_names is not None:
        names = tuple(str(name) for name in state_names)
        if not names:
            raise ValueError("state_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("state_names must be unique")
        if state_count is not None and int(state_count) != len(names):
            raise ValueError("state_count must match state_names length")
        return names
    if state_count is None:
        raise ValueError("either state_names or state_count is required")
    count = _positive_int(state_count, "state_count")
    return tuple(f"S{i}" for i in range(count))


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _resolve_label_count(model: WeightedFiniteAutomaton | None, label_count: int | None) -> int:
    if model is not None:
        count = len(model.symbols)
        if label_count is not None and int(label_count) != count:
            raise ValueError("label_count must match model symbol count")
        return count
    if label_count is None:
        raise ValueError("label_count is required when model is not supplied")
    return _positive_int(label_count, "label_count")


def _model_state_count(model: WeightedFiniteAutomaton | None) -> int | None:
    return None if model is None else int(model.state_count)


def _random_wfa_parameters(state_count: int, label_count: int, random_seed: int):
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.randn(state_count, generator=generator) * 0.1
    transitions = torch.randn(label_count, state_count, state_count, generator=generator) * 0.1
    final = torch.randn(state_count, generator=generator) * 0.1
    return initial, transitions, final


def _validate_wfa_shapes(
    initial: torch.Tensor,
    transitions: torch.Tensor,
    final: torch.Tensor,
    state_count: int,
    label_count: int,
) -> None:
    if initial.shape != (state_count,):
        raise ValueError("WFA initial vector shape does not match state_count")
    if final.shape != (state_count,):
        raise ValueError("WFA final vector shape does not match state_count")
    if transitions.shape != (label_count, state_count, state_count):
        raise ValueError("WFA transition tensor shape must be [label_count, state_count, state_count]")


def _coerce_label_to_token_id(
    label_to_token_id: Sequence[int | None] | None,
    label_count: int,
) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)


def _validate_label(label: int, label_count: int) -> int:
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range")
    return label
