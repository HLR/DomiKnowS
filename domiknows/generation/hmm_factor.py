"""HMM factor-graph helpers for DomiKnowS generation graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .constraints import GenerationConstraint
from .encoder import GenerationBundle, GenerationGraphContext
from .vocabulary import TokenVocabulary


@dataclass
class HMMFactorGraphBundle(GenerationBundle):
    """Generation graph bundle with explicit HMM latent-state structure."""

    latent_state: object
    is_next_rel: object
    current_token: object
    next_token: object
    state_names: tuple[str, ...]


class HMMFactorGraphContext(GenerationGraphContext):
    """Generation graph context with helpers for HMM hidden-state predicates."""

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        generated_token,
        is_before_rel,
        first_token,
        second_token,
        latent_state,
        is_next_rel,
        current_token,
        next_token,
        state_names: Sequence[str],
    ):
        super().__init__(vocabulary, generated_token, is_before_rel, first_token, second_token)
        self.latent_state = latent_state
        self.is_next_rel = is_next_rel
        self.current_token = current_token
        self.next_token = next_token
        self.state_names = tuple(state_names)

    def latent_state_value(self, state: str | int, variable: str, path=None):
        """Return a DomiKnowS predicate asserting a token has a latent state."""
        state_concept = getattr(self.latent_state, str(self.state_index(state)))
        if path is None:
            return state_concept(variable)
        return state_concept(variable, path=path)

    def state_index(self, state: str | int) -> int:
        """Resolve a state name or index to an integer enum value."""
        if isinstance(state, int):
            index = int(state)
        else:
            try:
                index = self.state_names.index(str(state))
            except ValueError as exc:
                raise KeyError(f"unknown latent state {state!r}") from exc
        if index < 0 or index >= len(self.state_names):
            raise IndexError(f"latent state index {index} is out of range")
        return index


class HMMFactorGraphEncoder:
    """Build a generation graph with explicit HMM latent-state factors."""

    def __init__(
        self,
        vocab: Sequence[str],
        eos_token: str,
        *,
        state_names: Sequence[str] | None = None,
        state_count: int | None = None,
        graph_name: str = "hmm_factor_generation",
        tokenizer: object | None = None,
        clear_graph: bool = True,
    ):
        self.vocabulary = TokenVocabulary(vocab, eos_token=eos_token, tokenizer=tokenizer)
        self.state_names = _resolve_state_names(state_names, state_count)
        self.graph_name = graph_name
        self.clear_graph = clear_graph

    def build_graph(self, constraints: Sequence[GenerationConstraint] = ()) -> tuple[object, HMMFactorGraphBundle]:
        """Construct the opt-in HMM factor graph and compile generation constraints."""
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
            latent_state = token(
                name="latent_state",
                ConceptClass=EnumConcept,
                values=[str(i) for i in range(len(self.state_names))],
            )

            context = HMMFactorGraphContext(
                self.vocabulary,
                generated_token,
                is_before_rel,
                first_token,
                second_token,
                latent_state,
                is_next_rel,
                current_token,
                next_token,
                self.state_names,
            )
            for constraint in constraints:
                if constraint.supports_domiknows:
                    constraint.apply_domiknows(context)

        bundle = HMMFactorGraphBundle(
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
            latent_state=latent_state,
            is_next_rel=is_next_rel,
            current_token=current_token,
            next_token=next_token,
            state_names=tuple(self.state_names),
        )
        return graph, bundle


class HMMFactorGraphHead(torch.nn.Module):
    """Differentiable HMM that exposes generated-token and latent-state marginals."""

    def __init__(
        self,
        *,
        label_count: int,
        state_names: Sequence[str] | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int = 0,
    ):
        super().__init__()
        self.label_count = _positive_int(label_count, "label_count")
        self.state_names = _resolve_state_names(state_names, state_count)
        self.state_count = len(self.state_names)
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = {
            int(token_id): label for label, token_id in enumerate(self.label_to_token_id) if token_id is not None
        }

        generator = torch.Generator().manual_seed(int(random_seed))
        initial = torch.rand(self.state_count, generator=generator) + 0.1
        transition = torch.rand(self.state_count, self.state_count, generator=generator) + 0.1
        emission = torch.rand(self.state_count, self.label_count, generator=generator) + 0.1
        self.initial_logits = torch.nn.Parameter(torch.log(initial / initial.sum()), requires_grad=trainable)
        self.transition_logits = torch.nn.Parameter(
            torch.log(transition / transition.sum(dim=-1, keepdim=True)),
            requires_grad=trainable,
        )
        self.emission_logits = torch.nn.Parameter(
            torch.log(emission / emission.sum(dim=-1, keepdim=True)),
            requires_grad=trainable,
        )

    @property
    def initial_probs(self) -> torch.Tensor:
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        return torch.softmax(self.emission_logits, dim=-1)

    def generated_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for generated tokens."""
        return _HMMFactorGeneratedProjection(self)

    def latent_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for latent states."""
        return _HMMFactorLatentProjection(self)

    def token_id_for_label(self, label: int) -> int:
        label = int(label)
        if label < 0 or label >= self.label_count:
            raise ValueError(f"label {label} is out of range")
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return next-step generated-token logits for DFA-constrained decoding."""
        prefix = self._labels_from_input_ids(input_ids)
        state = self.initial_probs
        transition = self.transition_probs
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        for label in prefix:
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)
        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def generated_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return log P(y_t | observed sequence) shaped ``[T, label_count]``."""
        gamma = self.latent_marginals(target_labels)
        generated_probs = torch.matmul(gamma, self.emission_probs)
        return torch.log(generated_probs.clamp_min(torch.finfo(generated_probs.dtype).eps))

    def latent_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return log P(z_t | observed sequence) shaped ``[T, state_count]``."""
        gamma = self.latent_marginals(target_labels)
        return torch.log(gamma.clamp_min(torch.finfo(gamma.dtype).eps))

    def latent_marginals(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Compute differentiable posterior state marginals with scaled forward/backward."""
        labels = self._target_labels(target_labels)
        alpha, scales = self._forward_scaled(labels)
        beta = self._backward_scaled(labels, scales)
        gamma = alpha * beta
        return gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(gamma.dtype).eps)

    def sequence_nll(self, target_labels: torch.Tensor | Sequence[int], reduction: str = "mean") -> torch.Tensor:
        """Return negative log-likelihood for the observed compact-label sequence."""
        labels = self._target_labels(target_labels)
        _alpha, scales = self._forward_scaled(labels)
        nll = -torch.log(scales.clamp_min(torch.finfo(scales.dtype).eps)).sum()
        if reduction == "sum":
            return nll
        if reduction == "mean":
            return nll / labels.numel()
        if reduction == "none":
            return nll
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def trainable_parameter_names(self) -> list[str]:
        """Return names of trainable HMM parameters."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _forward_scaled(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transition_probs
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        alphas = []
        scales = []
        alpha = self.initial_probs * emission[:, int(labels[0].item())]
        scale = alpha.sum().clamp_min(eps)
        alpha = alpha / scale
        alphas.append(alpha)
        scales.append(scale)
        for label in labels[1:]:
            alpha = torch.matmul(alpha, transition) * emission[:, int(label.item())]
            scale = alpha.sum().clamp_min(eps)
            alpha = alpha / scale
            alphas.append(alpha)
            scales.append(scale)
        return torch.stack(alphas, dim=0), torch.stack(scales, dim=0)

    def _backward_scaled(self, labels: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        transition = self.transition_probs
        emission = self.emission_probs
        betas = [None] * labels.numel()
        beta = torch.ones(self.state_count, dtype=emission.dtype, device=emission.device)
        betas[-1] = beta
        for t in range(labels.numel() - 2, -1, -1):
            next_emission = emission[:, int(labels[t + 1].item())]
            beta = torch.matmul(transition, next_emission * beta) / scales[t + 1].clamp_min(
                torch.finfo(emission.dtype).eps
            )
            betas[t] = beta
        return torch.stack(betas, dim=0)

    def _target_labels(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if isinstance(target_labels, torch.Tensor):
            labels = target_labels.detach().long().reshape(-1).to(self.initial_logits.device)
        else:
            labels = torch.tensor(list(target_labels), dtype=torch.long, device=self.initial_logits.device)
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

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        if isinstance(input_ids, torch.Tensor):
            raw_ids = input_ids.detach().reshape(-1).tolist()
        else:
            raw_ids = list(input_ids)
        return [self._token_id_to_label[int(token_id)] for token_id in raw_ids if int(token_id) in self._token_id_to_label]


class _HMMFactorGeneratedProjection(torch.nn.Module):
    def __init__(self, head: HMMFactorGraphHead):
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


class _HMMFactorLatentProjection(torch.nn.Module):
    def __init__(self, head: HMMFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.latent_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return self.head.pad_size


def hmm_factor_sequence_nll(
    head: HMMFactorGraphHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood helper for an HMM factor graph head."""
    if not isinstance(head, HMMFactorGraphHead):
        raise TypeError("hmm_factor_sequence_nll expects an HMMFactorGraphHead")
    return head.sequence_nll(target_labels, reduction=reduction)


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


def _coerce_label_to_token_id(
    label_to_token_id: Sequence[int | None] | None,
    label_count: int,
) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)
