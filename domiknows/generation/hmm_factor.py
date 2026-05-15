"""HMM factor-graph helpers for DomiKnowS generation graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .automata import DiscreteHMM
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
    forward_state: object | None = None
    backward_state: object | None = None
    transition_pair: object | None = None
    transition_pair_names: tuple[str, ...] = ()
    include_dp_factors: bool = False


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
        forward_state=None,
        backward_state=None,
        transition_pair=None,
        transition_pair_names: Sequence[str] = (),
    ):
        super().__init__(vocabulary, generated_token, is_before_rel, first_token, second_token)
        self.latent_state = latent_state
        self.is_next_rel = is_next_rel
        self.current_token = current_token
        self.next_token = next_token
        self.state_names = tuple(state_names)
        self.forward_state = forward_state
        self.backward_state = backward_state
        self.transition_pair = transition_pair
        self.transition_pair_names = tuple(transition_pair_names)

    def latent_state_value(self, state: str | int, variable: str, path=None):
        """Return a DomiKnowS predicate asserting a token has a latent state."""
        state_concept = getattr(self.latent_state, str(self.state_index(state)))
        if path is None:
            return state_concept(variable)
        return state_concept(variable, path=path)

    def forward_state_value(self, state: str | int, variable: str, path=None):
        """Return a predicate asserting a token has a scaled-alpha state value."""
        if self.forward_state is None:
            raise ValueError("forward_state is only available with include_dp_factors=True")
        state_concept = getattr(self.forward_state, str(self.state_index(state)))
        if path is None:
            return state_concept(variable)
        return state_concept(variable, path=path)

    def backward_state_value(self, state: str | int, variable: str, path=None):
        """Return a predicate asserting a token has a normalized beta state value."""
        if self.backward_state is None:
            raise ValueError("backward_state is only available with include_dp_factors=True")
        state_concept = getattr(self.backward_state, str(self.state_index(state)))
        if path is None:
            return state_concept(variable)
        return state_concept(variable, path=path)

    def transition_pair_value(self, from_state: str | int, to_state: str | int, variable: str, path=None):
        """Return a predicate asserting an adjacent relation has an HMM state pair."""
        if self.transition_pair is None:
            raise ValueError("transition_pair is only available with include_dp_factors=True")
        pair_concept = getattr(self.transition_pair, str(self.transition_pair_index(from_state, to_state)))
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
                raise KeyError(f"unknown latent state {state!r}") from exc
        if index < 0 or index >= len(self.state_names):
            raise IndexError(f"latent state index {index} is out of range")
        return index

    def transition_pair_index(self, from_state: str | int, to_state: str | int) -> int:
        """Resolve a ``from -> to`` state pair to the flattened transition enum index."""
        return self.state_index(from_state) * len(self.state_names) + self.state_index(to_state)


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
        include_dp_factors: bool = False,
    ):
        self.vocabulary = TokenVocabulary(vocab, eos_token=eos_token, tokenizer=tokenizer)
        self.state_names = _resolve_state_names(state_names, state_count)
        self.graph_name = graph_name
        self.clear_graph = clear_graph
        self.include_dp_factors = bool(include_dp_factors)

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
            forward_state = None
            backward_state = None
            transition_pair = None
            transition_pair_names: tuple[str, ...] = ()
            if self.include_dp_factors:
                state_values = [str(i) for i in range(len(self.state_names))]
                forward_state = token(name="forward_state", ConceptClass=EnumConcept, values=state_values)
                backward_state = token(name="backward_state", ConceptClass=EnumConcept, values=state_values)
                transition_pair_names = tuple(
                    f"{from_state}->{to_state}"
                    for from_state in self.state_names
                    for to_state in self.state_names
                )
                transition_pair = is_next_rel(
                    name="transition_pair",
                    ConceptClass=EnumConcept,
                    values=[str(i) for i in range(len(transition_pair_names))],
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
                forward_state=forward_state,
                backward_state=backward_state,
                transition_pair=transition_pair,
                transition_pair_names=transition_pair_names,
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
            forward_state=forward_state,
            backward_state=backward_state,
            transition_pair=transition_pair,
            transition_pair_names=transition_pair_names,
            include_dp_factors=self.include_dp_factors,
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

    def forward_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for scaled alpha factors."""
        return _HMMFactorForwardProjection(self)

    def backward_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for normalized beta factors."""
        return _HMMFactorBackwardProjection(self)

    def transition_pair_module(self) -> torch.nn.Module:
        """Return a ModuleLearner-compatible projection for adjacent xi factors."""
        return _HMMFactorTransitionPairProjection(self)

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
        return self.forward_backward_factors(target_labels)["gamma"]

    def forward_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return log normalized scaled-alpha factors shaped ``[T, state_count]``."""
        alpha = self.forward_backward_factors(target_labels)["alpha"]
        return torch.log(alpha.clamp_min(torch.finfo(alpha.dtype).eps))

    def backward_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return log normalized beta factors shaped ``[T, state_count]``."""
        beta = self.forward_backward_factors(target_labels)["beta"]
        return torch.log(beta.clamp_min(torch.finfo(beta.dtype).eps))

    def transition_pair_log_probs(self, target_labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return log xi transition-pair factors shaped ``[T - 1, state_count ** 2]``."""
        xi = self.forward_backward_factors(target_labels)["xi"].reshape(-1, self.state_count * self.state_count)
        return torch.log(xi.clamp_min(torch.finfo(xi.dtype).eps))

    def production_hmm(self) -> DiscreteHMM:
        """Return a Torch-backed HMM view of the current factor-head parameters."""
        return DiscreteHMM(
            self.transition_probs,
            self.emission_probs,
            self.initial_probs,
            tuple(range(self.label_count)),
            state_names=self.state_names,
            normalize=False,
        )

    def forward_backward_factors(self, target_labels: torch.Tensor | Sequence[int]) -> dict[str, torch.Tensor]:
        """Return scaled alpha, normalized beta, gamma, xi, and forward scales."""
        labels = self._target_labels(target_labels)
        factors = self.production_hmm().forward_backward(labels.unsqueeze(0), torch.tensor([labels.numel()], device=labels.device))
        alpha = factors.alpha[0]
        raw_beta = factors.beta[0]
        eps = torch.finfo(alpha.dtype).eps
        gamma = factors.gamma[0]
        beta = raw_beta / raw_beta.sum(dim=-1, keepdim=True).clamp_min(eps)
        xi = factors.xi[0]
        scales = factors.scales[0]
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "xi": xi, "scales": scales}

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

    def _transition_pair_marginals(
        self,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        transition = self.transition_probs
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        if labels.numel() < 2:
            return torch.empty(
                (0, self.state_count, self.state_count),
                dtype=emission.dtype,
                device=emission.device,
            )
        xis = []
        for t in range(labels.numel() - 1):
            next_emission = emission[:, int(labels[t + 1].item())]
            pair = alpha[t].unsqueeze(1) * transition * (next_emission * beta[t + 1]).unsqueeze(0)
            pair = pair / pair.sum().clamp_min(eps)
            xis.append(pair)
        return torch.stack(xis, dim=0)

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


class _HMMFactorForwardProjection(torch.nn.Module):
    def __init__(self, head: HMMFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.forward_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return self.head.pad_size


class _HMMFactorBackwardProjection(torch.nn.Module):
    def __init__(self, head: HMMFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.backward_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return self.head.pad_size


class _HMMFactorTransitionPairProjection(torch.nn.Module):
    def __init__(self, head: HMMFactorGraphHead):
        super().__init__()
        self.head = head

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.head.transition_pair_log_probs(target_labels)

    @property
    def pad_size(self) -> int:
        return max(0, self.head.pad_size - 1)


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


def hmm_forward_backward_factors(
    head: HMMFactorGraphHead,
    target_labels: torch.Tensor | Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return differentiable HMM DP factors exposed by ``HMMFactorGraphHead``."""
    if not isinstance(head, HMMFactorGraphHead):
        raise TypeError("hmm_forward_backward_factors expects an HMMFactorGraphHead")
    return head.forward_backward_factors(target_labels)


def hmm_dp_factor_consistency_loss(
    head: HMMFactorGraphHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return a differentiable diagnostic tying graph-visible DP factors together.

    The exposed alpha, beta, gamma, and xi factors are all derived from the same
    numeric forward/backward recurrence. This loss therefore acts as a small
    inspection/auxiliary term: it checks that normalized ``alpha * beta`` matches
    ``gamma`` and that xi row/column marginals agree with adjacent gammas.
    """
    factors = hmm_forward_backward_factors(head, target_labels)
    gamma = factors["gamma"]
    xi = factors["xi"]
    losses = [gamma.new_zeros((1,))]
    if xi.numel():
        losses.append((xi.sum(dim=2) - gamma[:-1]).pow(2).mean(dim=-1))
        losses.append((xi.sum(dim=1) - gamma[1:]).pow(2).mean(dim=-1))
    flat = torch.cat([loss.reshape(-1) for loss in losses], dim=0)
    if reduction == "none":
        return flat
    if reduction == "sum":
        return flat.sum()
    if reduction == "mean":
        return flat.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def apply_hmm_dp_consistency_constraints(bundle: HMMFactorGraphBundle) -> None:
    """Add weak PMD-visible logical consistency rules for graph-exposed DP factors."""
    if not bundle.include_dp_factors:
        raise ValueError("HMM DP consistency constraints require include_dp_factors=True")
    from domiknows.graph.logicalConstrain import andL, ifL

    ctx = bundle.context
    for state in bundle.state_names:
        ifL(
            andL(ctx.forward_state_value(state, "x"), ctx.backward_state_value(state, "x")),
            ctx.latent_state_value(state, "x"),
        )

    for from_state in bundle.state_names:
        for to_state in bundle.state_names:
            ifL(
                ctx.is_next_rel("next"),
                ifL(
                    ctx.transition_pair_value(from_state, to_state, "next"),
                    andL(
                        ctx.latent_state_value(from_state, "x", path=("next", ctx.current_token)),
                        ctx.latent_state_value(to_state, "y", path=("next", ctx.next_token)),
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


def _coerce_label_to_token_id(
    label_to_token_id: Sequence[int | None] | None,
    label_count: int,
) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)
