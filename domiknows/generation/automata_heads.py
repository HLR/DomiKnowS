"""Torch generation heads backed by HMMs and weighted finite automata.

These modules expose the small interface used by DomiKnowS ``ModuleLearner``
in generation graphs: ``forward(_contains, instruction_tokens, target_labels)``
returns log-probabilities shaped ``[seq_len, label_count]`` for single-example
``ModuleLearner`` use, while production helpers return batched tensors shaped
``[batch, seq_len, label_count]``. They also expose ``next_label_logits`` so
the same trained head can be decoded with the label-level DFA decoder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .automata import DFA, DiscreteHMM, ProbabilisticAutomaton, WeightedFiniteAutomaton
from .latent_potentials import (
    LatentTransitionPotential,
    apply_hmm_transition_potential,
    apply_wfa_transition_potential,
)

TransitionPotentialInput = LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None


class PromptEmbeddingEncoder(torch.nn.Module):
    """Small trainable prompt encoder for offline prompt-conditioned heads."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            _positive_int(vocab_size, "vocab_size"),
            _positive_int(hidden_size, "hidden_size"),
        )

    @property
    def output_size(self) -> int:
        return int(self.embedding.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.embedding(input_ids.long()).mean(dim=1)


class FrozenBackbonePromptEncoder(torch.nn.Module):
    """Frozen-backbone prompt encoder mirroring the HF compact-label learner."""

    def __init__(self, backbone: torch.nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self._output_size = int(hidden_size) if hidden_size is not None else _infer_backbone_hidden_size(backbone)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            try:
                output = self.backbone(input_ids.long(), output_hidden_states=True)
            except TypeError:
                output = self.backbone(input_ids.long())
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state
            elif hasattr(output, "hidden_states") and output.hidden_states:
                features = output.hidden_states[-1]
            elif hasattr(output, "logits"):
                features = output.logits
            else:
                raise ValueError("backbone output must expose tensor features, hidden states, or logits")
        if features.shape[-1] != self._output_size:
            raise ValueError(
                "backbone feature size does not match prompt encoder output_size; "
                "pass a backbone that exposes hidden states or set backbone_hidden_size"
            )
        if features.dim() == 2:
            return features.detach()
        return features[:, -1, :].detach()


class HMMGenerationHead(torch.nn.Module):
    """Compact-label generation head backed by a discrete HMM.

    The HMM is parameterized with trainable logits when ``trainable=True`` and
    frozen tensors otherwise.  Emissions are labels in the compact
    ``GenerationEncoder`` vocabulary, so the output can be attached directly to
    ``token[generated_token]`` via ``ModuleLearner``.
    """

    def __init__(
        self,
        model: DiscreteHMM | ProbabilisticAutomaton | None = None,
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
            initial = torch.as_tensor(model.initial, dtype=torch.float32)
            transition = torch.as_tensor(model.transition, dtype=torch.float32)
            emission = torch.as_tensor(model.emission, dtype=torch.float32)
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

    def transition_probs_with_potential(self, transition_potential: TransitionPotentialInput = None) -> torch.Tensor:
        """Return HMM transition probabilities after optional latent-potential reweighting."""
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        state = self.initial_probs
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps

        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)

        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return next-step logits over compact generation labels."""
        return self._next_logits_from_prefix_labels(
            self._labels_from_input_ids(input_ids),
            transition_potential=transition_potential,
        )

    def production_hmm(self, transition_potential: TransitionPotentialInput = None) -> DiscreteHMM:
        """Return a Torch-backed HMM view of the current head parameters."""
        return DiscreteHMM(
            self.transition_probs_with_potential(transition_potential),
            self.emission_probs,
            self.initial_probs,
            tuple(range(self.label_count)),
            normalize=False,
        )

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return teacher-forced log-probs shaped ``[batch, seq, label_count]``."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.initial_logits.device,
            lengths=lengths,
        )
        batch, seq_len = labels.shape
        state = self.initial_probs.expand(batch, -1)
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        outputs = []
        for step in range(seq_len):
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            posterior = state * emission[:, labels[:, step]].transpose(0, 1)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_state = torch.matmul(posterior, transition)
            active = (step < lengths_t).unsqueeze(-1)
            state = torch.where(active, next_state, state)
        result = torch.log_softmax(torch.stack(outputs, dim=1), dim=-1)
        return result[0] if squeeze else result

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
    ):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        return self.sequence_log_probs(target_labels, transition_potential=transition_potential)

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

    def transitions_with_potential(
        self,
        transition_potential: TransitionPotentialInput = None,
        *,
        mode: str = "multiply",
    ) -> torch.Tensor:
        """Return signed WFA transitions after optional latent-potential reweighting."""
        return apply_wfa_transition_potential(self.transitions, transition_potential, mode=mode)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _prefix_state(
        self,
        prefix_labels: Sequence[int],
        *,
        transitions: torch.Tensor | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        state = self.initial
        transitions = (
            self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
            if transitions is None
            else transitions
        )
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            state = torch.matmul(state, transitions[label])
        return state

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        transitions = self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
        state = self._prefix_state(prefix_labels, transitions=transitions)
        next_states = torch.einsum("s,lsd->ld", state, transitions)
        return torch.matmul(next_states, self.final)

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return next-step signed WFA scores over compact labels."""
        return self._next_logits_from_prefix_labels(
            self._labels_from_input_ids(input_ids),
            transition_potential=transition_potential,
            transition_potential_mode=transition_potential_mode,
        )

    def production_wfa(
        self,
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> WeightedFiniteAutomaton:
        """Return a Torch-backed WFA view of the current head parameters."""
        return WeightedFiniteAutomaton(
            self.initial,
            self.transitions_with_potential(transition_potential, mode=transition_potential_mode),
            self.final,
            tuple(range(self.label_count)),
        )

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return teacher-forced log-probs shaped ``[batch, seq, label_count]``."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.initial.device,
            lengths=lengths,
        )
        batch, seq_len = labels.shape
        state = self.initial.expand(batch, -1)
        transitions = self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
        outputs = []
        for step in range(seq_len):
            next_states = torch.einsum("bs,lsd->bld", state, transitions)
            outputs.append(torch.matmul(next_states, self.final))
            chosen = transitions.index_select(0, labels[:, step])
            next_state = torch.bmm(state.unsqueeze(1), chosen).squeeze(1)
            active = (step < lengths_t).unsqueeze(-1)
            state = torch.where(active, next_state, state)
        result = torch.log_softmax(torch.stack(outputs, dim=1), dim=-1)
        return result[0] if squeeze else result

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
    ):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        return self.sequence_log_probs(target_labels, transition_potential=transition_potential)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


class PromptConditionedHMMGenerationHead(torch.nn.Module):
    """HMM generation head whose initial state is conditioned on the prompt."""

    def __init__(
        self,
        *,
        label_count: int,
        state_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        prompt_encoder: torch.nn.Module | None = None,
        prompt_encoder_type: str = "embedding",
        prompt_vocab_size: int = 1024,
        prompt_hidden_size: int = 16,
        backbone: torch.nn.Module | None = None,
        backbone_hidden_size: int | None = None,
        dynamics_conditioning: str = "none",
        dynamics_expert_count: int = 2,
        step_dynamics_conditioning: str = "none",
        trainable: bool = True,
        random_seed: int = 0,
    ):
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _positive_int(label_count, "label_count")
        self.state_count = _positive_int(state_count, "state_count")
        self.dynamics_conditioning = _normalise_dynamics_conditioning(dynamics_conditioning)
        self.step_dynamics_conditioning = _normalise_step_dynamics_conditioning(step_dynamics_conditioning)
        if self.step_dynamics_conditioning != "none" and self.dynamics_conditioning != "gated":
            raise ValueError("step_dynamics_conditioning='prefix_gated' requires dynamics_conditioning='gated'")
        self.dynamics_expert_count = (
            _positive_int(dynamics_expert_count, "dynamics_expert_count")
            if self.dynamics_conditioning == "gated"
            else 1
        )
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)
        self.prompt_encoder = _build_prompt_encoder(
            prompt_encoder=prompt_encoder,
            prompt_encoder_type=prompt_encoder_type,
            prompt_vocab_size=prompt_vocab_size,
            prompt_hidden_size=prompt_hidden_size,
            backbone=backbone,
            backbone_hidden_size=backbone_hidden_size,
        )
        _configure_prompt_encoder_trainability(self.prompt_encoder, trainable)
        self.initial_projector = torch.nn.Linear(self.prompt_encoder.output_size, self.state_count)

        _initial, transition, emission = _random_hmm_parameters(self.state_count, self.label_count, random_seed)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission), requires_grad=trainable)
        if self.dynamics_conditioning == "gated":
            transition_experts, emission_experts = _random_hmm_dynamics_experts(
                self.dynamics_expert_count - 1,
                self.state_count,
                self.label_count,
                random_seed + 101,
            )
            self.transition_expert_logits = torch.nn.Parameter(transition_experts, requires_grad=trainable)
            self.emission_expert_logits = torch.nn.Parameter(emission_experts, requires_grad=trainable)
            self.dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size, self.dynamics_expert_count)
            for parameter in self.dynamics_gate.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.register_parameter("transition_expert_logits", None)
            self.register_parameter("emission_expert_logits", None)
            self.dynamics_gate = None
        if self.step_dynamics_conditioning == "prefix_gated":
            self.prefix_embedding = torch.nn.Embedding(self.label_count, self.prompt_encoder.output_size)
            self.step_dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size * 2, self.dynamics_expert_count)
            for parameter in self.prefix_embedding.parameters():
                parameter.requires_grad_(trainable)
            for parameter in self.step_dynamics_gate.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.prefix_embedding = None
            self.step_dynamics_gate = None
        for parameter in self.initial_projector.parameters():
            parameter.requires_grad_(trainable)

    @property
    def transition_probs(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        return torch.softmax(self.emission_logits, dim=-1)

    def transition_probs_with_potential(self, transition_potential: TransitionPotentialInput = None) -> torch.Tensor:
        """Return base HMM transitions after optional latent-potential reweighting."""
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def prompt_initial_probs(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return prompt-conditioned initial HMM state probabilities."""
        features = self._prompt_features(instruction_tokens)
        logits = self.initial_projector(features)[0]
        return torch.softmax(logits, dim=-1)

    def prompt_dynamics_weights(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return prompt-selected HMM dynamics expert weights."""
        if self.dynamics_conditioning != "gated":
            return torch.ones(1, dtype=self.transition_logits.dtype, device=self._parameter_device())
        features = self._prompt_features(instruction_tokens)
        return torch.softmax(self.dynamics_gate(features)[0], dim=-1)

    def prompt_transition_probs(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return the prompt-conditioned HMM transition matrix."""
        if self.dynamics_conditioning != "gated":
            return self.transition_probs_with_potential(transition_potential)
        weights = self.prompt_dynamics_weights(instruction_tokens)
        experts = _stack_base_and_optional_experts(self.transition_logits, self.transition_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return apply_hmm_transition_potential(torch.softmax(logits, dim=-1), transition_potential)

    def prompt_emission_probs(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return the prompt-conditioned HMM emission matrix."""
        if self.dynamics_conditioning != "gated":
            return self.emission_probs
        weights = self.prompt_dynamics_weights(instruction_tokens)
        experts = _stack_base_and_optional_experts(self.emission_logits, self.emission_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return torch.softmax(logits, dim=-1)

    def step_dynamics_weights(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
    ) -> torch.Tensor:
        """Return HMM dynamics expert weights for one generated-prefix step."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_dynamics_weights(instruction_tokens)
        if self.dynamics_expert_count == 1:
            return torch.ones(1, dtype=self.transition_logits.dtype, device=self._parameter_device())
        features = self._prompt_features(instruction_tokens)[0]
        summary = self._prefix_summary(prefix_labels)
        return torch.softmax(self.step_dynamics_gate(torch.cat([features, summary], dim=-1)), dim=-1)

    def step_transition_probs(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return HMM transitions gated by prompt and generated prefix."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_transition_probs(instruction_tokens, transition_potential=transition_potential)
        weights = self.step_dynamics_weights(instruction_tokens, prefix_labels)
        experts = _stack_base_and_optional_experts(self.transition_logits, self.transition_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return apply_hmm_transition_potential(torch.softmax(logits, dim=-1), transition_potential)

    def step_emission_probs(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
    ) -> torch.Tensor:
        """Return HMM emissions gated by prompt and generated prefix."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_emission_probs(instruction_tokens)
        weights = self.step_dynamics_weights(instruction_tokens, prefix_labels)
        experts = _stack_base_and_optional_experts(self.emission_logits, self.emission_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return torch.softmax(logits, dim=-1)

    def _prompt_features(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        prompt = _normalise_prompt_ids(instruction_tokens, device=self._parameter_device())
        return self.prompt_encoder(prompt)

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device

    def _prefix_summary(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        if self.prefix_embedding is None or not prefix_labels:
            return torch.zeros(self.prompt_encoder.output_size, dtype=self.transition_logits.dtype, device=self._parameter_device())
        labels = torch.tensor(
            [_validate_label(label, self.label_count) for label in prefix_labels],
            dtype=torch.long,
            device=self._parameter_device(),
        )
        return self.prefix_embedding(labels).mean(dim=0)

    def token_id_for_label(self, label: int) -> int:
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or ids[:1]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        return torch.tensor([prompt_ids], dtype=torch.long, device=device), labels

    def _next_logits_from_prompt_and_prefix(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        state = self.prompt_initial_probs(instruction_tokens)
        eps = torch.finfo(self.emission_logits.dtype).eps
        consumed: list[int] = []
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            emission = self.step_emission_probs(instruction_tokens, consumed)
            transition = self.step_transition_probs(
                instruction_tokens,
                consumed,
                transition_potential=transition_potential,
            )
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)
            consumed.append(label)
        emission = self.step_emission_probs(instruction_tokens, consumed)
        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        return self._next_logits_from_prompt_and_prefix(prompt_ids, prefix_labels, transition_potential=transition_potential)

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
    ):
        labels = _target_labels(target_labels, self.pad_size, device=self.transition_logits.device)
        generated = []
        prefix: list[int] = []
        for step in range(self.pad_size):
            generated.append(
                self._next_logits_from_prompt_and_prefix(
                    instruction_tokens,
                    prefix,
                    transition_potential=transition_potential,
                )
            )
            prefix.append(int(labels[step].item()))
        return torch.log_softmax(torch.stack(generated, dim=0), dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


class PromptConditionedSpectralWFAGenerationHead(torch.nn.Module):
    """Signed WFA generation head whose initial vector is prompt-conditioned."""

    def __init__(
        self,
        *,
        label_count: int,
        state_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        prompt_encoder: torch.nn.Module | None = None,
        prompt_encoder_type: str = "embedding",
        prompt_vocab_size: int = 1024,
        prompt_hidden_size: int = 16,
        backbone: torch.nn.Module | None = None,
        backbone_hidden_size: int | None = None,
        dynamics_conditioning: str = "none",
        dynamics_expert_count: int = 2,
        step_dynamics_conditioning: str = "none",
        trainable: bool = True,
        random_seed: int = 0,
    ):
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _positive_int(label_count, "label_count")
        self.state_count = _positive_int(state_count, "state_count")
        self.dynamics_conditioning = _normalise_dynamics_conditioning(dynamics_conditioning)
        self.step_dynamics_conditioning = _normalise_step_dynamics_conditioning(step_dynamics_conditioning)
        if self.step_dynamics_conditioning != "none" and self.dynamics_conditioning != "gated":
            raise ValueError("step_dynamics_conditioning='prefix_gated' requires dynamics_conditioning='gated'")
        self.dynamics_expert_count = (
            _positive_int(dynamics_expert_count, "dynamics_expert_count")
            if self.dynamics_conditioning == "gated"
            else 1
        )
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)
        self.prompt_encoder = _build_prompt_encoder(
            prompt_encoder=prompt_encoder,
            prompt_encoder_type=prompt_encoder_type,
            prompt_vocab_size=prompt_vocab_size,
            prompt_hidden_size=prompt_hidden_size,
            backbone=backbone,
            backbone_hidden_size=backbone_hidden_size,
        )
        _configure_prompt_encoder_trainability(self.prompt_encoder, trainable)
        self.initial_projector = torch.nn.Linear(self.prompt_encoder.output_size, self.state_count)

        _initial, transitions, final = _random_wfa_parameters(self.state_count, self.label_count, random_seed)
        self.transitions = torch.nn.Parameter(transitions, requires_grad=trainable)
        self.final = torch.nn.Parameter(final, requires_grad=trainable)
        if self.dynamics_conditioning == "gated":
            transition_experts, final_experts = _random_wfa_dynamics_experts(
                self.dynamics_expert_count - 1,
                self.state_count,
                self.label_count,
                random_seed + 211,
            )
            self.transition_experts = torch.nn.Parameter(transition_experts, requires_grad=trainable)
            self.final_experts = torch.nn.Parameter(final_experts, requires_grad=trainable)
            self.dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size, self.dynamics_expert_count)
            for parameter in self.dynamics_gate.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.register_parameter("transition_experts", None)
            self.register_parameter("final_experts", None)
            self.dynamics_gate = None
        if self.step_dynamics_conditioning == "prefix_gated":
            self.prefix_embedding = torch.nn.Embedding(self.label_count, self.prompt_encoder.output_size)
            self.step_dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size * 2, self.dynamics_expert_count)
            for parameter in self.prefix_embedding.parameters():
                parameter.requires_grad_(trainable)
            for parameter in self.step_dynamics_gate.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.prefix_embedding = None
            self.step_dynamics_gate = None
        for parameter in self.initial_projector.parameters():
            parameter.requires_grad_(trainable)

    def prompt_initial_state(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return the prompt-conditioned signed WFA initial vector."""
        features = self._prompt_features(instruction_tokens)
        return self.initial_projector(features)[0]

    def prompt_dynamics_weights(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return prompt-selected WFA dynamics expert weights."""
        if self.dynamics_conditioning != "gated":
            return torch.ones(1, dtype=self.transitions.dtype, device=self._parameter_device())
        features = self._prompt_features(instruction_tokens)
        return torch.softmax(self.dynamics_gate(features)[0], dim=-1)

    def prompt_transitions(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return the prompt-conditioned signed WFA transition tensor."""
        if self.dynamics_conditioning != "gated":
            return apply_wfa_transition_potential(
                self.transitions,
                transition_potential,
                mode=transition_potential_mode,
            )
        weights = self.prompt_dynamics_weights(instruction_tokens)
        experts = _stack_base_and_optional_experts(self.transitions, self.transition_experts)
        transitions = torch.einsum("e,elsd->lsd", weights, experts)
        return apply_wfa_transition_potential(transitions, transition_potential, mode=transition_potential_mode)

    def prompt_final(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return the prompt-conditioned signed WFA final/scoring vector."""
        if self.dynamics_conditioning != "gated":
            return self.final
        weights = self.prompt_dynamics_weights(instruction_tokens)
        experts = _stack_base_and_optional_experts(self.final, self.final_experts)
        return torch.einsum("e,es->s", weights, experts)

    def step_dynamics_weights(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
    ) -> torch.Tensor:
        """Return WFA dynamics expert weights for one generated-prefix step."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_dynamics_weights(instruction_tokens)
        if self.dynamics_expert_count == 1:
            return torch.ones(1, dtype=self.transitions.dtype, device=self._parameter_device())
        features = self._prompt_features(instruction_tokens)[0]
        summary = self._prefix_summary(prefix_labels)
        return torch.softmax(self.step_dynamics_gate(torch.cat([features, summary], dim=-1)), dim=-1)

    def step_transitions(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return WFA transitions gated by prompt and generated prefix."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_transitions(
                instruction_tokens,
                transition_potential=transition_potential,
                transition_potential_mode=transition_potential_mode,
            )
        weights = self.step_dynamics_weights(instruction_tokens, prefix_labels)
        experts = _stack_base_and_optional_experts(self.transitions, self.transition_experts)
        transitions = torch.einsum("e,elsd->lsd", weights, experts)
        return apply_wfa_transition_potential(transitions, transition_potential, mode=transition_potential_mode)

    def step_final(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int] = (),
    ) -> torch.Tensor:
        """Return WFA final/scoring vector gated by prompt and generated prefix."""
        if self.step_dynamics_conditioning == "none":
            return self.prompt_final(instruction_tokens)
        weights = self.step_dynamics_weights(instruction_tokens, prefix_labels)
        experts = _stack_base_and_optional_experts(self.final, self.final_experts)
        return torch.einsum("e,es->s", weights, experts)

    def _prompt_features(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        prompt = _normalise_prompt_ids(instruction_tokens, device=self._parameter_device())
        return self.prompt_encoder(prompt)

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device

    def _prefix_summary(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        if self.prefix_embedding is None or not prefix_labels:
            return torch.zeros(self.prompt_encoder.output_size, dtype=self.transitions.dtype, device=self._parameter_device())
        labels = torch.tensor(
            [_validate_label(label, self.label_count) for label in prefix_labels],
            dtype=torch.long,
            device=self._parameter_device(),
        )
        return self.prefix_embedding(labels).mean(dim=0)

    def token_id_for_label(self, label: int) -> int:
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or ids[:1]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        return torch.tensor([prompt_ids], dtype=torch.long, device=device), labels

    def _prefix_state(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int],
        transitions: torch.Tensor | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        state = self.prompt_initial_state(instruction_tokens)
        consumed: list[int] = []
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            active_transitions = (
                self.step_transitions(
                    instruction_tokens,
                    consumed,
                    transition_potential=transition_potential,
                    transition_potential_mode=transition_potential_mode,
                )
                if transitions is None
                else transitions
            )
            state = torch.matmul(state, active_transitions[label])
            consumed.append(label)
        return state

    def _next_logits_from_prompt_and_prefix(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        transitions = self.step_transitions(
            instruction_tokens,
            prefix_labels,
            transition_potential=transition_potential,
            transition_potential_mode=transition_potential_mode,
        )
        final = self.step_final(instruction_tokens, prefix_labels)
        state = self._prefix_state(
            instruction_tokens,
            prefix_labels,
            transition_potential=transition_potential,
            transition_potential_mode=transition_potential_mode,
        )
        next_states = torch.einsum("s,lsd->ld", state, transitions)
        return torch.matmul(next_states, final)

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        return self._next_logits_from_prompt_and_prefix(
            prompt_ids,
            prefix_labels,
            transition_potential=transition_potential,
            transition_potential_mode=transition_potential_mode,
        )

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
    ):
        labels = _target_labels(target_labels, self.pad_size, device=self.transitions.device)
        generated = []
        prefix: list[int] = []
        for step in range(self.pad_size):
            generated.append(
                self._next_logits_from_prompt_and_prefix(
                    instruction_tokens,
                    prefix,
                    transition_potential=transition_potential,
                )
            )
            prefix.append(int(labels[step].item()))
        return torch.log_softmax(torch.stack(generated, dim=0), dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


def hmm_sequence_nll(
    head: HMMGenerationHead | PromptConditionedHMMGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    transition_potential: TransitionPotentialInput = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of a target label sequence under an HMM head."""
    if not isinstance(head, (HMMGenerationHead, PromptConditionedHMMGenerationHead)):
        raise TypeError("hmm_sequence_nll expects an HMM generation head")
    device = head.transition_logits.device
    labels = _target_labels(target_labels, head.pad_size, device=device)
    prompt = _empty_or_prompt(instruction_tokens, device)
    log_probs = head(None, prompt, labels, transition_potential=transition_potential)
    return F.nll_loss(log_probs, labels, reduction=reduction)


def wfa_sequence_energy_loss(
    head: SpectralWFAGenerationHead | PromptConditionedSpectralWFAGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    transition_potential: TransitionPotentialInput = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Energy-style supervised loss for a WFA head.

    Signed WFA next-symbol scores are interpreted as logits and optimized with
    cross-entropy against the target compact labels.
    """
    if not isinstance(head, (SpectralWFAGenerationHead, PromptConditionedSpectralWFAGenerationHead)):
        raise TypeError("wfa_sequence_energy_loss expects a spectral WFA generation head")
    device = head.transitions.device if hasattr(head, "transitions") else head.initial.device
    labels = _target_labels(target_labels, head.pad_size, device=device)
    prompt = _empty_or_prompt(instruction_tokens, device)
    log_probs = head(None, prompt, labels, transition_potential=transition_potential)
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


def _normalise_dynamics_conditioning(value: str) -> str:
    value = str(value).lower().replace("-", "_")
    if value not in {"none", "gated"}:
        raise ValueError("dynamics_conditioning must be 'none' or 'gated'")
    return value


def _normalise_step_dynamics_conditioning(value: str) -> str:
    value = str(value).lower().replace("-", "_")
    if value not in {"none", "prefix_gated"}:
        raise ValueError("step_dynamics_conditioning must be 'none' or 'prefix_gated'")
    return value


def _stack_base_and_optional_experts(base: torch.Tensor, experts: torch.Tensor | None) -> torch.Tensor:
    if experts is None or experts.numel() == 0:
        return base.unsqueeze(0)
    return torch.cat([base.unsqueeze(0), experts], dim=0)


def _infer_backbone_hidden_size(backbone: torch.nn.Module) -> int:
    """Infer the final feature dimension exposed by a frozen prompt backbone."""
    embedding = getattr(backbone, "embedding", None)
    if embedding is not None and hasattr(embedding, "embedding_dim"):
        return int(embedding.embedding_dim)

    config = getattr(backbone, "config", None)
    if config is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

    for attr in ("hidden_size", "n_embd", "output_size"):
        value = getattr(backbone, attr, None)
        if value is not None:
            return int(value)

    raise ValueError("backbone hidden size could not be inferred; pass backbone_hidden_size")


def _build_prompt_encoder(
    *,
    prompt_encoder: torch.nn.Module | None,
    prompt_encoder_type: str,
    prompt_vocab_size: int,
    prompt_hidden_size: int,
    backbone: torch.nn.Module | None,
    backbone_hidden_size: int | None,
) -> torch.nn.Module:
    if prompt_encoder is not None:
        if not hasattr(prompt_encoder, "output_size"):
            raise ValueError("prompt_encoder must expose an output_size property")
        return prompt_encoder

    encoder_type = prompt_encoder_type.lower().replace("-", "_")
    if encoder_type == "embedding":
        return PromptEmbeddingEncoder(prompt_vocab_size, prompt_hidden_size)
    if encoder_type in {"frozen_backbone", "backbone"}:
        if backbone is None:
            raise ValueError("backbone is required when prompt_encoder_type='frozen_backbone'")
        return FrozenBackbonePromptEncoder(backbone, hidden_size=backbone_hidden_size)
    raise ValueError("prompt_encoder_type must be 'embedding' or 'frozen_backbone'")


def _configure_prompt_encoder_trainability(prompt_encoder: torch.nn.Module, trainable: bool) -> None:
    if isinstance(prompt_encoder, FrozenBackbonePromptEncoder):
        for parameter in prompt_encoder.backbone.parameters():
            parameter.requires_grad_(False)
        return
    for parameter in prompt_encoder.parameters():
        parameter.requires_grad_(trainable)


def _normalise_prompt_ids(
    input_ids: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        prompt = input_ids.detach().long()
        if prompt.dim() == 0:
            prompt = prompt.reshape(1, 1)
        elif prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        elif prompt.dim() != 2:
            raise ValueError("instruction_tokens must have shape [seq] or [batch, seq]")
        if prompt.numel() == 0:
            prompt = torch.zeros((1, 1), dtype=torch.long, device=prompt.device)
        return prompt.to(device)

    ids = [int(token_id) for token_id in input_ids]
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _normalise_flat_ids(input_ids: torch.Tensor | Sequence[int]) -> tuple[list[int], torch.device]:
    if isinstance(input_ids, torch.Tensor):
        device = input_ids.device
        if input_ids.dim() == 0:
            flat = [int(input_ids.item())]
        elif input_ids.dim() == 1:
            flat = [int(value) for value in input_ids.detach().long().tolist()]
        elif input_ids.dim() == 2 and input_ids.shape[0] == 1:
            flat = [int(value) for value in input_ids[0].detach().long().tolist()]
        else:
            raise ValueError("input_ids must describe a single sequence")
        return flat, device
    return [int(value) for value in input_ids], torch.device("cpu")


def _first_generated_index(ids: Sequence[int], token_id_to_label: Mapping[int, int]) -> int:
    for index, token_id in enumerate(ids):
        if int(token_id) in token_id_to_label:
            return index
    return len(ids)


def _empty_or_prompt(
    instruction_tokens: torch.Tensor | Sequence[int] | None,
    device: torch.device,
) -> torch.Tensor:
    if instruction_tokens is None:
        return torch.zeros((1, 1), dtype=torch.long, device=device)
    return _normalise_prompt_ids(instruction_tokens, device=device)


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


def _target_label_batch(
    target_labels: torch.Tensor | Sequence[int],
    pad_size: int,
    *,
    device: torch.device,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if isinstance(target_labels, torch.Tensor):
        labels = target_labels.detach().long().to(device)
    else:
        labels = torch.tensor(target_labels, dtype=torch.long, device=device)
    squeeze = labels.dim() == 1
    if squeeze:
        labels = labels.unsqueeze(0)
    if labels.dim() != 2:
        raise ValueError("target_labels must have shape [seq] or [batch, seq]")
    if labels.shape[1] >= pad_size:
        labels = labels[:, :pad_size]
    else:
        padding = torch.full(
            (labels.shape[0], pad_size - labels.shape[1]),
            int(eos_label),
            dtype=torch.long,
            device=device,
        )
        labels = torch.cat([labels, padding], dim=1)
    if torch.any(labels < 0):
        raise ValueError("target_labels must be non-negative")
    if lengths is None:
        lengths_t = torch.full((labels.shape[0],), labels.shape[1], dtype=torch.long, device=device)
    else:
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
    if lengths_t.numel() != labels.shape[0]:
        raise ValueError("lengths must contain one value per batch item")
    lengths_t = torch.clamp(lengths_t, min=1, max=labels.shape[1])
    return labels, lengths_t, squeeze


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


def _random_hmm_dynamics_experts(
    expert_count: int,
    state_count: int,
    label_count: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_count < 1:
        return (
            torch.empty((0, state_count, state_count), dtype=torch.float32),
            torch.empty((0, state_count, label_count), dtype=torch.float32),
        )
    transition_experts = []
    emission_experts = []
    for offset in range(expert_count):
        _initial, transition, emission = _random_hmm_parameters(
            state_count,
            label_count,
            random_seed + offset,
        )
        transition_experts.append(_safe_log(transition))
        emission_experts.append(_safe_log(emission))
    return torch.stack(transition_experts, dim=0), torch.stack(emission_experts, dim=0)


def _random_wfa_dynamics_experts(
    expert_count: int,
    state_count: int,
    label_count: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_count < 1:
        return (
            torch.empty((0, label_count, state_count, state_count), dtype=torch.float32),
            torch.empty((0, state_count), dtype=torch.float32),
        )
    transition_experts = []
    final_experts = []
    for offset in range(expert_count):
        _initial, transitions, final = _random_wfa_parameters(
            state_count,
            label_count,
            random_seed + offset,
        )
        transition_experts.append(transitions)
        final_experts.append(final)
    return torch.stack(transition_experts, dim=0), torch.stack(final_experts, dim=0)


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
