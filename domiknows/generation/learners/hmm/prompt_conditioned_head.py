"""Prompt-conditioned discrete HMM compact-label learner head."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..common.base import CompactLabelGenerationHead
from ...latent_potentials import LatentTransitionPotential, apply_hmm_transition_potential
from ..common.utils import (
    TransitionPotentialInput,
    _build_prompt_encoder,
    _coerce_label_to_token_id,
    _configure_prompt_encoder_trainability,
    _empty_or_prompt,
    _first_generated_index,
    _invert_label_to_token_id,
    _normalise_dynamics_conditioning,
    _normalise_flat_ids,
    _normalise_prompt_ids,
    _normalise_step_dynamics_conditioning,
    _positive_int,
    _random_hmm_dynamics_experts,
    _random_hmm_parameters,
    _resolve_label_count,
    _resolve_state_count,
    _safe_log,
    _stack_base_and_optional_experts,
    _target_label_batch,
    _target_labels,
    _validate_hmm_shapes,
    _validate_label,
)

__all__ = ["PromptConditionedHMMGenerationHead"]

class PromptConditionedHMMGenerationHead(CompactLabelGenerationHead):
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

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return teacher-forced prompt-conditioned log-probs over labels."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.transition_logits.device,
            lengths=lengths,
        )
        prompt = _empty_or_prompt(instruction_tokens, self.transition_logits.device)
        rows = []
        for batch_index, row in enumerate(labels):
            prompt_row = prompt if prompt.shape[0] == 1 else prompt[batch_index : batch_index + 1]
            rows.append(
                self.forward(
                    None,
                    prompt_row,
                    row,
                    transition_potential=transition_potential,
                )
            )
        result = torch.stack(rows, dim=0)
        mask = (torch.arange(result.shape[1], device=result.device).unsqueeze(0) < lengths_t.unsqueeze(1)).unsqueeze(-1)
        result = result * mask.to(result.dtype)
        return result[0] if squeeze else result

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
