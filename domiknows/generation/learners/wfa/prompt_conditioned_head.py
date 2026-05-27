"""Prompt-conditioned spectral WFA compact-label learner head."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..common.base import CompactLabelGenerationHead
from ...latent_potentials import LatentTransitionPotential, apply_wfa_transition_potential
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
    _random_wfa_dynamics_experts,
    _random_wfa_parameters,
    _resolve_wfa_label_count,
    _resolve_wfa_state_count,
    _safe_log,
    _seeded_torch_rng,
    _stack_base_and_optional_experts,
    _target_label_batch,
    _target_labels,
    _validate_label,
    _validate_wfa_shapes,
)

__all__ = ["PromptConditionedSpectralWFAGenerationHead"]

class PromptConditionedSpectralWFAGenerationHead(CompactLabelGenerationHead):
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
        random_seed: int | None = 0,
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
        with _seeded_torch_rng(random_seed):
            self.prompt_encoder = _build_prompt_encoder(
                prompt_encoder=prompt_encoder,
                prompt_encoder_type=prompt_encoder_type,
                prompt_vocab_size=prompt_vocab_size,
                prompt_hidden_size=prompt_hidden_size,
                backbone=backbone,
                backbone_hidden_size=backbone_hidden_size,
            )
            self.initial_projector = torch.nn.Linear(self.prompt_encoder.output_size, self.state_count)
        _configure_prompt_encoder_trainability(self.prompt_encoder, trainable)

        _initial, transitions, final = _random_wfa_parameters(self.state_count, self.label_count, random_seed)
        self.transitions = torch.nn.Parameter(transitions, requires_grad=trainable)
        self.final = torch.nn.Parameter(final, requires_grad=trainable)
        if self.dynamics_conditioning == "gated":
            transition_experts, final_experts = _random_wfa_dynamics_experts(
                self.dynamics_expert_count - 1,
                self.state_count,
                self.label_count,
                None if random_seed is None else int(random_seed) + 211,
            )
            self.transition_experts = torch.nn.Parameter(transition_experts, requires_grad=trainable)
            self.final_experts = torch.nn.Parameter(final_experts, requires_grad=trainable)
            with _seeded_torch_rng(None if random_seed is None else int(random_seed) + 1):
                self.dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size, self.dynamics_expert_count)
            for parameter in self.dynamics_gate.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.register_parameter("transition_experts", None)
            self.register_parameter("final_experts", None)
            self.dynamics_gate = None
        if self.step_dynamics_conditioning == "prefix_gated":
            with _seeded_torch_rng(None if random_seed is None else int(random_seed) + 2):
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

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return teacher-forced prompt-conditioned log-probs over labels."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.transitions.device,
            lengths=lengths,
        )
        prompt = _empty_or_prompt(instruction_tokens, self.transitions.device)
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
