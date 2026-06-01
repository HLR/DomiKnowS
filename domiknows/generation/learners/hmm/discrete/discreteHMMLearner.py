"""Discrete HMM compact-label learner head (unified unconditional + prompt-conditioned)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .discreteHMM import DiscreteHMM
from ....latent import LatentTransitionPotential, apply_hmm_transition_potential
from ...common.base import CompactLabelGenerationHead
from ...common.utils import (
    TransitionPotentialInput,
    _build_prompt_encoder,
    _coerce_label_to_token_id,
    _configure_prompt_encoder_trainability,
    _empty_or_prompt,
    _first_generated_index,
    _invert_label_to_token_id,
    _labels_from_input_ids,
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
    _seeded_torch_rng,
    _stack_base_and_optional_experts,
    _target_label_batch,
    _target_labels,
    _validate_hmm_shapes,
    _validate_label,
)

__all__ = ["HMMGenerationHead"]


def _normalise_prompt_conditioning(value: str) -> str:
    """Normalize the ``prompt_conditioning`` flag the same way other heads do."""
    text = str(value).replace("-", "_").lower()
    if text not in {"none", "initial"}:
        raise ValueError("prompt_conditioning must be 'none' or 'initial'")
    return text


class HMMGenerationHead(CompactLabelGenerationHead):
    """Compact-label generation head backed by a discrete HMM.

    A single class covers both unconditional and prompt-conditioned discrete
    HMMs, selected by *prompt_conditioning*:

    * ``prompt_conditioning="none"`` (default) — vanilla HMM head.  Trainable
      ``initial_logits`` / ``transition_logits`` / ``emission_logits``.  The
      ``forward`` signature still accepts an ``instruction_tokens`` positional
      so the sensor wiring matches the prompt-conditioned mode, but the prompt
      is ignored.  Equivalent to the legacy ``HMMGenerationHead``.
    * ``prompt_conditioning="initial"`` — the initial hidden-state distribution
      is produced from a trainable prompt encoder + ``initial_projector``.
      Optional discrete-HMM-specific knobs ``dynamics_conditioning="gated"``
      and ``step_dynamics_conditioning="prefix_gated"`` route the transition
      and emission matrices through prompt- (and prefix-) gated experts.
      Equivalent to the legacy ``PromptConditionedHMMGenerationHead``.

    Example (unconditional)::

        head = HMMGenerationHead(label_count=6, state_count=3, pad_size=6)

    Example (prompt-conditioned with gated dynamics)::

        head = HMMGenerationHead(
            label_count=6, state_count=3, pad_size=6,
            prompt_conditioning="initial",
            prompt_vocab_size=128,
            dynamics_conditioning="gated",
            dynamics_expert_count=3,
        )
    """

    def __init__(
        self,
        model: DiscreteHMM | None = None,
        *,
        # Core HMM shape
        label_count: int | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int | None = 0,
        # Prompt conditioning (mirrors GraphHMMGenerationHead)
        prompt_conditioning: str = "none",
        prompt_encoder: torch.nn.Module | None = None,
        prompt_encoder_type: str = "embedding",
        prompt_vocab_size: int = 1024,
        prompt_hidden_size: int = 16,
        backbone: torch.nn.Module | None = None,
        backbone_hidden_size: int | None = None,
        # Discrete-only: prompt-gated dynamics
        dynamics_conditioning: str = "none",
        dynamics_expert_count: int = 2,
        step_dynamics_conditioning: str = "none",
    ):
        super().__init__()
        # --- Static shapes / label maps shared by both modes ---
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _resolve_label_count(model, label_count)
        self.state_count = _resolve_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        # --- Mode normalisation + cross-flag validation ---
        self.prompt_conditioning = _normalise_prompt_conditioning(prompt_conditioning)
        self.dynamics_conditioning = _normalise_dynamics_conditioning(dynamics_conditioning)
        self.step_dynamics_conditioning = _normalise_step_dynamics_conditioning(step_dynamics_conditioning)
        if self.step_dynamics_conditioning != "none" and self.dynamics_conditioning != "gated":
            raise ValueError("step_dynamics_conditioning='prefix_gated' requires dynamics_conditioning='gated'")
        if self.prompt_conditioning == "none":
            if self.dynamics_conditioning != "none" or self.step_dynamics_conditioning != "none":
                raise ValueError(
                    "dynamics_conditioning / step_dynamics_conditioning require "
                    "prompt_conditioning='initial' — the gating signal comes from the prompt encoder"
                )
            if model is None:
                # Initialize a random, normalized HMM when no prebuilt model is supplied.
                initial, transition, emission = _random_hmm_parameters(
                    self.state_count,
                    self.label_count,
                    random_seed,
                )
            else:
                # Reuse provided model parameters after shape and dtype normalization.
                initial = torch.as_tensor(model.initial, dtype=torch.float32)
                transition = torch.as_tensor(model.transition, dtype=torch.float32)
                emission = torch.as_tensor(model.emission, dtype=torch.float32)
                _validate_hmm_shapes(initial, transition, emission, self.state_count, self.label_count)
        else:
            if model is not None:
                raise ValueError(
                    "model=DiscreteHMM(...) is only supported when prompt_conditioning='none'; "
                    "the prompt-conditioned head always initialises its own parameters"
                )
            # Prompt-conditioned mode: ignore the random initial vector (initial state
            # comes from the prompt encoder + initial_projector) and only seed
            # transition / emission from _random_hmm_parameters.
            _initial, transition, emission = _random_hmm_parameters(
                self.state_count, self.label_count, random_seed,
            )
            initial = None  # placeholder; not used in this mode

        self.dynamics_expert_count = (
            _positive_int(dynamics_expert_count, "dynamics_expert_count")
            if self.dynamics_conditioning == "gated"
            else 1
        )

        # --- Trainable HMM logits ---
        if self.prompt_conditioning == "none":
            # Vanilla HMM keeps initial_logits as a learnable parameter.
            self.initial_logits = torch.nn.Parameter(_safe_log(initial), requires_grad=trainable)
        else:
            # Prompt-conditioned mode replaces initial_logits with the
            # initial_projector head — register the attribute as None so the
            # property below can dispatch on its presence.
            self.register_parameter("initial_logits", None)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission), requires_grad=trainable)

        # --- Prompt-conditioned scaffolding ---
        if self.prompt_conditioning == "initial":
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
            for parameter in self.initial_projector.parameters():
                parameter.requires_grad_(trainable)

            if self.dynamics_conditioning == "gated":
                transition_experts, emission_experts = _random_hmm_dynamics_experts(
                    self.dynamics_expert_count - 1,
                    self.state_count,
                    self.label_count,
                    None if random_seed is None else int(random_seed) + 101,
                )
                self.transition_expert_logits = torch.nn.Parameter(transition_experts, requires_grad=trainable)
                self.emission_expert_logits = torch.nn.Parameter(emission_experts, requires_grad=trainable)
                with _seeded_torch_rng(None if random_seed is None else int(random_seed) + 1):
                    self.dynamics_gate = torch.nn.Linear(self.prompt_encoder.output_size, self.dynamics_expert_count)
                for parameter in self.dynamics_gate.parameters():
                    parameter.requires_grad_(trainable)
            else:
                self.register_parameter("transition_expert_logits", None)
                self.register_parameter("emission_expert_logits", None)
                self.dynamics_gate = None

            if self.step_dynamics_conditioning == "prefix_gated":
                with _seeded_torch_rng(None if random_seed is None else int(random_seed) + 2):
                    self.prefix_embedding = torch.nn.Embedding(self.label_count, self.prompt_encoder.output_size)
                    self.step_dynamics_gate = torch.nn.Linear(
                        self.prompt_encoder.output_size * 2,
                        self.dynamics_expert_count,
                    )
                for parameter in self.prefix_embedding.parameters():
                    parameter.requires_grad_(trainable)
                for parameter in self.step_dynamics_gate.parameters():
                    parameter.requires_grad_(trainable)
            else:
                self.prefix_embedding = None
                self.step_dynamics_gate = None
        else:
            # Unconditional mode: no prompt-related submodules.
            self.prompt_encoder = None
            self.initial_projector = None
            self.register_parameter("transition_expert_logits", None)
            self.register_parameter("emission_expert_logits", None)
            self.dynamics_gate = None
            self.prefix_embedding = None
            self.step_dynamics_gate = None

    # ------------------------------------------------------------------ #
    # Shared properties / latent-potential helpers                       #
    # ------------------------------------------------------------------ #

    @property
    def initial_probs(self) -> torch.Tensor:
        """Return the (prompt-independent) initial hidden-state distribution.

        Only defined when ``prompt_conditioning='none'``.  In the prompt-
        conditioned mode, call :meth:`prompt_initial_probs` instead, which
        depends on the supplied instruction tokens.
        """
        if self.prompt_conditioning != "none":
            raise AttributeError(
                "initial_probs is undefined when prompt_conditioning='initial'; "
                "use prompt_initial_probs(instruction_tokens) to get the prompt-conditioned initial state"
            )
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        # State-to-state transition matrix.
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        # State-to-label emission matrix.
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

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ #
    # Prompt-conditioned helpers (only valid for prompt_conditioning='initial') #
    # ------------------------------------------------------------------ #

    def _require_prompt_mode(self, method_name: str) -> None:
        if self.prompt_conditioning == "none":
            raise RuntimeError(
                f"{method_name} requires prompt_conditioning='initial'; "
                "this head was constructed without prompt conditioning"
            )

    def _prompt_features(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        self._require_prompt_mode("_prompt_features")
        prompt = _normalise_prompt_ids(instruction_tokens, device=self._parameter_device())
        return self.prompt_encoder(prompt)

    def prompt_initial_probs(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return prompt-conditioned initial HMM state probabilities.

        Returns a vector ``(state_count,)`` for a single prompt and a matrix
        ``(batch, state_count)`` when *instruction_tokens* is a true batched
        prompt tensor with ``batch > 1``.
        """
        self._require_prompt_mode("prompt_initial_probs")
        features = self._prompt_features(instruction_tokens)
        logits = self.initial_projector(features)
        probs = torch.softmax(logits, dim=-1)
        return probs[0] if probs.shape[0] == 1 else probs

    def prompt_dynamics_weights(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return prompt-selected HMM dynamics expert weights."""
        self._require_prompt_mode("prompt_dynamics_weights")
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
        self._require_prompt_mode("prompt_transition_probs")
        if self.dynamics_conditioning != "gated":
            return self.transition_probs_with_potential(transition_potential)
        weights = self.prompt_dynamics_weights(instruction_tokens)
        experts = _stack_base_and_optional_experts(self.transition_logits, self.transition_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return apply_hmm_transition_potential(torch.softmax(logits, dim=-1), transition_potential)

    def prompt_emission_probs(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return the prompt-conditioned HMM emission matrix."""
        self._require_prompt_mode("prompt_emission_probs")
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
        self._require_prompt_mode("step_dynamics_weights")
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
        self._require_prompt_mode("step_transition_probs")
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
        self._require_prompt_mode("step_emission_probs")
        if self.step_dynamics_conditioning == "none":
            return self.prompt_emission_probs(instruction_tokens)
        weights = self.step_dynamics_weights(instruction_tokens, prefix_labels)
        experts = _stack_base_and_optional_experts(self.emission_logits, self.emission_expert_logits)
        logits = torch.einsum("e,eij->ij", weights, experts)
        return torch.softmax(logits, dim=-1)

    def _prefix_summary(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        if self.prefix_embedding is None or not prefix_labels:
            return torch.zeros(
                self.prompt_encoder.output_size,
                dtype=self.transition_logits.dtype,
                device=self._parameter_device(),
            )
        labels = torch.tensor(
            [_validate_label(label, self.label_count) for label in prefix_labels],
            dtype=torch.long,
            device=self._parameter_device(),
        )
        return self.prefix_embedding(labels).mean(dim=0)

    # ------------------------------------------------------------------ #
    # Decoding-side helpers                                              #
    # ------------------------------------------------------------------ #

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or ids[:1]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        return torch.tensor([prompt_ids], dtype=torch.long, device=device), labels

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        # Unconditional forward filter (legacy HMMGenerationHead path).
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

    def _next_logits_from_prompt_and_prefix(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        # Prompt-conditioned forward filter; supports prefix-gated dynamics.
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
        **kwargs,
    ) -> torch.Tensor:
        """Return next-step logits over compact generation labels."""
        if transition_potential is None:
            transition_potential = kwargs.get("transition_potential")
        if self.prompt_conditioning == "none":
            return self._next_logits_from_prefix_labels(
                self._labels_from_input_ids(input_ids),
                transition_potential=transition_potential,
            )
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        return self._next_logits_from_prompt_and_prefix(
            prompt_ids, prefix_labels, transition_potential=transition_potential,
        )

    # ------------------------------------------------------------------ #
    # Production view (unconditional only)                               #
    # ------------------------------------------------------------------ #

    def production_hmm(
        self,
        transition_potential: TransitionPotentialInput = None,
        *,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> DiscreteHMM:
        """Return a Torch-backed HMM view of the current head parameters.

        When ``prompt_conditioning='initial'``, *instruction_tokens* must be
        supplied; the snapshot uses :meth:`prompt_initial_probs`,
        :meth:`prompt_transition_probs`, and :meth:`prompt_emission_probs`.
        """
        if self.prompt_conditioning == "none":
            return DiscreteHMM(
                self.transition_probs_with_potential(transition_potential),
                self.emission_probs,
                self.initial_probs,
                tuple(range(self.label_count)),
                normalize=False,
            )
        if instruction_tokens is None:
            raise ValueError(
                "production_hmm requires instruction_tokens when prompt_conditioning='initial'"
            )
        return DiscreteHMM(
            self.prompt_transition_probs(instruction_tokens, transition_potential=transition_potential),
            self.prompt_emission_probs(instruction_tokens),
            self.prompt_initial_probs(instruction_tokens),
            tuple(range(self.label_count)),
            normalize=False,
        )

    # ------------------------------------------------------------------ #
    # Teacher-forced sequence log-probs                                  #
    # ------------------------------------------------------------------ #

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
        **kwargs,
    ) -> torch.Tensor:
        """Teacher-forced log-probs shaped ``[batch, seq, label_count]``.

        Padded positions (those beyond *lengths*) are zeroed out so they
        contribute a constant — and therefore gradient-free — term to a
        downstream cross-entropy loss.
        """
        if transition_potential is None:
            transition_potential = kwargs.get("transition_potential")
        if self.prompt_conditioning == "none":
            return self._sequence_log_probs_unconditional(
                target_labels,
                lengths=lengths,
                transition_potential=transition_potential,
            )
        return self._sequence_log_probs_prompt_conditioned(
            target_labels,
            lengths=lengths,
            instruction_tokens=instruction_tokens,
            transition_potential=transition_potential,
        )

    def _sequence_log_probs_unconditional(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None,
        transition_potential: TransitionPotentialInput,
    ) -> torch.Tensor:
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
        mask = (
            torch.arange(result.shape[1], device=result.device).unsqueeze(0)
            < lengths_t.unsqueeze(1)
        ).unsqueeze(-1)
        result = result * mask.to(result.dtype)
        return result[0] if squeeze else result

    def _sequence_log_probs_prompt_conditioned(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None,
        instruction_tokens: torch.Tensor | Sequence[int] | None,
        transition_potential: TransitionPotentialInput,
    ) -> torch.Tensor:
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
        mask = (
            torch.arange(result.shape[1], device=result.device).unsqueeze(0)
            < lengths_t.unsqueeze(1)
        ).unsqueeze(-1)
        result = result * mask.to(result.dtype)
        return result[0] if squeeze else result

    # ------------------------------------------------------------------ #
    # forward — DomiKnowS sensor entry point                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
        **kwargs,
    ):
        """Teacher-forced log-probs for ``pad_size`` autoregressive steps.

        When ``prompt_conditioning='none'`` the *instruction_tokens* argument
        is accepted for sensor compatibility but ignored.  When
        ``prompt_conditioning='initial'``, uses a single forward-filter sweep
        (linear in ``pad_size``) unless ``step_dynamics_conditioning`` is
        active, in which case emission/transition are recomputed per step
        from the growing prefix.
        """
        if self.prompt_conditioning == "none":
            return self.sequence_log_probs(
                target_labels,
                transition_potential=transition_potential,
                **kwargs,
            )
        if self.step_dynamics_conditioning == "none":
            return self._forward_static(instruction_tokens, target_labels, transition_potential)
        return self._forward_prefix_gated(instruction_tokens, target_labels, transition_potential)

    def _forward_static(
        self,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput,
    ) -> torch.Tensor:
        # Emission and transition are prefix-independent; compute once.
        labels = _target_labels(target_labels, self.pad_size, device=self.transition_logits.device)
        emission = self.prompt_emission_probs(instruction_tokens)
        transition = self.prompt_transition_probs(
            instruction_tokens, transition_potential=transition_potential
        )
        state = self.prompt_initial_probs(instruction_tokens)
        eps = torch.finfo(self.emission_logits.dtype).eps
        outputs = []
        for step in range(self.pad_size):
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            label = int(labels[step].item())
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)
        return torch.log_softmax(torch.stack(outputs, dim=0), dim=-1)

    def _forward_prefix_gated(
        self,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput,
    ) -> torch.Tensor:
        # Emission/transition depend on the generated prefix; recompute per step.
        labels = _target_labels(target_labels, self.pad_size, device=self.transition_logits.device)
        state = self.prompt_initial_probs(instruction_tokens)
        eps = torch.finfo(self.emission_logits.dtype).eps
        outputs = []
        consumed: list[int] = []
        for step in range(self.pad_size):
            emission = self.step_emission_probs(instruction_tokens, consumed)
            transition = self.step_transition_probs(
                instruction_tokens, consumed, transition_potential=transition_potential
            )
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            label = int(labels[step].item())
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)
            consumed.append(label)
        return torch.log_softmax(torch.stack(outputs, dim=0), dim=-1)
