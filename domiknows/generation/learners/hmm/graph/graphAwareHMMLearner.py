"""Graph-HMM compact-label learner head."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Mapping

import torch

from ...common.base import CompactLabelGenerationHead
from ...common.utils import (
    _coerce_label_to_token_id,
    _first_generated_index,
    _invert_label_to_token_id,
    _seeded_torch_rng,
    _validate_label,
)
from .constraints import ConstraintApplicationReport, project_matrix_rows, validate_mask
from .dynamic import DynamicConstraintContext, FactorizedStateSpace, apply_transition_energy, transition_energy_matrix
from .graph_head_utils import (
    _flat_input_ids,
    _labels_from_input_ids,
    _normalize_vector,
    _normalise_prompt_ids,
    _random_hmm_parameters,
    _safe_log,
    _target_label_batch,
    _validate_hmm_shapes,
)

__all__ = ["GraphHMMGenerationHead"]

class GraphHMMGenerationHead(CompactLabelGenerationHead):
    """Compact-label HMM head that projects parameters through graph constraints."""

    @classmethod
    def from_bundle(
        cls,
        bundle,
        *,
        graph=None,
        dfa=None,
        trainable: bool = True,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        prompt_conditioning: str = "none",
        prompt_vocab_size: int = 128,
        prompt_hidden_size: int = 16,
        random_seed: int | None = None,
    ) -> "GraphHMMGenerationHead":
        """Build a compact head directly from a generation bundle.

        If ``dfa`` is not provided, the method falls back to the bundle graph
        for DFA-first HMM compilation.
        """

        from .constraint_compiler import domiknows_hmm_from_generation_constraints
        from .graphAwareHMM import DomiKnowSAwareHMM

        if dfa is not None:
            fitted_hmm = DomiKnowSAwareHMM.from_dfa(bundle, dfa)
        else:
            if graph is None:
                graph = getattr(bundle, "graph", None)
            if graph is None:
                raise ValueError("from_bundle requires graph or dfa support")
            fitted_hmm = domiknows_hmm_from_generation_constraints(
                graph,
                bundle,
                dtype=torch.float64,
            )
        return cls.from_graph_hmm(
            fitted_hmm,
            trainable=trainable,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            prompt_conditioning=prompt_conditioning,
            prompt_vocab_size=prompt_vocab_size,
            prompt_hidden_size=prompt_hidden_size,
            random_seed=random_seed,
        )

    def __init__(
        self,
        *,
        n_hidden_states: int,
        label_count: int,
        symbols: Sequence[Any] | None = None,
        state_names: Sequence[str] | None = None,
        transition_mask=None,
        emission_mask=None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int | None = 0,
        initial=None,
        transition=None,
        emission=None,
        dynamic_transition: Callable[[DynamicConstraintContext], Any] | None = None,
        transition_energy: Callable[[DynamicConstraintContext], Any] | None = None,
        energy_weight: float = 1.0,
        state_space: FactorizedStateSpace | None = None,
        dynamic_metadata: Mapping[str, Any] | None = None,
        prompt_conditioning: str = "none",
        prompt_vocab_size: int = 128,
        prompt_hidden_size: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize a compact-label HMM head with graph-constrained support.

        Parameters
        ----------
        n_hidden_states:
            Number of latent HMM states.
        label_count:
            Size of the compact label alphabet handled by this head.
        symbols:
            Optional human-readable symbol names for labels.
        state_names:
            Optional latent-state names used in diagnostics and debugging.
        transition_mask:
            Hard support mask over latent-state transitions.
        emission_mask:
            Hard support mask over latent-state emissions.
        pad_size:
            Maximum prefix length used when splitting prompt/prefix inputs.
        label_to_token_id:
            Mapping from compact labels to tokenizer token ids.
        trainable:
            Whether the logits and prompt-conditioning layers require gradients.
        random_seed:
            Seed used for deterministic parameter initialization.
        initial:
            Optional initial-state probabilities.
        transition:
            Optional transition probability matrix.
        emission:
            Optional emission probability matrix.
        dynamic_transition:
            Optional callback that provides per-step hard transition support.
        transition_energy:
            Optional callback that provides per-step soft transition penalties.
        energy_weight:
            Scale applied to the soft transition-energy penalty.
        state_space:
            Optional factorized state space used for diagnostics and dynamics.
        dynamic_metadata:
            Extra metadata forwarded to dynamic callbacks.
        prompt_conditioning:
            Prompt-conditioning mode for the initial state.
        prompt_vocab_size:
            Size of the prompt token vocabulary when prompt conditioning is on.
        prompt_hidden_size:
            Hidden size of the prompt encoder when prompt conditioning is on.
        dtype:
            Tensor dtype used for mask validation and parameter initialization.
        """
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
        self.prompt_conditioning = str(prompt_conditioning).replace("-", "_").lower()
        if self.prompt_conditioning not in {"none", "initial"}:
            raise ValueError("prompt_conditioning must be 'none' or 'initial'")
        self.prompt_vocab_size = int(prompt_vocab_size)
        self.prompt_hidden_size = int(prompt_hidden_size)
        if self.prompt_conditioning != "none":
            if self.prompt_vocab_size < 1:
                raise ValueError("prompt_vocab_size must be positive")
            if self.prompt_hidden_size < 1:
                raise ValueError("prompt_hidden_size must be positive")
            with _seeded_torch_rng(random_seed):
                self.prompt_embedding = torch.nn.Embedding(self.prompt_vocab_size, self.prompt_hidden_size)
                self.prompt_initial_projector = torch.nn.Linear(self.prompt_hidden_size, self.n_hidden_states)
            torch.nn.init.zeros_(self.prompt_initial_projector.weight)
            torch.nn.init.zeros_(self.prompt_initial_projector.bias)
            for parameter in self.prompt_embedding.parameters():
                parameter.requires_grad_(trainable)
            for parameter in self.prompt_initial_projector.parameters():
                parameter.requires_grad_(trainable)
        else:
            self.prompt_embedding = None
            self.prompt_initial_projector = None

        if transition_mask is None or emission_mask is None:
            raise ValueError(
                "transition_mask and emission_mask are required; "
                "construct masks via DFA/plan compilers before creating GraphHMMGenerationHead"
            )

        # Step 1: Validate and materialize the hard-support masks up front.
        transition_mask_t = validate_mask(
            transition_mask,
            (self.n_hidden_states, self.n_hidden_states),
            name="transition_mask",
            dtype=dtype,
        ).to(dtype=torch.float32)
        emission_mask_t = validate_mask(
            emission_mask,
            (self.n_hidden_states, self.label_count),
            name="emission_mask",
            dtype=dtype,
        ).to(dtype=torch.float32)
        self.register_buffer("transition_mask", transition_mask_t)
        self.register_buffer("emission_mask", emission_mask_t)
        self.constraint_report = ConstraintApplicationReport()

        # Step 2: Either use caller-provided probabilities or build a seeded
        # random initialization that already respects the masks.
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
        prompt_conditioning: str = "none",
        prompt_vocab_size: int = 128,
        prompt_hidden_size: int = 16,
        random_seed: int | None = None,
    ) -> "GraphHMMGenerationHead":
        """Create a PMD head initialized from a fitted ``DomiKnowSAwareHMM``."""
        learner._require_fitted()
        # Reuse the fitted learner's support and parameter tensors directly so
        # the head starts from the same constrained distribution.
        head = cls(
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
            prompt_conditioning=prompt_conditioning,
            prompt_vocab_size=prompt_vocab_size,
            prompt_hidden_size=prompt_hidden_size,
            random_seed=random_seed,
            dtype=learner.dtype,
        )
        head.apply_seeded_logit_jitter(random_seed)
        return head

    def apply_seeded_logit_jitter(self, random_seed: int | None, *, scale: float = 0.01) -> None:
        """Apply deterministic small logit noise for seeded trainable starts.

        ``from_graph_hmm`` initializes from graph-constrained fitted tensors,
        so the constructor's random parameter path is not used. This method
        gives callers the same single-argument seeding ergonomics as the neural
        compact heads while keeping ``random_seed=None`` exactly deterministic.
        """
        if random_seed is None:
            return
        # Keep the perturbation reproducible without changing the public state
        # distribution in a large or unstable way.
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(random_seed))
            self.initial_logits.add_(float(scale) * torch.randn_like(self.initial_logits))
            self.transition_logits.add_(float(scale) * torch.randn_like(self.transition_logits))
            self.emission_logits.add_(float(scale) * torch.randn_like(self.emission_logits))

    @property
    def initial_probs(self) -> torch.Tensor:
        return _normalize_vector(torch.exp(self.initial_logits))

    @property
    def transition_probs(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1) * self.transition_mask

    @property
    def emission_probs(self) -> torch.Tensor:
        return torch.softmax(self.emission_logits, dim=-1) * self.emission_mask

    def _projected_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial = _normalize_vector(torch.exp(self.initial_logits))
        transition = torch.softmax(self.transition_logits, dim=-1) * self.transition_mask
        emission = torch.softmax(self.emission_logits, dim=-1) * self.emission_mask
        transition = project_matrix_rows(transition, self.transition_mask)
        emission = project_matrix_rows(emission, self.emission_mask)
        return initial, transition, emission

    def _dynamic_transition_matrix(self, *, step: int, prompt_ids: torch.Tensor | None, prefix_labels: torch.Tensor, belief: torch.Tensor | None, input_ids: torch.Tensor | None) -> torch.Tensor:
        transition = torch.softmax(self.transition_logits, dim=-1) * self.transition_mask
        if self.dynamic_transition is None and self.transition_energy is None:
            return project_matrix_rows(transition, self.transition_mask)
        context = DynamicConstraintContext(
            step=step,
            prefix=tuple(prefix_labels.tolist()) if hasattr(prefix_labels, "tolist") else tuple(prefix_labels),
            belief=None if belief is None else belief.detach().clone(),
            sequence=None if input_ids is None else tuple(input_ids.tolist()) if hasattr(input_ids, "tolist") else tuple(input_ids),
            metadata={
                "state_names": self.state_names,
                "symbols": self.symbols,
                "state_space": self.state_space,
                **self.dynamic_metadata,
            },
        )
        weighted = transition
        effective_mask = self.transition_mask
        if self.dynamic_transition is not None:
            dynamic = self.dynamic_transition(context)
            if dynamic is not None:
                factor = validate_mask(
                    dynamic,
                    (self.n_hidden_states, self.n_hidden_states),
                    name="dynamic_transition",
                    dtype=self.transition_mask.dtype,
                )
                weighted = weighted * factor
                effective_mask = effective_mask * (factor > 0).to(dtype=self.transition_mask.dtype)
        if self.transition_energy is not None:
            energy = self.transition_energy(context)
            if energy is not None:
                weighted = apply_transition_energy(
                    weighted,
                    transition_energy_matrix(
                        energy,
                        shape=(self.n_hidden_states, self.n_hidden_states),
                        dtype=self.transition_mask.dtype,
                    ),
                    weight=self.energy_weight,
                )
        return project_matrix_rows(weighted, effective_mask)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _initial_distribution(self, prompt_ids: torch.Tensor | None = None) -> torch.Tensor:
        initial = _normalize_vector(torch.exp(self.initial_logits))
        if self.prompt_conditioning == "none" or prompt_ids is None:
            return initial
        prompt_ids = _normalise_prompt_ids(prompt_ids, self.prompt_vocab_size)
        prompt_hidden = self.prompt_embedding(prompt_ids).mean(dim=1)
        prompt_logits = self.prompt_initial_projector(prompt_hidden)
        prompt_bias = torch.softmax(prompt_logits, dim=-1)
        return _normalize_vector(initial * prompt_bias)

    def _build_prompt_context(self, input_ids: torch.Tensor | Sequence[int] | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if input_ids is None:
            return None, None
        flat_ids = _flat_input_ids(input_ids)
        if flat_ids.numel() == 0:
            return None, None
        prompt_ids = flat_ids[:, : self.pad_size]
        return flat_ids, prompt_ids

    def _target_labels(self, input_ids: torch.Tensor | Sequence[int] | None, labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return _target_label_batch(labels, self.label_count)

    def _labels_from_inputs(self, input_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        return _labels_from_input_ids(input_ids, self.label_count)

    def _first_generated_index(self, input_ids: torch.Tensor | Sequence[int] | None) -> int:
        return _first_generated_index(input_ids, self.pad_size)

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

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths=None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Compute per-step log-probabilities for one batch of label sequences."""
        labels, _lengths_t, step_mask, squeeze = _target_label_batch(target_labels, self.pad_size, lengths=lengths)
        batch, seq_len = labels.shape
        # Start from the initial latent belief, optionally conditioned on the prompt.
        state = self._initial_probs_for_prompt(instruction_tokens, batch)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        outputs = []
        prefixes: list[list[int]] = [[] for _ in range(batch)]
        for step in range(seq_len):
            # Predict the next-label distribution from the current latent belief.
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            # Update the hidden-state belief after observing the current label.
            posterior = state * emission[:, labels[:, step]].transpose(0, 1)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_states = []
            for batch_index in range(batch):
                # Apply any step-specific transition constraint before advancing.
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

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **_kwargs) -> torch.Tensor:
        """Return next-label logits for a decoded prefix of token ids."""
        if self.prompt_conditioning == "none":
            prompt_ids = None
            prefix_labels = _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)
        else:
            prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        # Recover the current latent belief by replaying the observed prefix.
        state = self._initial_probs_for_prompt(prompt_ids, 1)[0]
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        for step, label in enumerate(prefix_labels):
            # Consume one observed label and advance the belief one step.
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            transition = self._transition_for_prefix(step=step, prefix=tuple(prefix_labels[: step + 1]), belief=posterior)
            state = torch.matmul(posterior, transition)
        # Project the final belief to the next-label distribution.
        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor, **_kwargs):
        """PMD module interface: returns sequence log-probabilities."""
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    def _initial_probs_for_prompt(
        self,
        instruction_tokens: torch.Tensor | Sequence[int] | None,
        batch_size: int,
    ) -> torch.Tensor:
        """Return initial state probabilities, optionally conditioned on prompt ids."""
        base = self.initial_logits.reshape(1, -1).expand(batch_size, -1)
        if self.prompt_conditioning == "none" or instruction_tokens is None:
            return torch.softmax(base, dim=-1)
        # Convert the prompt to token ids, then summarize it into a prompt-level
        # latent feature vector.
        prompt = _normalise_prompt_ids(instruction_tokens, device=self.initial_logits.device)
        if torch.any(prompt < 0) or torch.any(prompt >= self.prompt_vocab_size):
            raise ValueError(f"instruction_tokens contains ids outside prompt_vocab_size={self.prompt_vocab_size}")
        features = self.prompt_embedding(prompt).mean(dim=1)
        if features.shape[0] == 1 and batch_size > 1:
            features = features.expand(batch_size, -1)
        if features.shape[0] != batch_size:
            raise ValueError("instruction_tokens batch size must be 1 or match target_labels")
        # Combine the prompt-conditioned bias with the base initial logits.
        return torch.softmax(base + self.prompt_initial_projector(features), dim=-1)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        # Separate prompt tokens from already-generated labels using the label map.
        ids = _flat_input_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or [0]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        return torch.tensor([prompt_ids], dtype=torch.long, device=self.initial_logits.device), labels

    def _transition_for_prefix(self, *, step: int, prefix: tuple[int, ...], belief: torch.Tensor | None) -> torch.Tensor:
        """Build per-step transition matrix under optional dynamic constraints."""
        # Begin with the globally legal transition distribution.
        transition = self.transition_probs
        if self.dynamic_transition is None and self.transition_energy is None:
            return transition
        # Build the per-step dynamic context using decoded symbols, not raw ids.
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
            # Apply hard multiplicative transition compatibility from the callback.
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
            # Apply a soft penalty by multiplying by exp(-weight * energy).
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
