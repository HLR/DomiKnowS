"""Strict HMM+DFA product decoder for hybrid generation."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .hmm_utils import HMMRuntimeView, hmm_next_label_logits, lookahead_remaining_after_label, resolve_hmm_snapshot
from ..dfa.core import DFA
from ..dfa.stop_policy import (
    DecodeProgress,
    StopPolicy,
    make_progress_tracker,
    remaining_steps_for,
    should_stop_decoding,
    should_stop_on_token,
    stop_policy_from_legacy,
)
from ..dfa.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class HMMDFADecodeResult:
    """Direct output of strict HMM+DFA product decoding."""

    text: str | None
    token_ids: list[int]
    labels: list[int]
    final_state: Any
    accepted: bool
    score: float
    scores: list[float]
    final_hmm_belief: torch.Tensor
    search: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HMMDFAProductState:
    """Explicit product state for strict HMM+DFA decoding."""

    hmm_belief: torch.Tensor
    dfa_state: Any
    full_ids: list[int]
    labels: list[int]
    score: float
    scores: list[float] = field(default_factory=list)
    last_dfa_state_change_step: int = 0


@dataclass(frozen=True)
class HMMDFAStepScores:
    """Shared next-label scores and cached product transitions."""

    masked_logits: torch.Tensor
    allowed: set[int]
    next_beliefs: Mapping[int, torch.Tensor] = field(default_factory=dict)
    next_dfa_states: Mapping[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HMMDFAStaticLookahead:
    """Static HMM+DFA success table indexed by depth and DFA state."""

    values: list[dict[Any, torch.Tensor]]
    max_depth: int

    def success_after(self, belief: torch.Tensor, dfa_state, remaining_steps: int | None) -> torch.Tensor:
        """Return success probability from a belief and DFA state."""
        depth = self.max_depth if remaining_steps is None else min(max(0, int(remaining_steps)), self.max_depth)
        vector = self.values[depth].get(dfa_state)
        if vector is None:
            return belief.new_tensor(0.0)
        vector = vector.to(device=belief.device, dtype=belief.dtype)
        return torch.dot(belief, vector).clamp(0.0, 1.0)


class HMMDFADecoder:
    """Decode strict product paths over an HMM belief state and DFA state."""

    def __init__(
        self,
        *,
        dfa: DFA,
        vocabulary: TokenVocabulary,
        generator,
        scorer_head,
        tokenizer=None,
        backend_label_logits: Callable[..., torch.Tensor | None],
        emittable_labels: Callable[[Any, TokenVocabulary], set[int]],
        flat_ids: Callable[[torch.Tensor | Sequence[int]], list[int]],
        mask_label_logits: Callable[[torch.Tensor, set[int]], torch.Tensor],
        select_label: Callable[..., tuple[int, float]],
        token_id_for_generated_label: Callable[[Any, TokenVocabulary, int], int],
    ):
        self.dfa = dfa
        self.vocabulary = vocabulary
        self.generator = generator
        self.scorer_head = scorer_head
        self.tokenizer = tokenizer
        self._backend_label_logits = backend_label_logits
        self._emittable_labels = emittable_labels
        self._flat_ids = flat_ids
        self._mask_label_logits = mask_label_logits
        self._select_label = select_label
        self._token_id_for_generated_label = token_id_for_generated_label

    def decode_hmm_dfa(
        self,
        prompt_ids: torch.Tensor | Sequence[int],
        *,
        search: str = "beam",
        num_return_sequences: int = 1,
        beam_size: int = 4,
        max_new_tokens: int | None = 16,
        stop_policy: StopPolicy | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator_seed: int = 0,
        keep_rejected: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs,
    ) -> list[HMMDFADecodeResult]:
        """Direct strict HMM+DFA decoding over explicit product states."""
        if self.scorer_head is None:
            raise ValueError("decode_hmm_dfa requires scorer_head")
        search_n = str(search).strip().lower().replace("-", "_")
        if search_n not in {"beam", "greedy", "sample"}:
            raise ValueError("search must be one of 'beam', 'greedy', or 'sample'")
        if int(num_return_sequences) < 1:
            raise ValueError("num_return_sequences must be at least 1")
        if int(beam_size) < 1:
            raise ValueError("beam_size must be at least 1")
        if float(length_penalty) <= 0.0:
            raise ValueError("length_penalty must be positive")

        prompt_ids_t = self._prompt_tensor(prompt_ids)
        policy = stop_policy_from_legacy(max_new_tokens=max_new_tokens, stop_policy=stop_policy)
        hmm_weight = float(kwargs.pop("hmm_weight", kwargs.pop("compact_logit_weight", 1.0)))
        hf_weight = float(kwargs.pop("hf_weight", kwargs.pop("backend_logit_weight", 1.0)))
        lookahead_weight = float(kwargs.pop("lookahead_weight", 1.0))
        transition_potential = kwargs.pop("transition_potential", None)
        lookahead_max_steps = kwargs.pop("lookahead_max_steps", max_new_tokens if max_new_tokens is not None else 8)
        lookahead_max_steps = None if lookahead_max_steps is None else int(lookahead_max_steps)
        max_attempts = int(kwargs.pop("product_decode_max_attempts", max(1, int(num_return_sequences) * 4)))
        if kwargs:
            unknown = ", ".join(sorted(str(key) for key in kwargs))
            raise ValueError(f"unsupported decode_hmm_dfa kwargs: {unknown}")

        runtime = resolve_hmm_snapshot(self.scorer_head, prompt_ids_t, transition_potential=transition_potential)
        emittable = self._emittable_labels(self.scorer_head, self.vocabulary)
        eos_label = int(self.vocabulary.eos_label)
        static_lookahead = self._build_static_lookahead(
            runtime,
            policy=policy,
            emittable=emittable,
            eos_label=eos_label,
            lookahead_weight=lookahead_weight,
            lookahead_max_steps=lookahead_max_steps,
        )
        lookahead_backend = "disabled" if not lookahead_weight else ("static_dp" if static_lookahead is not None else "recursive")
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor] = {}
        common = {
            "runtime": runtime,
            "policy": policy,
            "emittable": emittable,
            "eos_label": eos_label,
            "hmm_weight": hmm_weight,
            "hf_weight": hf_weight,
            "lookahead_weight": lookahead_weight,
            "lookahead_max_steps": lookahead_max_steps,
            "static_lookahead": static_lookahead,
            "recursive_cache": recursive_cache,
        }
        if search_n == "greedy":
            states = self._decode_greedy(
                prompt_ids_t,
                keep_rejected=keep_rejected,
                **common,
            )
        elif search_n == "sample":
            states = self._decode_sample(
                prompt_ids_t,
                num_return_sequences=int(num_return_sequences),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator_seed=int(generator_seed),
                keep_rejected=keep_rejected,
                max_attempts=max_attempts,
                **common,
            )
        else:
            states = self._decode_beam(
                prompt_ids_t,
                num_return_sequences=int(num_return_sequences),
                beam_size=int(beam_size),
                length_penalty=float(length_penalty),
                early_stopping=bool(early_stopping),
                keep_rejected=keep_rejected,
                **common,
            )
        return [
            self._decode_result_from_state(
                state,
                prompt_ids_t,
                search=search_n,
                metadata={
                    "decode_strategy": "product_hmm_dfa",
                    "tracks_hmm_belief": True,
                    "backend_logit_integration": "one_token_label_bias" if hf_weight else "none",
                    "hmm_weight": hmm_weight,
                    "hf_weight": hf_weight,
                    "lookahead_weight": lookahead_weight,
                    "lookahead_backend": lookahead_backend,
                },
            )
            for state in states[: int(num_return_sequences)]
        ]

    def _init_product_state(self, prompt_ids: torch.Tensor, runtime: HMMRuntimeView) -> HMMDFAProductState:
        """Initialize the explicit HMM+DFA product state from prompt ids."""
        return HMMDFAProductState(
            hmm_belief=runtime.initial_belief.clone(),
            dfa_state=self.dfa.start_state,
            full_ids=self._flat_ids(prompt_ids),
            labels=[],
            score=0.0,
            scores=[],
            last_dfa_state_change_step=0,
        )

    def _score_next_labels(
        self,
        product_state: HMMDFAProductState,
        runtime: HMMRuntimeView,
        *,
        policy: StopPolicy,
        step_index: int,
        emittable: set[int],
        eos_label: int,
        hmm_weight: float,
        hf_weight: float,
        lookahead_weight: float,
        lookahead_max_steps: int | None,
        static_lookahead: HMMDFAStaticLookahead | None,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
    ) -> HMMDFAStepScores:
        """Score the next compact labels for every HMM+DFA search mode."""
        remaining_steps = remaining_steps_for(policy, step_index)
        allowed = {int(label) for label in self.dfa.allowed_tokens(product_state.dfa_state, remaining_steps=remaining_steps)}
        allowed &= emittable

        emission = runtime.emission_for(product_state.labels).to(
            device=product_state.hmm_belief.device,
            dtype=product_state.hmm_belief.dtype,
        )
        hmm_logits = hmm_next_label_logits(product_state.hmm_belief, emission)
        if not allowed:
            return HMMDFAStepScores(torch.full_like(hmm_logits, -1e9), set(), {}, {})

        combined = hmm_logits * float(hmm_weight)
        next_beliefs: dict[int, torch.Tensor] = {}
        next_dfa_states: dict[int, Any] = {}

        if float(lookahead_weight):
            lookahead_logits = torch.full_like(hmm_logits, -1e9)
            lookahead_remaining = lookahead_remaining_after_label(remaining_steps, lookahead_max_steps)
            for label in allowed:
                next_state = self.dfa.step(product_state.dfa_state, int(label))
                if next_state is None:
                    continue
                next_belief = runtime.forward_update(product_state.hmm_belief, product_state.labels, int(label))
                next_beliefs[int(label)] = next_belief
                next_dfa_states[int(label)] = next_state
                success = self._lookahead_success_after_label(
                    runtime,
                    next_belief,
                    next_state,
                    int(label),
                    product_state.labels + [int(label)],
                    lookahead_remaining,
                    emittable,
                    policy=policy,
                    eos_label=eos_label,
                    static_lookahead=static_lookahead,
                    recursive_cache=recursive_cache,
                )
                lookahead_logits[int(label)] = torch.log(success.clamp_min(torch.finfo(success.dtype).eps))
            combined = combined + lookahead_logits * float(lookahead_weight)

        if float(hf_weight):
            hf_logits = self._backend_label_logits(
                self.generator,
                product_state.full_ids,
                self.scorer_head,
                self.vocabulary,
                product_state.hmm_belief.device,
                label_count=hmm_logits.numel(),
            )
            if hf_logits is not None:
                combined = combined + hf_logits.to(device=combined.device, dtype=combined.dtype) * float(hf_weight)

        try:
            masked = self._mask_label_logits(combined, allowed)
        except ValueError:
            return HMMDFAStepScores(torch.full_like(combined, -1e9), set(), next_beliefs, next_dfa_states)
        return HMMDFAStepScores(masked, allowed, next_beliefs, next_dfa_states)

    def _expand_state(
        self,
        product_state: HMMDFAProductState,
        label: int,
        log_prob: float,
        step_scores: HMMDFAStepScores,
        runtime: HMMRuntimeView,
    ) -> HMMDFAProductState:
        """Advance one product state by one compact label."""
        label = int(label)
        next_dfa_state = step_scores.next_dfa_states.get(label)
        if next_dfa_state is None:
            next_dfa_state = self.dfa.step(product_state.dfa_state, label)
        if next_dfa_state is None:
            raise RuntimeError(f"DFA has no transition from {product_state.dfa_state!r} on label {label!r}")
        next_belief = step_scores.next_beliefs.get(label)
        if next_belief is None:
            next_belief = runtime.forward_update(product_state.hmm_belief, product_state.labels, label)
        token_id = self._token_id_for_generated_label(self.scorer_head, self.vocabulary, label)
        next_step = len(product_state.labels) + 1
        return HMMDFAProductState(
            hmm_belief=next_belief,
            dfa_state=next_dfa_state,
            full_ids=product_state.full_ids + [token_id],
            labels=product_state.labels + [label],
            score=product_state.score + float(log_prob),
            scores=product_state.scores + [float(log_prob)],
            last_dfa_state_change_step=(
                next_step
                if next_dfa_state != product_state.dfa_state
                else product_state.last_dfa_state_change_step
            ),
        )

    def _progress(
        self,
        product_state: HMMDFAProductState,
        *,
        prompt_len: int,
        update_progress,
        eos_label: int,
    ) -> DecodeProgress:
        """Create a StopPolicy progress snapshot for one product state."""
        base = update_progress(
            step_index=len(product_state.labels),
            dfa_state=product_state.dfa_state,
            prompt_token_count=prompt_len,
            generated_token_ids=tuple(product_state.full_ids[prompt_len:]),
            generated_labels=tuple(product_state.labels),
            accepted=self.dfa.is_accepting(product_state.dfa_state),
            eos_emitted=bool(product_state.labels and int(product_state.labels[-1]) == int(eos_label)),
        )
        return DecodeProgress(
            step_index=base.step_index,
            elapsed_seconds=base.elapsed_seconds,
            dfa_state=base.dfa_state,
            prompt_token_count=base.prompt_token_count,
            generated_token_ids=base.generated_token_ids,
            generated_labels=base.generated_labels,
            accepted=base.accepted,
            eos_emitted=base.eos_emitted,
            last_dfa_state_change_step=product_state.last_dfa_state_change_step,
        )

    def _decode_greedy(
        self,
        prompt_ids: torch.Tensor,
        *,
        runtime: HMMRuntimeView,
        policy: StopPolicy,
        emittable: set[int],
        eos_label: int,
        hmm_weight: float,
        hf_weight: float,
        lookahead_weight: float,
        lookahead_max_steps: int | None,
        static_lookahead: HMMDFAStaticLookahead | None,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
        keep_rejected: bool,
    ) -> list[HMMDFAProductState]:
        """Deterministically decode one product path by argmax at every step."""
        product = self._init_product_state(prompt_ids, runtime)
        prompt_len = int(prompt_ids.shape[-1])
        update_progress = make_progress_tracker()
        while True:
            progress = self._progress(product, prompt_len=prompt_len, update_progress=update_progress, eos_label=eos_label)
            if should_stop_decoding(policy, progress):
                break
            step_scores = self._score_next_labels(
                product,
                runtime,
                policy=policy,
                step_index=len(product.labels),
                emittable=emittable,
                eos_label=eos_label,
                hmm_weight=hmm_weight,
                hf_weight=hf_weight,
                lookahead_weight=lookahead_weight,
                lookahead_max_steps=lookahead_max_steps,
                static_lookahead=static_lookahead,
                recursive_cache=recursive_cache,
            )
            if not step_scores.allowed:
                break
            label = int(torch.argmax(step_scores.masked_logits).item())
            log_probs = torch.log_softmax(step_scores.masked_logits, dim=-1)
            product = self._expand_state(
                product,
                label,
                float(log_probs[label].detach().item()),
                step_scores,
                runtime,
            )
            if should_stop_on_token(
                policy,
                eos_emitted=(label == int(eos_label)),
                accepted=self.dfa.is_accepting(product.dfa_state),
            ):
                break
        if self.dfa.is_accepting(product.dfa_state) or keep_rejected:
            return [product]
        return []

    def _decode_sample(
        self,
        prompt_ids: torch.Tensor,
        *,
        runtime: HMMRuntimeView,
        policy: StopPolicy,
        emittable: set[int],
        eos_label: int,
        hmm_weight: float,
        hf_weight: float,
        lookahead_weight: float,
        lookahead_max_steps: int | None,
        static_lookahead: HMMDFAStaticLookahead | None,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
        num_return_sequences: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator_seed: int,
        keep_rejected: bool,
        max_attempts: int,
    ) -> list[HMMDFAProductState]:
        """Stochastic HMM+DFA product decoding with repeated attempts."""
        accepted: list[HMMDFAProductState] = []
        rejected: list[HMMDFAProductState] = []
        prompt_len = int(prompt_ids.shape[-1])
        device = prompt_ids.device
        for attempt in range(max(1, int(max_attempts))):
            if len(accepted) >= int(num_return_sequences):
                break
            product = self._init_product_state(prompt_ids, runtime)
            rng = torch.Generator(device=device).manual_seed(int(generator_seed) + attempt)
            update_progress = make_progress_tracker()
            while True:
                progress = self._progress(product, prompt_len=prompt_len, update_progress=update_progress, eos_label=eos_label)
                if should_stop_decoding(policy, progress):
                    break
                step_scores = self._score_next_labels(
                    product,
                    runtime,
                    policy=policy,
                    step_index=len(product.labels),
                    emittable=emittable,
                    eos_label=eos_label,
                    hmm_weight=hmm_weight,
                    hf_weight=hf_weight,
                    lookahead_weight=lookahead_weight,
                    lookahead_max_steps=lookahead_max_steps,
                    static_lookahead=static_lookahead,
                    recursive_cache=recursive_cache,
                )
                if not step_scores.allowed:
                    break
                label, log_prob = self._select_label(
                    step_scores.masked_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=rng,
                )
                product = self._expand_state(
                    product,
                    label,
                    log_prob,
                    step_scores,
                    runtime,
                )
                if should_stop_on_token(
                    policy,
                    eos_emitted=(int(label) == int(eos_label)),
                    accepted=self.dfa.is_accepting(product.dfa_state),
                ):
                    break
            if self.dfa.is_accepting(product.dfa_state):
                accepted.append(product)
            else:
                rejected.append(product)
        if accepted:
            return accepted[: int(num_return_sequences)]
        if keep_rejected:
            return rejected[: int(num_return_sequences)]
        return []

    def _decode_beam(
        self,
        prompt_ids: torch.Tensor,
        *,
        runtime: HMMRuntimeView,
        policy: StopPolicy,
        emittable: set[int],
        eos_label: int,
        hmm_weight: float,
        hf_weight: float,
        lookahead_weight: float,
        lookahead_max_steps: int | None,
        static_lookahead: HMMDFAStaticLookahead | None,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
        num_return_sequences: int,
        beam_size: int,
        length_penalty: float,
        early_stopping: bool,
        keep_rejected: bool,
    ) -> list[HMMDFAProductState]:
        """Beam search over explicit HMM+DFA product states."""
        prompt_len = int(prompt_ids.shape[-1])
        active = [self._init_product_state(prompt_ids, runtime)]
        finished: list[HMMDFAProductState] = []
        stopped: list[HMMDFAProductState] = []

        while active:
            expanded: list[HMMDFAProductState] = []
            for product in active:
                update_progress = make_progress_tracker()
                progress = self._progress(product, prompt_len=prompt_len, update_progress=update_progress, eos_label=eos_label)
                if should_stop_decoding(policy, progress):
                    stopped.append(product)
                    continue
                step_scores = self._score_next_labels(
                    product,
                    runtime,
                    policy=policy,
                    step_index=len(product.labels),
                    emittable=emittable,
                    eos_label=eos_label,
                    hmm_weight=hmm_weight,
                    hf_weight=hf_weight,
                    lookahead_weight=lookahead_weight,
                    lookahead_max_steps=lookahead_max_steps,
                    static_lookahead=static_lookahead,
                    recursive_cache=recursive_cache,
                )
                if not step_scores.allowed:
                    stopped.append(product)
                    continue
                valid_labels = [label for label in step_scores.allowed if 0 <= int(label) < step_scores.masked_logits.numel()]
                if not valid_labels:
                    stopped.append(product)
                    continue
                log_probs = torch.log_softmax(step_scores.masked_logits, dim=-1)
                local = sorted(
                    ((int(label), float(log_probs[int(label)].detach().item())) for label in valid_labels),
                    key=lambda item: item[1],
                    reverse=True,
                )[: int(beam_size)]
                for label, log_prob in local:
                    child = self._expand_state(
                        product,
                        label,
                        log_prob,
                        step_scores,
                        runtime,
                    )
                    if should_stop_on_token(
                        policy,
                        eos_emitted=(label == int(eos_label)),
                        accepted=self.dfa.is_accepting(child.dfa_state),
                    ):
                        finished.append(child)
                    else:
                        expanded.append(child)
            if early_stopping and len([state for state in finished if self.dfa.is_accepting(state.dfa_state)]) >= int(num_return_sequences):
                break
            if not expanded:
                break
            expanded.sort(key=lambda state: self._rank(state, length_penalty), reverse=True)
            active = expanded[: int(beam_size)]

        pool = finished + stopped + active
        accepted = [state for state in pool if self.dfa.is_accepting(state.dfa_state)]
        ranked = accepted if accepted or not keep_rejected else pool
        ranked.sort(key=lambda state: self._rank(state, length_penalty), reverse=True)
        return ranked[: int(num_return_sequences)]

    def _decode_result_from_state(
        self,
        product_state: HMMDFAProductState,
        prompt_ids: torch.Tensor,
        *,
        search: str,
        metadata: Mapping[str, Any],
    ) -> HMMDFADecodeResult:
        """Convert a product state into the public direct decode result."""
        prompt_len = int(prompt_ids.shape[-1])
        generated_ids = list(product_state.full_ids[prompt_len:])
        text = self.tokenizer.decode(generated_ids) if self.tokenizer is not None else None
        return HMMDFADecodeResult(
            text=text,
            token_ids=generated_ids,
            labels=list(product_state.labels),
            final_state=product_state.dfa_state,
            accepted=self.dfa.is_accepting(product_state.dfa_state),
            score=float(product_state.score),
            scores=list(product_state.scores),
            final_hmm_belief=product_state.hmm_belief.detach().clone(),
            search=search,
            metadata=dict(metadata),
        )

    def _build_static_lookahead(
        self,
        runtime: HMMRuntimeView,
        *,
        policy: StopPolicy,
        emittable: set[int],
        eos_label: int,
        lookahead_weight: float,
        lookahead_max_steps: int | None,
    ) -> HMMDFAStaticLookahead | None:
        """Precompute static HMM+DFA lookahead values when matrices are fixed."""
        if not float(lookahead_weight):
            return None
        if runtime.static_transition is None or runtime.static_emission is None:
            return None
        max_depth = self._lookahead_table_depth(policy, lookahead_max_steps)
        if max_depth is None:
            return None

        device = runtime.initial_belief.device
        dtype = runtime.initial_belief.dtype
        transition = runtime.static_transition.to(device=device, dtype=dtype)
        emission = runtime.static_emission.to(device=device, dtype=dtype)
        if transition.dim() != 2 or emission.dim() != 2:
            return None
        if transition.shape[0] != transition.shape[1] or transition.shape[0] != emission.shape[0]:
            return None

        state_count = int(transition.shape[0])
        zero = torch.zeros((state_count,), dtype=dtype, device=device)
        values: list[dict[Any, torch.Tensor]] = [
            {state: zero.clone() for state in self.dfa.states}
        ]
        for depth in range(1, int(max_depth) + 1):
            previous = values[depth - 1]
            current: dict[Any, torch.Tensor] = {}
            for dfa_state in self.dfa.states:
                vector = zero.clone()
                allowed = {int(label) for label in self.dfa.allowed_tokens(dfa_state, remaining_steps=depth)}
                allowed &= emittable
                for label in allowed:
                    if label < 0 or label >= emission.shape[-1]:
                        continue
                    next_state = self.dfa.step(dfa_state, label)
                    if next_state is None:
                        continue
                    emit = emission[:, label]
                    if should_stop_on_token(
                        policy,
                        eos_emitted=(label == int(eos_label)),
                        accepted=self.dfa.is_accepting(next_state),
                    ):
                        vector = vector + emit
                    else:
                        next_vector = previous.get(next_state, zero)
                        continuation = torch.matmul(transition, next_vector)
                        vector = vector + emit * continuation
                current[dfa_state] = vector.clamp(0.0, 1.0)
            values.append(current)
        return HMMDFAStaticLookahead(values=values, max_depth=int(max_depth))

    @staticmethod
    def _lookahead_table_depth(policy: StopPolicy, lookahead_max_steps: int | None) -> int | None:
        """Return finite depth for static lookahead table construction."""
        if lookahead_max_steps is not None:
            return max(0, int(lookahead_max_steps))
        if policy.max_steps is not None:
            return max(0, int(policy.max_steps))
        return 8

    def _lookahead_success_after_label(
        self,
        runtime: HMMRuntimeView,
        belief: torch.Tensor,
        dfa_state,
        label: int,
        prefix_labels: Sequence[int],
        remaining_steps: int | None,
        emittable: set[int],
        *,
        policy: StopPolicy,
        eos_label: int,
        static_lookahead: HMMDFAStaticLookahead | None,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
    ) -> torch.Tensor:
        """Return lookahead success probability after committing one label."""
        if should_stop_on_token(
            policy,
            eos_emitted=(int(label) == int(eos_label)),
            accepted=self.dfa.is_accepting(dfa_state),
        ):
            return belief.new_tensor(1.0)
        if remaining_steps is not None and int(remaining_steps) <= 0:
            return belief.new_tensor(0.0)
        if static_lookahead is not None:
            return static_lookahead.success_after(belief, dfa_state, remaining_steps)
        return self._success_probability(
            runtime,
            belief,
            dfa_state,
            prefix_labels,
            remaining_steps,
            emittable,
            policy=policy,
            eos_label=eos_label,
            recursive_cache=recursive_cache,
        )

    def _success_probability(
        self,
        runtime: HMMRuntimeView,
        belief: torch.Tensor,
        dfa_state,
        prefix_labels: Sequence[int],
        remaining_steps: int | None,
        emittable: set[int],
        *,
        policy: StopPolicy,
        eos_label: int,
        recursive_cache: dict[tuple[tuple[int, ...], Any, int | None], torch.Tensor],
    ) -> torch.Tensor:
        """Estimate recursive success probability for HMM+DFA lookahead."""
        one = belief.new_tensor(1.0)
        zero = belief.new_tensor(0.0)
        if should_stop_on_token(
            policy,
            eos_emitted=bool(prefix_labels and int(prefix_labels[-1]) == int(eos_label)),
            accepted=self.dfa.is_accepting(dfa_state),
        ):
            return one
        if remaining_steps is not None and int(remaining_steps) <= 0:
            return zero

        cache_key = (tuple(int(label) for label in prefix_labels), dfa_state, None if remaining_steps is None else int(remaining_steps))
        try:
            cached = recursive_cache.get(cache_key)
        except TypeError:
            cached = None
            cache_key = None
        if cached is not None:
            return cached.to(device=belief.device, dtype=belief.dtype)

        emission = runtime.emission_for(prefix_labels).to(device=belief.device, dtype=belief.dtype)
        label_probs = torch.matmul(belief, emission)
        allowed = {int(label) for label in self.dfa.allowed_tokens(dfa_state, remaining_steps=remaining_steps)}
        allowed &= emittable
        if not allowed:
            return zero

        next_remaining = None if remaining_steps is None else max(0, int(remaining_steps) - 1)
        total = zero
        for label in allowed:
            next_state = self.dfa.step(dfa_state, int(label))
            if next_state is None:
                continue
            next_belief = runtime.forward_update(belief, prefix_labels, int(label))
            child = self._success_probability(
                runtime,
                next_belief,
                next_state,
                tuple(prefix_labels) + (int(label),),
                next_remaining,
                emittable,
                policy=policy,
                eos_label=eos_label,
                recursive_cache=recursive_cache,
            )
            total = total + label_probs[int(label)] * child
        result = total.clamp(0.0, 1.0)
        if cache_key is not None:
            recursive_cache[cache_key] = result.detach()
        return result

    @staticmethod
    def _prompt_tensor(prompt_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Normalize prompt ids into a 2D long tensor."""
        if isinstance(prompt_ids, torch.Tensor):
            return prompt_ids.long().unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.long()
        return torch.tensor([list(int(item) for item in prompt_ids)], dtype=torch.long)

    @staticmethod
    def _rank(product_state: HMMDFAProductState, length_penalty: float) -> float:
        """Rank beams with a length-normalized accumulated score."""
        return product_state.score / (max(1, len(product_state.labels)) ** float(length_penalty))


__all__ = ["HMMDFADecodeResult", "HMMDFADecoder"]
