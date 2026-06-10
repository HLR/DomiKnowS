"""Hybrid generation controller/scorer utilities.

This module keeps the open-vocabulary generator and the compact DomiKnowS
generation head in separate roles:

* a large backend proposes text/token candidates;
* the DFA verifies hard graph constraints;
* the compact head scores domain/style/latent preferences and risk.

For HuggingFace-backed generation, this module also supports two strict
product-state decode paths:

* ``product_compact_learner_dfa`` combines compact-head next-label logits with
    DFA-allowed transitions at each step.
* ``product_hmm_dfa`` tracks an explicit HMM belief state jointly with DFA
    state, with optional lookahead and optional backend-token logit biasing.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import re

from .adapters import GenerationResult, HuggingFaceGenerationAdapter, OpenAIResponsesAdapter
from .hmm_dfa_decoder import (
    HMMDFADecodeResult,
    HMMDFADecoder,
)
from .hmm_utils import (
    has_hmm_matrices,
    static_hmm_teacher_forced_log_probs,
)
from ..dfa.core import DFA
from ..dfa.visualization import explain_dfa_rejection
from ..dfa.decoder import ConstrainedGenerationResult
from ..dfa.stop_policy import (
    StopPolicy,
    make_progress_tracker,
    remaining_steps_for,
    should_stop_decoding,
    should_stop_on_token,
    stop_policy_from_legacy,
)
from ..latent import GenerationEnforcement, LatentLossBreakdown, token_probs_from_log_probs
from ..dfa.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class GenerationCandidate:
    """One generated candidate, usually containing only the generated portion."""

    text: str | None = None
    token_ids: list[int] | None = None
    labels: list[int] | None = None
    raw: Any = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    """Score and diagnostics for one hybrid candidate."""

    total: float
    head_logprob: float
    validity: float
    latent_preference: float
    risk: float
    length: int
    accepted: bool
    rejection: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredCandidate:
    """Candidate paired with its hybrid score."""

    candidate: GenerationCandidate
    score: CandidateScore


@dataclass(frozen=True)
class HybridScoreWeights:
    """Weights used to combine candidate scoring terms."""

    head_logprob: float = 1.0
    validity: float = 10.0
    latent_preference: float = 1.0
    risk: float = 1.0
    length: float = 0.0


@dataclass(frozen=True)
class ConstraintBundle:
    """Named precompiled DFA used by prompt-level constraint selection."""

    name: str
    dfa: DFA
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dfa(self, vocabulary: TokenVocabulary) -> DFA:
        """Return the precompiled DFA; *vocabulary* is accepted for API symmetry."""
        return self.dfa


class ManualConstraintSelector:
    """Simple keyword/manual selector for named constraint bundles."""

    def __init__(self, rules: Mapping[str, str] | None = None, default: str | None = None):
        """Store keyword-to-bundle rules and an optional default bundle name."""
        self.rules = {str(key).lower(): str(value) for key, value in (rules or {}).items()}
        self.default = default

    def select(self, prompt: str, bundles: Sequence[ConstraintBundle]) -> ConstraintBundle:
        """Return the first bundle whose configured keyword appears in *prompt*."""
        if not bundles:
            raise ValueError("constraint_bundles must not be empty")
        by_name = {bundle.name: bundle for bundle in bundles}
        prompt_l = prompt.lower()
        for keyword, bundle_name in self.rules.items():
            if keyword in prompt_l and bundle_name in by_name:
                return by_name[bundle_name]
        if self.default and self.default in by_name:
            return by_name[self.default]
        return bundles[0]


class CompactConstraintSelector(torch.nn.Module):
    """Small trainable prompt classifier for constraint-bundle selection."""

    def __init__(self, vocab_size: int, bundle_names: Sequence[str], hidden_size: int = 16):
        """Initialize embedding and classifier layers for bundle prediction."""
        super().__init__()
        if not bundle_names:
            raise ValueError("bundle_names must not be empty")
        self.bundle_names = tuple(str(name) for name in bundle_names)
        self.embedding = torch.nn.Embedding(int(vocab_size), int(hidden_size))
        self.classifier = torch.nn.Linear(int(hidden_size), len(self.bundle_names))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return bundle logits from mean pooled token embeddings."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        features = self.embedding(input_ids.long()).mean(dim=1)
        return self.classifier(features)

    def select(self, input_ids: torch.Tensor, bundles: Sequence[ConstraintBundle]) -> ConstraintBundle:
        """Select the highest-scoring bundle for encoded prompt IDs."""
        names = [bundle.name for bundle in bundles]
        logits = self.forward(input_ids)[0]
        for label in torch.argsort(logits, descending=True).tolist():
            name = self.bundle_names[int(label)]
            if name in names:
                return bundles[names.index(name)]
        return bundles[0]


def preference_pair_ranking_loss(
    chosen_score: torch.Tensor | float,
    rejected_score: torch.Tensor | float,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise ranking loss for domain-style/preference training."""
    chosen = chosen_score if isinstance(chosen_score, torch.Tensor) else torch.tensor(float(chosen_score))
    rejected = rejected_score if isinstance(rejected_score, torch.Tensor) else chosen.new_tensor(float(rejected_score))
    return torch.relu(rejected - chosen + float(margin))


class HybridController:
    """Controller that verifies large-model candidates and scores them with a compact head."""

    def __init__(
        self,
        generator=None,
        vocabulary: TokenVocabulary | None = None,
        dfa: DFA | None = None,
        scorer_head=None,
        *,
        enforcement: GenerationEnforcement | None = None,
        tokenizer=None,
        weights: HybridScoreWeights | Mapping[str, float] | None = None,
        constraints: Sequence[Any] | None = None,
        constraint_selector: Any = None,
    ):
        """Initialize generator, DFA, compact scorer, and runtime decode settings."""
        if vocabulary is None:
            raise ValueError("vocabulary is required")
        if dfa is None:
            raise ValueError("dfa is required")
        self.generator = generator
        self.vocabulary = vocabulary
        self.dfa = dfa
        self.scorer_head = scorer_head
        self.enforcement = enforcement
        self.tokenizer = tokenizer or getattr(generator, "tokenizer", None) or getattr(vocabulary, "tokenizer", None)
        self.weights = _coerce_weights(weights)
        self.constraints = tuple(constraints or ())
        self.constraint_selector = constraint_selector

    def generate_verify_rerank(
        self,
        prompt: str | torch.Tensor | Sequence[int],
        num_candidates: int,
        *,
        backend: str = "auto",
        keep_rejected: bool = False,
        explain: bool = False,
        candidates: Sequence[GenerationCandidate | GenerationResult | ConstrainedGenerationResult | str] | None = None,
        max_new_tokens: int | None = 16,
        stop_policy: StopPolicy | None = None,
        hard_decode: bool = True,
        decode_strategy: str | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        generator_seed: int = 0,
        **generate_kwargs,
    ) -> list[ScoredCandidate]:
        """Generate or consume candidates, verify them, and return ranked results."""
        prompt_ids = self._prompt_ids(prompt)
        if candidates is None:
            candidates = self._generate_candidates(
                prompt,
                prompt_ids,
                num_candidates,
                backend=backend,
                max_new_tokens=max_new_tokens,
                stop_policy=stop_policy,
                hard_decode=hard_decode,
                decode_strategy=decode_strategy,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
                explain=explain,
                keep_rejected=keep_rejected,
                **generate_kwargs,
            )
        ranked = self.rerank_candidates(
            prompt_ids,
            candidates,
            keep_rejected=keep_rejected,
            explain=explain,
        )
        return ranked[: int(num_candidates)]

    def rerank_candidates(
        self,
        prompt_ids: torch.Tensor | Sequence[int],
        candidates: Sequence[GenerationCandidate | GenerationResult | ConstrainedGenerationResult | str],
        *,
        keep_rejected: bool = False,
        explain: bool = False,
    ) -> list[ScoredCandidate]:
        """Verify and score precomputed candidates."""
        prompt_ids_t = self._prompt_ids(prompt_ids)
        with_scores = [
            ScoredCandidate(candidate_n, self.score_candidate(prompt_ids_t, candidate_n, explain=explain))
            for candidate_n in (self._normalise_candidate(candidate, prompt_ids_t) for candidate in candidates)
        ]
        if not keep_rejected:
            accepted = [item for item in with_scores if item.score.accepted]
            if accepted:
                with_scores = accepted
        return sorted(with_scores, key=lambda item: item.score.total, reverse=True)

    def score_candidate(
        self,
        prompt_ids: torch.Tensor | Sequence[int],
        candidate: GenerationCandidate | GenerationResult | ConstrainedGenerationResult | str,
        *,
        explain: bool = False,
    ) -> CandidateScore:
        """Score one candidate using DFA validity, compact-head score, latent loss, and risk."""
        prompt_ids_t = self._prompt_ids(prompt_ids)
        candidate_n = self._normalise_candidate(candidate, prompt_ids_t)
        labels = list(candidate_n.labels or ())
        accepted = self.dfa.accepts(labels)
        rejection = explain_dfa_rejection(self.dfa, labels) if explain and not accepted else None
        head_logprob, log_probs = self._head_logprob(prompt_ids_t, candidate_n)
        risk = self.predict_failure_risk(prompt_ids_t, labels)
        latent_preference, latent_diag = self._latent_preference(log_probs)
        validity = 1.0 if accepted else 0.0
        length = len(labels)
        total = (
            self.weights.validity * validity
            + self.weights.head_logprob * head_logprob
            + self.weights.latent_preference * latent_preference
            - self.weights.risk * risk
            - self.weights.length * length
        )
        diagnostics = {
            "labels": labels,
            "tokens": [self.vocabulary.token_for_label(label) for label in labels],
            "latent": latent_diag,
        }
        return CandidateScore(
            total=float(total),
            head_logprob=float(head_logprob),
            validity=float(validity),
            latent_preference=float(latent_preference),
            risk=float(risk),
            length=length,
            accepted=accepted,
            rejection=rejection,
            diagnostics=diagnostics,
        )

    def predict_failure_risk(
        self,
        prompt_ids: torch.Tensor | Sequence[int],
        prefix_labels: Sequence[int],
        *,
        remaining_steps: int | None = None,
    ) -> float:
        """Estimate next-step risk as compact-head probability mass outside DFA-allowed labels."""
        if self.scorer_head is None:
            return 0.0
        prompt_ids_t = self._prompt_ids(prompt_ids)
        state = self.dfa.start_state
        prefix_token_ids = _flat_ids(prompt_ids_t)
        for label in prefix_labels:
            next_state = self.dfa.step(state, int(label))
            if next_state is None:
                return 1.0
            state = next_state
            try:
                prefix_token_ids.append(self.vocabulary.token_id_for_label(int(label)))
            except ValueError:
                pass
        if remaining_steps is None:
            remaining_steps = max(1, len(prefix_labels) + 1)
        allowed = {int(label) for label in self.dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
        try:
            logits = _next_label_logits(self.scorer_head, prefix_token_ids, prompt_ids_t)
        except (TypeError, ValueError):
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        allowed_mass = probs.new_zeros(())
        for label in allowed:
            if 0 <= label < probs.numel():
                allowed_mass = allowed_mass + probs[label]
        return float((1.0 - allowed_mass.clamp(0.0, 1.0)).detach().item())

    def suggest_repair(
        self,
        candidate: GenerationCandidate | GenerationResult | ConstrainedGenerationResult | str,
        *,
        prompt_ids: torch.Tensor | Sequence[int] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return DFA/constraint repair diagnostics and likely next compact labels."""
        prompt_ids_t = self._prompt_ids(prompt_ids if prompt_ids is not None else [])
        candidate_n = self._normalise_candidate(candidate, prompt_ids_t)
        labels = list(candidate_n.labels or ())
        tokens = [self.vocabulary.token_for_label(label) for label in labels]
        suggestions = _constraint_repair_suggestions(self.dfa, labels, self.vocabulary)
        suggestions.extend(_constraint_name_repair_suggestions(self.constraints, tokens))
        suggestions = list(dict.fromkeys(suggestions))
        state = self.dfa.start_state
        for label in labels:
            next_state = self.dfa.step(state, int(label))
            if next_state is None:
                break
            state = next_state
        allowed = {int(label) for label in self.dfa.allowed_tokens(state, remaining_steps=max(1, top_k))}
        next_labels = []
        logits = None
        if self.scorer_head is not None:
            try:
                logits = _next_label_logits(self.scorer_head, _candidate_full_ids(prompt_ids_t, candidate_n), prompt_ids_t)
            except ValueError:
                logits = None
        if logits is not None:
            masked = logits.clone()
            for label in range(masked.numel()):
                if label not in allowed:
                    masked[label] = -1e9
            for label in torch.argsort(masked, descending=True).tolist()[:top_k]:
                if masked[int(label)] > -5e8:
                    next_labels.append(
                        {
                            "label": int(label),
                            "token": self.vocabulary.token_for_label(int(label)),
                            "score": float(masked[int(label)].detach().item()),
                        }
                    )
        return {
            "accepted": self.dfa.accepts(labels),
            "rejection": None if self.dfa.accepts(labels) else explain_dfa_rejection(self.dfa, labels),
            "tokens": tokens,
            "suggestions": suggestions,
            "next_labels": next_labels,
        }

    def select_constraints(
        self,
        prompt: str | torch.Tensor | Sequence[int],
        constraint_bundles: Sequence[ConstraintBundle],
    ) -> ConstraintBundle:
        """Select which named constraint bundle should apply to a prompt."""
        if not constraint_bundles:
            raise ValueError("constraint_bundles must not be empty")
        selector = self.constraint_selector
        if selector is None:
            return constraint_bundles[0]
        if hasattr(selector, "select"):
            if isinstance(selector, CompactConstraintSelector):
                return selector.select(self._prompt_ids(prompt), constraint_bundles)
            return selector.select(prompt if isinstance(prompt, str) else "", constraint_bundles)
        selected = selector(prompt, constraint_bundles)
        if isinstance(selected, ConstraintBundle):
            return selected
        for bundle in constraint_bundles:
            if bundle.name == selected:
                return bundle
        raise ValueError(f"selector returned unknown constraint bundle {selected!r}")

    def decode_hmm_dfa(
        self,
        prompt,
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
        decoder = HMMDFADecoder(
            dfa=self.dfa,
            vocabulary=self.vocabulary,
            generator=self.generator,
            scorer_head=self.scorer_head,
            tokenizer=self.tokenizer,
            backend_label_logits=_backend_label_logits,
            emittable_labels=_emittable_labels,
            flat_ids=_flat_ids,
            mask_label_logits=_mask_label_logits,
            select_label=_select_label,
            token_id_for_generated_label=_token_id_for_generated_label,
        )
        return decoder.decode_hmm_dfa(
            self._prompt_ids(prompt),
            search=search,
            num_return_sequences=num_return_sequences,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            stop_policy=stop_policy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator_seed=generator_seed,
            keep_rejected=keep_rejected,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **kwargs,
        )

    def _generate_candidates(
        self,
        prompt,
        prompt_ids: torch.Tensor,
        num_candidates: int,
        *,
        backend: str,
        max_new_tokens: int | None,
        stop_policy: StopPolicy | None,
        hard_decode: bool,
        decode_strategy: str | None,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        explain: bool,
        keep_rejected: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        """Dispatch candidate generation to backend-specific implementations."""
        backend = _resolve_backend(backend, self.generator)
        if backend == "hf":
            return self._generate_hf_candidates(
                prompt_ids,
                num_candidates,
                max_new_tokens=max_new_tokens,
                stop_policy=stop_policy,
                hard_decode=hard_decode,
                decode_strategy=decode_strategy,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
                keep_rejected=keep_rejected,
                **generate_kwargs,
            )
        if backend == "openai":
            if not isinstance(prompt, str):
                raise ValueError("OpenAI-compatible generation requires a text prompt")
            return self._generate_openai_candidates(
                prompt,
                num_candidates,
                max_output_tokens=max_new_tokens,
                explain=explain,
                **generate_kwargs,
            )
        raise ValueError(f"unsupported backend {backend!r}; pass candidates=... for precomputed reranking")

    def _generate_hf_candidates(
        self,
        prompt_ids: torch.Tensor,
        num_candidates: int,
        *,
        max_new_tokens: int | None,
        stop_policy: StopPolicy | None,
        hard_decode: bool,
        decode_strategy: str | None,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        keep_rejected: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        """Generate candidates with HF backend under the selected decode strategy."""
        if not isinstance(self.generator, HuggingFaceGenerationAdapter):
            raise ValueError("HF candidate generation requires a HuggingFaceGenerationAdapter")
        strategy = _resolve_decode_strategy(decode_strategy, hard_decode)
        # Product decode paths are routed early because they do not call
        # transformers.generate(...); they run custom token/label loops that
        # jointly enforce DFA constraints with compact-model structure.
        if strategy == "product_hmm_dfa":
            return self._generate_product_hmm_dfa_candidates(
                prompt_ids,
                num_candidates,
                max_new_tokens=max_new_tokens,
                stop_policy=stop_policy,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
                keep_rejected=keep_rejected,
                **generate_kwargs,
            )
        if strategy == "product_compact_learner_dfa":
            return self._generate_product_compact_learner_dfa_candidates(
                prompt_ids,
                num_candidates,
                max_new_tokens=max_new_tokens,
                stop_policy=stop_policy,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
                keep_rejected=keep_rejected,
                **generate_kwargs,
            )
        results = []
        prompt_len = int(prompt_ids.shape[-1])
        for index in range(int(num_candidates)):
            if strategy == "hard_dfa":
                result = self.generator.constrained_sample(
                    prompt_ids,
                    self.dfa,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    generator=torch.Generator(device=prompt_ids.device).manual_seed(generator_seed + index),
                    stop_policy=stop_policy,
                    **_filter_kwargs(generate_kwargs, {"use_cache"}),
                )
                generated_ids = result.token_ids[prompt_len:]
                text = self.tokenizer.decode(generated_ids) if self.tokenizer is not None else None
                results.append(
                    GenerationCandidate(
                        text=text,
                        token_ids=generated_ids,
                        labels=list(result.labels),
                        raw=result,
                        source=f"hf_sample_{index}",
                    )
                )
            elif strategy == "unconstrained":
                if not hasattr(self.generator.model, "generate"):
                    raise ValueError("unconstrained HF candidate generation requires model.generate(...)")
                output = self.generator.model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    **generate_kwargs,
                )
                ids = [int(item) for item in output[0].tolist()]
                generated_ids = ids[prompt_len:]
                results.append(
                    GenerationCandidate(
                        text=self.tokenizer.decode(generated_ids) if self.tokenizer is not None else None,
                        token_ids=generated_ids,
                        source=f"hf_generate_{index}",
                    )
                )
            else:
                raise ValueError(f"unsupported HF decode_strategy {decode_strategy!r}")
        return results

    def _generate_product_compact_learner_dfa_candidates(
        self,
        prompt_ids: torch.Tensor,
        num_candidates: int,
        *,
        max_new_tokens: int | None,
        stop_policy: StopPolicy | None,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        keep_rejected: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        """Decode over a compact-head/DFA product process.

        At each step this path:
        1) gets compact next-label logits for the current token prefix,
        2) masks labels that violate DFA transitions,
        3) optionally adds one-token backend label bias from the HF model,
        4) samples a label and appends its mapped token id.

        This is a strict hard-constraint path: no label outside the current
        DFA-allowed set can be sampled.
        """
        if self.scorer_head is None:
            raise ValueError("product_compact_learner_dfa decoding requires scorer_head")
        compact_weight = float(generate_kwargs.pop("compact_logit_weight", 1.0))
        backend_weight = float(generate_kwargs.pop("backend_logit_weight", 0.0))
        top_k = generate_kwargs.pop("top_k", None)
        max_attempts = int(generate_kwargs.pop("product_decode_max_attempts", max(1, int(num_candidates) * 4)))
        if generate_kwargs:
            # Keep this mode explicit: unconsumed kwargs are often misspellings
            # because this loop does not call transformers.generate(...).
            unknown = ", ".join(sorted(str(key) for key in generate_kwargs))
            raise ValueError(f"unsupported product_compact_learner_dfa kwargs: {unknown}")

        policy = stop_policy_from_legacy(max_new_tokens=max_new_tokens, stop_policy=stop_policy)
        prompt_len = int(prompt_ids.shape[-1])
        eos_label = int(self.vocabulary.eos_label)
        emittable = _emittable_labels(self.scorer_head, self.vocabulary)
        results: list[GenerationCandidate] = []
        rejected: list[GenerationCandidate] = []
        for index in range(max_attempts):
            if len(results) >= int(num_candidates):
                break
            # Per-attempt state starts from the prompt and DFA start state.
            full_ids = _flat_ids(prompt_ids)
            labels: list[int] = []
            scores: list[float] = []
            state = self.dfa.start_state
            rng = torch.Generator(device=prompt_ids.device).manual_seed(int(generator_seed) + index)
            update_progress = make_progress_tracker()

            step_index = 0
            while True:
                progress = update_progress(
                    step_index=step_index,
                    dfa_state=state,
                    prompt_token_count=prompt_len,
                    generated_token_ids=tuple(full_ids[prompt_len:]),
                    generated_labels=tuple(labels),
                    accepted=self.dfa.is_accepting(state),
                    eos_emitted=False,
                )
                if should_stop_decoding(policy, progress):
                    break
                remaining_steps = remaining_steps_for(policy, step_index)
                allowed = {int(label) for label in self.dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
                allowed &= emittable
                if not allowed:
                    break

                # Build step logits from compact head and optionally blend in
                # one-token HF backend evidence mapped into label space.
                compact_logits = _next_label_logits(self.scorer_head, full_ids, prompt_ids)
                combined = compact_logits * compact_weight
                if backend_weight:
                    backend_logits = _backend_label_logits(
                        self.generator,
                        full_ids,
                        self.scorer_head,
                        self.vocabulary,
                        prompt_ids.device,
                        label_count=compact_logits.numel(),
                    )
                    if backend_logits is not None:
                        combined = combined + backend_logits.to(device=combined.device, dtype=combined.dtype) * backend_weight
                masked = _mask_label_logits(combined, allowed)
                selected_label, log_prob = _select_label(
                    masked,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=rng,
                )
                next_state = self.dfa.step(state, int(selected_label))
                if next_state is None:
                    raise RuntimeError(f"DFA has no transition from {state!r} on label {selected_label!r}")
                token_id = _token_id_for_generated_label(self.scorer_head, self.vocabulary, selected_label)

                # Apply the sampled transition to both token and label traces.
                full_ids.append(token_id)
                labels.append(selected_label)
                scores.append(log_prob)
                state = next_state
                step_index += 1

                if should_stop_on_token(
                    policy,
                    eos_emitted=(selected_label == eos_label),
                    accepted=self.dfa.is_accepting(state),
                ):
                    break

            generated_ids = full_ids[prompt_len:]
            text = self.tokenizer.decode(generated_ids) if self.tokenizer is not None else None
            accepted = self.dfa.is_accepting(state)
            candidate = GenerationCandidate(
                text=text,
                token_ids=generated_ids,
                labels=labels,
                raw=ConstrainedGenerationResult(
                    token_ids=full_ids,
                    labels=labels,
                    final_state=state,
                    accepted=accepted,
                    score=sum(scores),
                    scores=scores,
                ),
                source=f"hf_product_compact_learner_dfa_{index}",
                metadata={
                    "decode_strategy": "product_compact_learner_dfa",
                    "backend_logit_integration": "one_token_label_bias" if backend_weight else "none",
                },
            )
            if accepted:
                results.append(candidate)
            else:
                rejected.append(candidate)
        if keep_rejected:
            return (results + rejected)[: int(num_candidates)]
        return results

    def _generate_product_hmm_dfa_candidates(
        self,
        prompt_ids: torch.Tensor,
        num_candidates: int,
        *,
        max_new_tokens: int | None,
        stop_policy: StopPolicy | None,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        keep_rejected: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        """Compatibility wrapper for ``generate_verify_rerank(..., product_hmm_dfa)``."""
        search = generate_kwargs.pop("product_hmm_search", "beam")
        beam_size = int(generate_kwargs.pop("beam_size", 4))
        length_penalty = float(generate_kwargs.pop("length_penalty", 1.0))
        early_stopping = bool(generate_kwargs.pop("early_stopping", True))
        top_k = generate_kwargs.pop("top_k", None)
        results = self.decode_hmm_dfa(
            prompt_ids,
            search=search,
            num_return_sequences=num_candidates,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            stop_policy=stop_policy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator_seed=generator_seed,
            keep_rejected=keep_rejected,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **generate_kwargs,
        )
        prompt_full_ids = _flat_ids(prompt_ids)
        candidates: list[GenerationCandidate] = []
        for index, result in enumerate(results):
            full_ids = prompt_full_ids + list(result.token_ids)
            metadata = dict(result.metadata)
            metadata["final_hmm_belief"] = result.final_hmm_belief.detach().cpu()
            candidates.append(
                GenerationCandidate(
                    text=result.text,
                    token_ids=list(result.token_ids),
                    labels=list(result.labels),
                    raw=ConstrainedGenerationResult(
                        token_ids=full_ids,
                        labels=list(result.labels),
                        final_state=result.final_state,
                        accepted=result.accepted,
                        score=result.score,
                        scores=list(result.scores),
                    ),
                    source=f"hf_product_hmm_dfa_{index}",
                    metadata=metadata,
                )
            )
        return candidates

    def _generate_openai_candidates(
        self,
        prompt: str,
        num_candidates: int,
        *,
        max_output_tokens: int,
        explain: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        """Generate candidates through the OpenAI adapter and normalize outputs."""
        if not isinstance(self.generator, OpenAIResponsesAdapter):
            raise ValueError("OpenAI candidate generation requires an OpenAIResponsesAdapter")
        results = []
        for index in range(int(num_candidates)):
            result = self.generator.generate_and_verify(
                prompt,
                self.vocabulary,
                self.dfa,
                max_output_tokens=max_output_tokens,
                explain=explain,
                **generate_kwargs,
            )
            results.append(self._normalise_candidate(result, self._prompt_ids(prompt), source=f"openai_{index}"))
        return results

    def _prompt_ids(self, prompt: str | torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        """Convert prompt text/ids/tensor into a 2D long tensor of token ids."""
        if prompt is None:
            return torch.empty((1, 0), dtype=torch.long)
        if isinstance(prompt, torch.Tensor):
            return prompt.long().unsqueeze(0) if prompt.dim() == 1 else prompt.long()
        if isinstance(prompt, str):
            if self.tokenizer is None:
                return torch.empty((1, 0), dtype=torch.long)
            if callable(self.tokenizer):
                encoded = self.tokenizer(prompt, return_tensors="pt")
                return encoded.input_ids.long()
            if hasattr(self.tokenizer, "encode"):
                return torch.tensor([list(int(item) for item in self.tokenizer.encode(prompt))], dtype=torch.long)
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor([list(int(item) for item in prompt)], dtype=torch.long)

    def _normalise_candidate(
        self,
        candidate: GenerationCandidate | GenerationResult | ConstrainedGenerationResult | str,
        prompt_ids: torch.Tensor,
        *,
        source: str | None = None,
    ) -> GenerationCandidate:
        """Convert supported candidate representations to GenerationCandidate."""
        if isinstance(candidate, GenerationCandidate):
            token_ids = candidate.token_ids
            labels = candidate.labels
            text = candidate.text
            if labels is None and token_ids is not None:
                labels = self.vocabulary.labels_for_token_ids(token_ids)
            if token_ids is None and text is not None:
                token_ids = self._encode_text(text)
            if labels is None and token_ids is not None:
                labels = self.vocabulary.labels_for_token_ids(token_ids)
            return GenerationCandidate(text=text, token_ids=token_ids, labels=labels, raw=candidate.raw, source=candidate.source or source, metadata=candidate.metadata)
        if isinstance(candidate, ConstrainedGenerationResult):
            prompt_len = int(prompt_ids.shape[-1])
            generated_ids = list(candidate.token_ids[prompt_len:])
            text = self.tokenizer.decode(generated_ids) if self.tokenizer is not None else None
            return GenerationCandidate(text=text, token_ids=generated_ids, labels=list(candidate.labels), raw=candidate, source=source or "constrained")
        if isinstance(candidate, GenerationResult):
            token_ids = list(candidate.token_ids) if candidate.token_ids is not None else self._encode_text(candidate.text)
            labels = list(candidate.labels) if candidate.labels is not None else self.vocabulary.labels_for_token_ids(token_ids)
            return GenerationCandidate(text=candidate.text, token_ids=token_ids, labels=labels, raw=candidate.raw, source=source or "generation_result")
        if isinstance(candidate, str):
            token_ids = self._encode_text(candidate)
            labels = self.vocabulary.labels_for_token_ids(token_ids)
            return GenerationCandidate(text=candidate, token_ids=token_ids, labels=labels, source=source or "text")
        raise TypeError(f"unsupported candidate type {type(candidate)!r}")

    def _encode_text(self, text: str) -> list[int]:
        """Encode text into token ids using the active tokenizer."""
        tokenizer = self.tokenizer or getattr(self.vocabulary, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer is required to encode text candidates")
        return list(int(item) for item in tokenizer.encode(text))

    def _head_logprob(self, prompt_ids: torch.Tensor, candidate: GenerationCandidate) -> tuple[float, torch.Tensor | None]:
        """Compute average teacher-forced log probability of candidate labels."""
        labels = list(candidate.labels or ())
        if self.scorer_head is None or not labels:
            return 0.0, None
        label_tensor = torch.tensor(labels, dtype=torch.long)
        log_probs = _teacher_forced_log_probs(self.scorer_head, prompt_ids, label_tensor, candidate.token_ids)
        if log_probs.dim() == 3:
            log_probs = log_probs[0]
        steps = min(len(labels), int(log_probs.shape[0]))
        score = log_probs.new_zeros(())
        for step in range(steps):
            label = labels[step]
            if 0 <= label < log_probs.shape[-1]:
                score = score + log_probs[step, label]
        normalised = score / max(1, steps)
        return float(normalised.detach().item()), log_probs[:steps].detach()

    def _latent_preference(self, log_probs: torch.Tensor | None) -> tuple[float, dict[str, Any]]:
        """Return latent-preference bonus and diagnostics from enforcement loss."""
        if log_probs is None or self.enforcement is None:
            return 0.0, {}
        if not getattr(self.enforcement, "latent_specs", ()):
            return 0.0, {}
        probs = token_probs_from_log_probs(log_probs)
        breakdown: LatentLossBreakdown = self.enforcement.latent_breakdown(
            probs,
            eos_label=self.vocabulary.eos_label,
        )
        latent_loss = float(breakdown.total.detach().item())
        return -latent_loss, {
            "loss": latent_loss,
            "terms": [item.name for item in breakdown.items],
        }


def _teacher_forced_log_probs(
    model,
    prompt_ids: torch.Tensor,
    labels: torch.Tensor,
    candidate_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return per-step log probabilities for labels under teacher forcing."""
    if has_hmm_matrices(model) and not hasattr(model, "sequence_log_probs"):
        return static_hmm_teacher_forced_log_probs(model, labels)
    if hasattr(model, "sequence_log_probs"):
        try:
            return model.sequence_log_probs(labels)
        except (TypeError, ValueError):
            pass
    if hasattr(model, "next_label_logits"):
        return _stepwise_log_probs(model, prompt_ids, labels, candidate_token_ids)
    try:
        return model(None, prompt_ids, labels)
    except (TypeError, ValueError, NotImplementedError):
        pass
    return _stepwise_log_probs(model, prompt_ids, labels, candidate_token_ids)


def _stepwise_log_probs(
    model,
    prompt_ids: torch.Tensor,
    labels: torch.Tensor,
    candidate_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Compute per-step log probs by repeatedly querying next-label logits."""
    ids = _flat_ids(prompt_ids)
    rows = []
    candidate_token_ids = list(candidate_token_ids or ())
    for step, label in enumerate(labels.tolist()):
        logits = _next_label_logits(model, ids, prompt_ids)
        rows.append(torch.log_softmax(logits, dim=-1))
        token_id = candidate_token_ids[step] if step < len(candidate_token_ids) else None
        if token_id is None:
            try:
                token_id = model.token_id_for_label(int(label))
            except Exception:
                token_id = None
        if token_id is not None:
            ids.append(int(token_id))
    if not rows:
        label_count = int(getattr(model, "label_count", 1))
        return torch.empty((0, label_count), dtype=torch.float32)
    return torch.stack(rows, dim=0)


def _next_label_logits(model, input_ids: Sequence[int], prompt_ids: torch.Tensor) -> torch.Tensor:
    """Resolve one-step label logits from compact head or model forward pass."""
    ids = torch.tensor([list(int(item) for item in input_ids)], dtype=torch.long, device=prompt_ids.device)
    if hasattr(model, "next_label_logits"):
        logits = model.next_label_logits(ids)
        if logits.dim() != 1:
            raise ValueError(f"expected compact label logits to be 1D, got shape {tuple(logits.shape)}")
        return logits
    try:
        output = model(ids)
    except NotImplementedError as exc:
        raise ValueError("scorer head must expose next_label_logits for risk/repair scoring") from exc
    logits = output.logits if hasattr(output, "logits") else output
    return logits[0, -1, :]


def _resolve_decode_strategy(decode_strategy: str | None, hard_decode: bool) -> str:
    """Normalize public decode-strategy names to internal routing keys.

    This keeps legacy aliases stable while exposing the new product paths
    under explicit names.
    """
    if decode_strategy is None:
        return "hard_dfa" if hard_decode else "unconstrained"
    strategy = str(decode_strategy).strip().lower().replace("-", "_")
    aliases = {
        "hard": "hard_dfa",
        "hard_decode": "hard_dfa",
        "hard_dfa": "hard_dfa",
        "dfa": "hard_dfa",
        "constrained": "hard_dfa",
        "generate": "unconstrained",
        "unconstrained": "unconstrained",
        "unconstrained_generate": "unconstrained",
        "product_compact_learner_dfa": "product_compact_learner_dfa",
        "compact_learner_dfa": "product_compact_learner_dfa",
        "compact_dfa": "product_compact_learner_dfa",
        "product_hmm_dfa": "product_hmm_dfa",
        "hmm_dfa": "product_hmm_dfa",
        "strict_hmm_dfa": "product_hmm_dfa",
    }
    try:
        return aliases[strategy]
    except KeyError as exc:
        raise ValueError(
            "decode_strategy must be one of hard_dfa, unconstrained, "
            "product_compact_learner_dfa, or product_hmm_dfa"
        ) from exc


def _mask_label_logits(logits: torch.Tensor, allowed_labels: set[int], fill_value: float = -1e9) -> torch.Tensor:
    """Mask logits to allowed labels and keep disallowed labels at fill_value."""
    if logits.dim() != 1:
        raise ValueError(f"expected compact label logits to be 1D, got shape {tuple(logits.shape)}")
    masked = torch.full_like(logits, fill_value)
    for label in allowed_labels:
        if 0 <= int(label) < logits.numel():
            masked[int(label)] = logits[int(label)]
    if torch.all(masked <= fill_value / 2):
        raise ValueError("DFA mask removed all compact labels")
    return masked


def _filter_sampling_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Apply top-k and/or top-p filtering to a 1D logit vector."""
    filtered = logits.clone()
    if top_k is not None:
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        if top_k < filtered.numel():
            values, _indices = torch.topk(filtered, top_k)
            filtered = filtered.masked_fill(filtered < values[-1], fill_value)
    if top_p is not None:
        top_p = float(top_p)
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = fill_value
    return filtered


def _select_label(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> tuple[int, float]:
    """Sample or argmax a label and return label id with selected log-prob."""
    if float(temperature) <= 0.0:
        label = int(torch.argmax(logits).item())
        log_probs = torch.log_softmax(logits, dim=-1)
        return label, float(log_probs[label].detach().item())
    filtered = _filter_sampling_logits(logits / float(temperature), top_k=top_k, top_p=top_p)
    if torch.all(filtered <= -5e8):
        filtered = logits / float(temperature)
    probs = torch.softmax(filtered, dim=-1)
    label = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    log_prob = torch.log(probs[label].clamp_min(torch.finfo(probs.dtype).tiny))
    return label, float(log_prob.detach().item())


def _token_id_for_generated_label(model, vocabulary: TokenVocabulary, label: int) -> int:
    """Map generated label id to token id via model override or vocabulary."""
    if hasattr(model, "token_id_for_label"):
        return int(model.token_id_for_label(int(label)))
    return int(vocabulary.token_id_for_label(int(label)))


def _emittable_labels(model, vocabulary: TokenVocabulary) -> set[int]:
    """Return labels that can be mapped to token ids for generation."""
    labels = set()
    for label in range(vocabulary.label_count):
        try:
            _token_id_for_generated_label(model, vocabulary, label)
        except ValueError:
            continue
        labels.add(label)
    return labels


def _backend_label_logits(
    generator: HuggingFaceGenerationAdapter,
    input_ids: Sequence[int],
    compact_model,
    vocabulary: TokenVocabulary,
    device: torch.device,
    *,
    label_count: int,
) -> torch.Tensor | None:
    """Project one-step HF token logits into compact label-logit space."""
    model = getattr(generator, "model", None)
    if model is None:
        return None
    try:
        ids = torch.tensor([list(int(item) for item in input_ids)], dtype=torch.long, device=device)
        output = model(ids)
    except Exception:
        return None
    token_logits = output.logits[0, -1, :] if hasattr(output, "logits") else output
    if token_logits.dim() != 1:
        return None
    label_logits = token_logits.new_full((int(label_count),), -1e9)
    for label in range(int(label_count)):
        try:
            token_id = _token_id_for_generated_label(compact_model, vocabulary, label)
        except ValueError:
            continue
        if 0 <= token_id < token_logits.numel():
            label_logits[label] = token_logits[token_id]
    return label_logits


def _constraint_repair_suggestions(
    dfa: DFA,
    labels: Sequence[int],
    vocabulary: TokenVocabulary,
) -> list[str]:
    """Generate human-readable DFA repair hints for an invalid label sequence."""
    if dfa.accepts(labels):
        return ["sequence already satisfies the hard DFA"]
    state = dfa.start_state
    for index, label in enumerate(labels):
        next_state = dfa.step(state, int(label))
        if next_state is None or next_state in dfa.dead_states:
            token = vocabulary.token_for_label(int(label))
            return [f"revise token {index} ({token!r}) or choose one of the DFA-allowed next labels before it"]
        state = next_state
    allowed = sorted(int(label) for label in dfa.allowed_tokens(state, remaining_steps=1))
    if allowed:
        rendered = ", ".join(repr(vocabulary.token_for_label(label)) for label in allowed[:5])
        return [f"append or replace with a DFA-allowed next token such as {rendered}"]
    return ["no DFA-specific repair suggestion available"]


def _constraint_name_repair_suggestions(constraints: Sequence[Any], tokens: Sequence[str]) -> list[str]:
    """Derive readable repair hints from discovered hard-constraint names."""

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    suggestions: list[str] = []
    for constraint in constraints:
        name = getattr(constraint, "name", "")
        if not name:
            continue
        required_match = re.fullmatch(r"at least (\d+) ('[^']+'|\{.+\}|tokens .+) token\(s\) are generated", name)
        forbidden_match = re.fullmatch(r"no ('[^']+'|\{.+\}|tokens .+) token\(s\) are generated", name)
        capped_match = re.fullmatch(r"at most (\d+) ('[^']+'|\{.+\}|tokens .+) token\(s\) are generated", name)
        if required_match:
            limit = int(required_match.group(1))
            target = required_match.group(2)
            token = _single_constraint_token(target)
            if token is not None and counts.get(token, 0) < limit:
                suggestions.append(f"add {token!r}")
        elif forbidden_match:
            token = _single_constraint_token(forbidden_match.group(1))
            if token is not None and counts.get(token, 0) > 0:
                suggestions.append(f"remove {token!r}")
        elif capped_match:
            limit = int(capped_match.group(1))
            token = _single_constraint_token(capped_match.group(2))
            if token is not None and counts.get(token, 0) > limit:
                suggestions.append(f"remove extra {token!r}")
    return suggestions


def _single_constraint_token(fragment: str) -> str | None:
    """Return the single token encoded in a readable constraint fragment."""

    token_match = re.fullmatch(r"'([^']+)'", fragment)
    if token_match:
        return token_match.group(1)
    return None


def _candidate_full_ids(prompt_ids: torch.Tensor, candidate: GenerationCandidate) -> list[int]:
    """Return full token ids by concatenating prompt and candidate token ids."""
    return _flat_ids(prompt_ids) + list(candidate.token_ids or ())


def _flat_ids(input_ids: torch.Tensor | Sequence[int]) -> list[int]:
    """Flatten tensor/sequence token ids into a Python int list."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.numel() == 0:
            return []
        return [int(item) for item in input_ids.reshape(-1).tolist()]
    return [int(item) for item in input_ids]


def _coerce_weights(weights: HybridScoreWeights | Mapping[str, float] | None) -> HybridScoreWeights:
    """Normalize user-provided weight config to HybridScoreWeights."""
    if weights is None:
        return HybridScoreWeights()
    if isinstance(weights, HybridScoreWeights):
        return weights
    return HybridScoreWeights(**{key: float(value) for key, value in weights.items()})


def _resolve_backend(backend: str, generator) -> str:
    """Resolve backend='auto' using generator adapter type."""
    if backend != "auto":
        return backend
    if isinstance(generator, HuggingFaceGenerationAdapter):
        return "hf"
    if isinstance(generator, OpenAIResponsesAdapter):
        return "openai"
    return "precomputed"


def _filter_kwargs(values: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Return a dictionary containing only keys present in allowed."""
    return {key: value for key, value in values.items() if key in allowed}
