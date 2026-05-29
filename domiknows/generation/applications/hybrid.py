"""Hybrid generation controller/scorer utilities.

This module keeps the open-vocabulary generator and the compact DomiKnowS
generation head in separate roles:

* a large backend proposes text/token candidates;
* the DFA verifies hard graph constraints;
* the compact head scores domain/style/latent preferences and risk.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .adapters import GenerationResult, HuggingFaceGenerationAdapter, OpenAIResponsesAdapter
from ..dfa.core import DFA
from ..dfa.visualization import explain_dfa_rejection
from ..dfa.decoder import ConstrainedGenerationResult
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
        super().__init__()
        if not bundle_names:
            raise ValueError("bundle_names must not be empty")
        self.bundle_names = tuple(str(name) for name in bundle_names)
        self.embedding = torch.nn.Embedding(int(vocab_size), int(hidden_size))
        self.classifier = torch.nn.Linear(int(hidden_size), len(self.bundle_names))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        constraint_selector: Any = None,
    ):
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
        max_new_tokens: int = 16,
        hard_decode: bool = True,
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
                hard_decode=hard_decode,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
                explain=explain,
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
        except ValueError:
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

    def _generate_candidates(
        self,
        prompt,
        prompt_ids: torch.Tensor,
        num_candidates: int,
        *,
        backend: str,
        max_new_tokens: int,
        hard_decode: bool,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        explain: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        backend = _resolve_backend(backend, self.generator)
        if backend == "hf":
            return self._generate_hf_candidates(
                prompt_ids,
                num_candidates,
                max_new_tokens=max_new_tokens,
                hard_decode=hard_decode,
                temperature=temperature,
                top_p=top_p,
                generator_seed=generator_seed,
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
        max_new_tokens: int,
        hard_decode: bool,
        temperature: float,
        top_p: float | None,
        generator_seed: int,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
        if not isinstance(self.generator, HuggingFaceGenerationAdapter):
            raise ValueError("HF candidate generation requires a HuggingFaceGenerationAdapter")
        results = []
        prompt_len = int(prompt_ids.shape[-1])
        for index in range(int(num_candidates)):
            if hard_decode:
                result = self.generator.constrained_sample(
                    prompt_ids,
                    self.dfa,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    generator=torch.Generator(device=prompt_ids.device).manual_seed(generator_seed + index),
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
            else:
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
        return results

    def _generate_openai_candidates(
        self,
        prompt: str,
        num_candidates: int,
        *,
        max_output_tokens: int,
        explain: bool,
        **generate_kwargs,
    ) -> list[GenerationCandidate]:
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
        tokenizer = self.tokenizer or getattr(self.vocabulary, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer is required to encode text candidates")
        return list(int(item) for item in tokenizer.encode(text))

    def _head_logprob(self, prompt_ids: torch.Tensor, candidate: GenerationCandidate) -> tuple[float, torch.Tensor | None]:
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
    ids = torch.tensor([list(int(item) for item in input_ids)], dtype=torch.long, device=prompt_ids.device)
    if hasattr(model, "next_label_logits"):
        return model.next_label_logits(ids)
    try:
        output = model(ids)
    except NotImplementedError as exc:
        raise ValueError("scorer head must expose next_label_logits for risk/repair scoring") from exc
    logits = output.logits if hasattr(output, "logits") else output
    return logits[0, -1, :]


def _constraint_repair_suggestions(
    dfa: DFA,
    labels: Sequence[int],
    vocabulary: TokenVocabulary,
) -> list[str]:
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


def _candidate_full_ids(prompt_ids: torch.Tensor, candidate: GenerationCandidate) -> list[int]:
    return _flat_ids(prompt_ids) + list(candidate.token_ids or ())


def _flat_ids(input_ids: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.numel() == 0:
            return []
        return [int(item) for item in input_ids.reshape(-1).tolist()]
    return [int(item) for item in input_ids]


def _coerce_weights(weights: HybridScoreWeights | Mapping[str, float] | None) -> HybridScoreWeights:
    if weights is None:
        return HybridScoreWeights()
    if isinstance(weights, HybridScoreWeights):
        return weights
    return HybridScoreWeights(**{key: float(value) for key, value in weights.items()})


def _resolve_backend(backend: str, generator) -> str:
    if backend != "auto":
        return backend
    if isinstance(generator, HuggingFaceGenerationAdapter):
        return "hf"
    if isinstance(generator, OpenAIResponsesAdapter):
        return "openai"
    return "precomputed"


def _filter_kwargs(values: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in allowed}
