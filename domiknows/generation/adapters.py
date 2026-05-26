"""Adapter layer that bridges generation backends to DomiKnowS constraints.

Each adapter wraps a concrete generation backend (HuggingFace Transformers,
OpenAI Responses API, …) and exposes a uniform interface so that the rest of
the DomiKnowS pipeline can work with any backend without modification.

Class overview:
- ``GenerationResult``: lightweight dataclass returned by every adapter.
- ``HuggingFaceGenerationAdapter``: wraps a HuggingFace model + tokenizer;
  supports constrained greedy / beam / sampling decoding and training loss
  computation.
- ``OpenAIResponsesAdapter``: wraps the OpenAI Responses API; supports text
  generation and DFA-based post-hoc verification only (no gradient access).

Adapter capability flags (class attributes):
- ``supports_training_loss``: whether the adapter can compute a
  differentiable loss for DomiKnowS training.
- ``supports_hard_decoding``: whether the adapter can enforce DFA constraints
  token-by-token during generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .learners.dfa.core import DFA
from .learners.dfa.visualization import explain_dfa_rejection
from .decoder import (
    ConstrainedGenerationResult,
    constrained_beam_search_decode,
    constrained_greedy_decode,
    constrained_sample_decode,
)
from .vocabulary import TokenVocabulary


@dataclass
class GenerationResult:
    """Unified result object returned by all generation adapters.

    Attributes:
        text: The decoded output string.
        token_ids: Optional list of raw token IDs from the backend tokenizer.
            ``None`` when the backend does not expose token-level output or no
            tokenizer was provided.
        labels: Optional list of DomiKnowS vocabulary label IDs corresponding
            to *token_ids*.  Populated only when the caller encodes the output
            through a :class:`~.vocabulary.TokenVocabulary`.
        accepted: Whether the generated sequence was accepted by the
            constraint DFA used during/after generation.  ``None`` when no DFA
            check was performed.
        raw: The unmodified backend response object, preserved for debugging
            or downstream inspection.
        rejection: Optional human-readable DFA rejection explanation. Populated
            only by verification helpers when requested and the output is not
            accepted.
    """
    text: str
    token_ids: list[int] | None = None
    labels: list[int] | None = None
    accepted: bool | None = None
    raw: Any = None
    rejection: str | None = None


class HuggingFaceGenerationAdapter:
    """Adapter for HuggingFace Transformers models.

    Wraps a HuggingFace ``PreTrainedModel`` and its associated tokenizer,
    exposing constrained greedy, beam-search, and sampling decoding while
    supporting gradient computation for DomiKnowS training losses.

    Class attributes:
        supports_training_loss (bool): ``True`` — gradients flow through the
            model's logits, enabling DomiKnowS loss computation.
        supports_hard_decoding (bool): ``True`` — DFA constraints can be
            enforced token-by-token during autoregressive generation.
    """

    supports_training_loss = True
    supports_hard_decoding = True

    def __init__(self, model, tokenizer, vocabulary: TokenVocabulary):
        """Initialise the adapter.

        Args:
            model: A HuggingFace ``PreTrainedModel`` (or compatible) instance.
            tokenizer: The tokenizer associated with *model*.
            vocabulary: A :class:`~.vocabulary.TokenVocabulary` that maps
                token IDs to DomiKnowS label indices.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

    def constrained_greedy(
        self,
        input_ids: torch.Tensor,
        dfa: DFA,
        max_new_tokens: int,
        use_cache: bool = True,
    ) -> ConstrainedGenerationResult:
        """Run DFA-constrained greedy decoding from *input_ids*.

        At each step only tokens allowed by the current DFA state are
        considered; the highest-logit permitted token is chosen.  Generation
        stops when the EOS token is produced or *max_new_tokens* is reached.

        Args:
            input_ids: Prompt token IDs as a ``(1, T)`` or ``(T,)`` tensor.
            dfa: Constraint :class:`~.learners.DFA` whose alphabet must be
                compatible with ``self.vocabulary``.
            max_new_tokens: Hard upper bound on generated tokens.
            use_cache: Whether to use ``past_key_values`` caching.  Set to
                ``False`` to force full re-encoding at every step.

        Returns:
            A :class:`~.decoder.ConstrainedGenerationResult` containing the
            generated token IDs, label sequence, final DFA state, and
            whether the final state is accepting.
        """
        return constrained_greedy_decode(
            self.model,
            input_ids,
            self.vocabulary,
            dfa,
            max_new_tokens,
            # Pass the tokenizer's EOS id so the decoder knows when to stop.
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            use_cache=use_cache,
        )

    def constrained_beam_search(
        self,
        input_ids: torch.Tensor,
        dfa: DFA,
        max_new_tokens: int,
        beam_size: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        num_return_sequences: int = 1,
        use_cache: bool = True,
    ) -> ConstrainedGenerationResult:
        """Run DFA-constrained beam search from *input_ids*.

        Each beam carries its own DFA state.  At every expansion step logits
        are masked by the active DFA state before log-probabilities are
        computed, so no returned candidate can violate the constraint.

        Args:
            input_ids: Prompt token IDs as a ``(1, T)`` or ``(T,)`` tensor.
            dfa: Constraint :class:`~.learners.DFA` compatible with
                ``self.vocabulary``.
            max_new_tokens: Hard upper bound on generated tokens per beam.
            beam_size: Number of active beams to maintain at each step.
                Must be ≥ 1.
            length_penalty: Exponent applied to sequence length when ranking
                finished candidates.  Values > 1 favour longer sequences;
                values < 1 favour shorter ones.  Must be positive.
            early_stopping: When ``True``, stop as soon as
                *num_return_sequences* accepted candidates have been found.
            num_return_sequences: Number of top-ranked candidates to return.
                Must be ≥ 1.
            use_cache: Whether to use ``past_key_values`` caching.  Set to
                ``False`` to force full re-encoding at every step.

        Returns:
            A :class:`~.decoder.ConstrainedGenerationResult` for the
            best-ranked candidate, with ``candidates`` populated for all
            *num_return_sequences* returned sequences.
        """
        return constrained_beam_search_decode(
            self.model,
            input_ids,
            self.vocabulary,
            dfa,
            max_new_tokens,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            beam_size=beam_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
        )

    def constrained_sample(
        self,
        input_ids: torch.Tensor,
        dfa: DFA,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> ConstrainedGenerationResult:
        """Run DFA-constrained stochastic decoding from *input_ids*.

        The DFA mask is applied first to remove tokens disallowed by the
        current constraint state.  Temperature scaling, top-k, and nucleus
        (top-p) filtering are then applied inside the constrained token set
        before a token is drawn via ``torch.multinomial``.

        Args:
            input_ids: Prompt token IDs as a ``(1, T)`` or ``(T,)`` tensor.
            dfa: Constraint :class:`~.learners.DFA` compatible with
                ``self.vocabulary``.
            max_new_tokens: Hard upper bound on generated tokens.
            temperature: Softmax temperature.  Values < 1 sharpen the
                distribution; values > 1 flatten it.  Must be positive.
            top_k: If set, only the *top_k* highest-logit tokens (after DFA
                masking) are kept before sampling.  Must be ≥ 1 when provided.
            top_p: Nucleus-sampling threshold in ``(0, 1]``.  Applied after
                *top_k* when both are given.
            generator: Optional :class:`torch.Generator` for reproducible
                sampling.
            use_cache: Whether to use ``past_key_values`` caching.  Set to
                ``False`` to force full re-encoding at every step.

        Returns:
            A :class:`~.decoder.ConstrainedGenerationResult` with the
            generated token IDs, label sequence, final DFA state,
            acceptance flag, cumulative log-probability, and per-step scores.
        """
        return constrained_sample_decode(
            self.model,
            input_ids,
            self.vocabulary,
            dfa,
            max_new_tokens,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            use_cache=use_cache,
        )


class OpenAIResponsesAdapter:
    """OpenAI Responses API adapter.

    OpenAI generation is verification-only in v1 because hosted APIs do not
    expose a per-token decoder hook or gradients for DomiKnowS loss.

    Typical usage:
    1. Call :meth:`generate` to obtain a text response from the model.
    2. Call :meth:`verify_result` or :meth:`generate_and_verify` to map the
       text into DomiKnowS labels and check it against a DFA.

    Class attributes:
        supports_training_loss (bool): ``False`` — no gradient access.
        supports_hard_decoding (bool): ``False`` — DFA constraints can only
            be applied post-hoc, not token-by-token.
    """

    supports_training_loss = False
    supports_hard_decoding = False

    def __init__(self, client=None, model: str = "gpt-4.1-mini", tokenizer=None):
        """Initialise the adapter.

        Args:
            client: An ``openai.OpenAI`` client instance.  If ``None``, one is
                created with default settings (reads ``OPENAI_API_KEY`` from
                the environment).
            model: OpenAI model identifier string.  Defaults to
                ``"gpt-4.1-mini"``.
            tokenizer: Optional tokenizer used to convert generated text into
                token IDs.  When provided, :meth:`generate` populates
                ``GenerationResult.token_ids``.
        """
        if client is None:
            # Lazily import openai to avoid a hard dependency at import time.
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_output_tokens: int | None = None, **kwargs) -> GenerationResult:
        """Send *prompt* to the OpenAI Responses API and return the output.

        Args:
            prompt: The user input / instruction string.
            max_output_tokens: Optional cap on the number of tokens the model
                may generate.  Maps directly to the ``max_output_tokens``
                request field.
            **kwargs: Additional keyword arguments forwarded verbatim to
                ``client.responses.create``.

        Returns:
            A :class:`GenerationResult` with ``text`` set to the model's
            response and ``token_ids`` populated when a tokenizer is
            available.  ``raw`` holds the unmodified API response object.
        """
        # Build the Responses API request dict; inject max_output_tokens only
        # when the caller supplied a value.
        request = {"model": self.model, "input": prompt, **kwargs}
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        response = self.client.responses.create(**request)
        # Prefer the convenience attribute; fall back to manual extraction.
        text = getattr(response, "output_text", None)
        if text is None:
            text = self._extract_output_text(response)
        token_ids = None
        if self.tokenizer is not None:
            token_ids = list(self.tokenizer.encode(text))
        return GenerationResult(text=text, token_ids=token_ids, raw=response)

    def verify_result(
        self,
        result: GenerationResult,
        vocabulary: TokenVocabulary,
        dfa: DFA,
        *,
        explain: bool = False,
    ) -> GenerationResult:
        """Encode and verify an OpenAI output against a constraint DFA.

        This is the adapter-level generate-then-verify bridge for hosted
        OpenAI and OpenAI-compatible servers.  It does not mutate *result*;
        instead it returns a new :class:`GenerationResult` with compact labels
        and ``accepted`` populated.

        Args:
            result: A previously generated OpenAI result.
            vocabulary: Compact generation vocabulary used by the DFA.
            dfa: Constraint DFA over compact vocabulary label IDs.
            explain: When ``True``, include a human-readable rejection reason
                for rejected outputs.

        Returns:
            A verified :class:`GenerationResult`.
        """
        tokenizer = self._encoding_tokenizer(vocabulary)
        token_ids = list(tokenizer.encode(result.text))
        labels = vocabulary.labels_for_token_ids(token_ids)
        accepted = dfa.accepts(labels)
        rejection = explain_dfa_rejection(dfa, labels) if explain and not accepted else None
        return GenerationResult(
            text=result.text,
            token_ids=token_ids,
            labels=labels,
            accepted=accepted,
            raw=result.raw,
            rejection=rejection,
        )

    def generate_and_verify(
        self,
        prompt: str,
        vocabulary: TokenVocabulary,
        dfa: DFA,
        *,
        max_output_tokens: int | None = None,
        explain: bool = False,
        **kwargs,
    ) -> GenerationResult:
        """Generate text, then encode and verify it against *dfa*.

        Args:
            prompt: User input / instruction string.
            vocabulary: Compact generation vocabulary used by the DFA.
            dfa: Constraint DFA over compact vocabulary label IDs.
            max_output_tokens: Optional cap forwarded to :meth:`generate`.
            explain: When ``True``, include a rejection reason on failure.
            **kwargs: Additional request fields forwarded to ``generate``.

        Returns:
            A verified :class:`GenerationResult` with ``accepted`` populated.
        """
        result = self.generate(prompt, max_output_tokens=max_output_tokens, **kwargs)
        return self.verify_result(result, vocabulary, dfa, explain=explain)

    def encode_output(self, text: str, vocabulary: TokenVocabulary) -> list[int]:
        """Encode *text* into DomiKnowS vocabulary label IDs.

        Uses either the adapter's own tokenizer (if set) or the tokenizer
        embedded in *vocabulary* to first convert the text to token IDs, then
        maps those IDs to label indices via
        :meth:`~.vocabulary.TokenVocabulary.labels_for_token_ids`.

        Args:
            text: Raw text string to encode (typically from :meth:`generate`).
            vocabulary: The :class:`~.vocabulary.TokenVocabulary` that defines
                the label mapping.

        Returns:
            A list of integer label IDs suitable for DFA acceptance checking
            or DomiKnowS constraint evaluation.

        Raises:
            ValueError: If no tokenizer is available on the adapter or in
                *vocabulary*.
        """
        tokenizer = self._encoding_tokenizer(vocabulary)
        return vocabulary.labels_for_token_ids(tokenizer.encode(text))

    def _encoding_tokenizer(self, vocabulary: TokenVocabulary):
        """Return the tokenizer used for OpenAI output encoding."""
        if self.tokenizer is not None:
            return self.tokenizer
        if vocabulary.tokenizer is not None:
            return vocabulary.tokenizer
        raise ValueError("a tokenizer is required to encode OpenAI output")

    @staticmethod
    def _extract_output_text(response) -> str:
        """Extract concatenated text content from an OpenAI Responses API object.

        Traverses ``response.output[*].content[*].text`` and joins all text
        fragments.  This fallback is used when the response object does not
        expose a top-level ``output_text`` convenience attribute.

        Args:
            response: An OpenAI Responses API response object.

        Returns:
            Concatenated text from all content parts, or an empty string if
            no text content is found.
        """
        parts = []
        for output in getattr(response, "output", []) or []:
            for content in getattr(output, "content", []) or []:
                text = getattr(content, "text", None)
                if text is not None:
                    parts.append(text)
        return "".join(parts)
