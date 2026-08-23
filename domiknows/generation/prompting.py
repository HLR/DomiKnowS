"""Consistent chat-template prompt encoding for label-space generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LabelPrefixPromptEncoding:
    """Tokenized user prompt plus an assistant-side label prefix.

    ``boundary_positions[0]`` is the hidden-state position used to predict the
    first label.  Every later boundary follows one complete prefix label and is
    therefore the position used to predict the next label.
    """

    input_ids: tuple[int, ...]
    boundary_positions: tuple[int, ...]
    rendered_text: str
    used_chat_template: bool


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids", ())
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token_id) for token_id in value]


def _apply_chat_template(tokenizer, messages, *, tokenize: bool, enable_thinking: bool):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
        # Qwen3 otherwise inserts a thinking preamble before the action labels.
        # Other Hugging Face templates accept this as an unused template value.
        "enable_thinking": bool(enable_thinking),
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Compatibility with older/custom tokenizers whose Python wrapper does
        # not forward arbitrary template variables.
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_label_prefix_prompt(
    tokenizer,
    user_content: str,
    prefix_tokens: Sequence[str] = (),
    *,
    max_length: int | None = None,
    fallback_assistant_header: str = "Generated action tokens:",
    enable_thinking: bool = False,
) -> LabelPrefixPromptEncoding:
    """Encode one user message and an unfinished assistant label sequence.

    The chat generation marker is rendered exactly once. Prefix labels are then
    appended as assistant continuation tokens, so no end-of-turn marker appears
    between labels.  Training, rollout sampling, and rescoring can all consume
    this same encoding and its next-label boundary positions.
    """

    messages = [{"role": "user", "content": str(user_content)}]
    use_chat = bool(
        callable(getattr(tokenizer, "apply_chat_template", None))
        and getattr(tokenizer, "chat_template", None)
    )
    if use_chat:
        base_ids = _token_ids(
            _apply_chat_template(
                tokenizer,
                messages,
                tokenize=True,
                enable_thinking=enable_thinking,
            )
        )
        rendered = str(
            _apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        )
    else:
        rendered = f"{user_content.rstrip()}\n{fallback_assistant_header}"
        base_ids = _token_ids(tokenizer(rendered, add_special_tokens=True))

    if not base_ids:
        fallback_id = getattr(tokenizer, "eos_token_id", None)
        base_ids = [int(fallback_id) if fallback_id is not None else 0]

    input_ids = list(base_ids)
    boundaries = [len(input_ids) - 1]
    for token in prefix_tokens:
        chunk = " " + str(token)
        chunk_ids = _token_ids(tokenizer(chunk, add_special_tokens=False))
        if not chunk_ids:
            chunk_ids = _token_ids(tokenizer(str(token), add_special_tokens=False))
        if not chunk_ids:
            raise ValueError(f"label prefix token {token!r} has no tokenizer representation")
        input_ids.extend(chunk_ids)
        rendered += chunk
        boundaries.append(len(input_ids) - 1)

    if max_length is not None and len(input_ids) > int(max_length):
        offset = len(input_ids) - int(max_length)
        input_ids = input_ids[offset:]
        boundaries = [max(0, position - offset) for position in boundaries]

    return LabelPrefixPromptEncoding(
        input_ids=tuple(input_ids),
        boundary_positions=tuple(boundaries),
        rendered_text=rendered,
        used_chat_template=use_chat,
    )


__all__ = ["LabelPrefixPromptEncoding", "encode_label_prefix_prompt"]
