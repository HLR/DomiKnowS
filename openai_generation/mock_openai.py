"""Offline OpenAI-compatible mocks for the openai_generation task."""
from __future__ import annotations

from dataclasses import dataclass


class MockTokenizer:
    """Tiny tokenizer with one id per demo token."""

    eos_token = "<eos>"
    eos_token_id = 0

    def __init__(self):
        self.token_to_id = {
            "<eos>": 0,
            " The": 1,
            " cat": 2,
            " mat": 3,
            " dog": 4,
            "<prompt>": 5,
        }
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

    def encode(self, text):
        ids = []
        cursor = 0
        while cursor < len(text):
            matched = None
            for token in sorted(self.token_to_id, key=len, reverse=True):
                if text.startswith(token, cursor):
                    matched = token
                    break
            if matched is not None:
                ids.append(self.token_to_id[matched])
                cursor += len(matched)
                continue
            if text[cursor].isspace():
                cursor += 1
                continue
            next_space = text.find(" ", cursor + 1)
            if next_space == -1:
                next_space = len(text)
            surface = " " + text[cursor:next_space]
            ids.append(self.token_to_id.get(surface, self.token_to_id["<prompt>"]))
            cursor = next_space
        return ids or [self.token_to_id["<prompt>"]]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.id_to_token.get(int(token_id), "<unk>") for token_id in token_ids)


@dataclass
class MockLogprobToken:
    token: str
    logprob: float


class MockResponse:
    """Small object shaped like a Responses API result for adapter tests."""

    def __init__(self, text: str, *, include_logprobs: bool = False):
        self.output_text = text
        self.output = []
        if include_logprobs:
            self.logprobs = [
                MockLogprobToken(token=token, logprob=-0.1 * index)
                for index, token in enumerate(text.split() or [text])
            ]


class MockResponses:
    def __init__(self, text: str, *, include_logprobs: bool = False):
        self.text = text
        self.include_logprobs = include_logprobs
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return MockResponse(self.text, include_logprobs=self.include_logprobs)


class MockOpenAIClient:
    """Minimal client exposing ``client.responses.create(...)``."""

    def __init__(self, text: str, *, include_logprobs: bool = False):
        self.responses = MockResponses(text, include_logprobs=include_logprobs)
