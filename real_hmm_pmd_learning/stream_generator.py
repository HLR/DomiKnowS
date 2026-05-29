"""Deterministic mock generator stream for the beginner PMD demo."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator, Sequence

import torch

try:
    from .graph import EOS_TOKEN, VOCAB
    from .learned_model_interface import labels_for_symbols
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph import EOS_TOKEN, VOCAB
    from learned_model_interface import labels_for_symbols


NORMAL_SYMBOLS = tuple(symbol for symbol in VOCAB if symbol != EOS_TOKEN)
NON_B_SYMBOLS = tuple(symbol for symbol in NORMAL_SYMBOLS if symbol != "B")
NORMAL_WEIGHT = 8
END_WEIGHT = 3

PROMPTS = {
    "AB": {
        "token_id": 90,
        "text": "prefer A and B",
        "description": "The generator gives A and B higher probability than C and D.",
        "preferred_symbols": ("A", "B"),
    },
    "CD": {
        "token_id": 91,
        "text": "prefer C and D",
        "description": "The generator gives C and D higher probability than A and B.",
        "preferred_symbols": ("C", "D"),
    },
    "short": {
        "token_id": 92,
        "text": "make a short string",
        "description": "The generator usually stops quickly with END.",
        "preferred_symbols": ("A", "C"),
    },
}
PROMPT_ORDER = ("AB", "CD", "short")
PROMPT_VOCAB_SIZE = 128
PREFERRED_WEIGHT = 9
OTHER_WEIGHT = 2


@dataclass(frozen=True)
class StreamTrainingExample:
    """One generated string plus the DomiKnowS sample derived from it."""

    name: str
    prompt_name: str
    prompt_text: str
    prompt_token_id: int
    symbols: tuple[str, ...]
    labels: tuple[int, ...]
    sample_data: dict


def make_sample_data(bundle, symbols: Sequence[str], *, prompt_name: str = "AB") -> dict:
    """Create one PMD training example from a generated symbol sequence."""
    labels = labels_for_symbols(bundle, symbols)
    prompt = prompt_spec(prompt_name)
    return {
        "instruction_tokens": torch.tensor([[prompt["token_id"]]], dtype=torch.long),
        "sequence_labels_input": torch.tensor([labels], dtype=torch.long),
    }


def prompt_spec(prompt_name: str) -> dict[str, object]:
    """Return one prompt definition by name."""
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt_name!r}; expected one of {tuple(PROMPTS)}")
    return PROMPTS[prompt_name]


def _weighted_choice(rng: random.Random, symbols: Sequence[str], weights: Sequence[int]) -> str:
    return rng.choices(tuple(symbols), weights=tuple(weights), k=1)[0]


def _prompt_weights(prompt_name: str, choices: Sequence[str]) -> list[int]:
    preferred = set(prompt_spec(prompt_name)["preferred_symbols"])
    return [PREFERRED_WEIGHT if symbol in preferred else OTHER_WEIGHT for symbol in choices]


def _sample_prompt_biased_symbol(
    rng: random.Random,
    *,
    prompt_name: str,
    choices: Sequence[str],
) -> str:
    return _weighted_choice(rng, choices, _prompt_weights(prompt_name, choices))


def _bounded_body_length(rng: random.Random, *, max_length: int, low: int, high: int) -> int:
    high = min(int(high), max(1, int(max_length) - 1))
    low = min(int(low), high)
    return rng.randint(low, high)


def _random_valid_symbols(rng: random.Random, *, max_length: int, prompt_name: str) -> tuple[str, ...]:
    if prompt_name == "AB":
        body_length = _bounded_body_length(rng, max_length=max_length, low=3, high=6)
        b_index = rng.randrange(body_length)
        body = [
            _sample_prompt_biased_symbol(rng, prompt_name=prompt_name, choices=NON_B_SYMBOLS)
            for _index in range(body_length)
        ]
        body[b_index] = "B"
        return tuple((*body, EOS_TOKEN))
    if prompt_name == "CD":
        body_length = _bounded_body_length(rng, max_length=max_length, low=3, high=7)
        body = [
            _sample_prompt_biased_symbol(rng, prompt_name=prompt_name, choices=NON_B_SYMBOLS)
            for _index in range(body_length)
        ]
        return tuple((*body, EOS_TOKEN))
    if prompt_name == "short":
        body_length = _bounded_body_length(rng, max_length=max_length, low=1, high=2)
        choices = NORMAL_SYMBOLS
        body = [
            _sample_prompt_biased_symbol(rng, prompt_name=prompt_name, choices=choices)
            for _index in range(body_length)
        ]
        return tuple((*body, EOS_TOKEN))

    symbols: list[str] = []
    has_b = False
    for step in range(int(max_length)):
        normal_choices = NON_B_SYMBOLS if has_b else NORMAL_SYMBOLS
        choices = (*normal_choices, EOS_TOKEN)
        weights = (*([NORMAL_WEIGHT] * len(normal_choices)), END_WEIGHT)
        symbol = _weighted_choice(rng, choices, weights)
        if symbol == "B":
            has_b = True
        symbols.append(symbol)
        if symbol == EOS_TOKEN:
            break
    if symbols and symbols[-1] != EOS_TOKEN:
        symbols[-1] = EOS_TOKEN
    return tuple(symbols)


def _random_invalid_symbols(rng: random.Random, *, max_length: int, prompt_name: str) -> tuple[str, ...]:
    max_length = int(max_length)
    if max_length <= 2:
        return ("B", "B")
    if prompt_name == "short":
        length = min(max_length, rng.randint(3, 5))
    elif prompt_name == "CD":
        length = min(max_length, rng.randint(4, 8))
    else:
        length = min(max_length, rng.randint(5, 9))
    body_length = length - 1
    first = rng.randrange(body_length)
    second = rng.randrange(body_length - 1)
    if second >= first:
        second += 1
    symbols = [
        _sample_prompt_biased_symbol(rng, prompt_name=prompt_name, choices=NON_B_SYMBOLS)
        for _index in range(body_length)
    ]
    symbols[first] = "B"
    symbols[second] = "B"
    symbols.append(EOS_TOKEN)
    return tuple(symbols)


def mock_generator_stream(
    *,
    count: int,
    seed: int = 0,
    max_length: int = 6,
) -> Iterator[tuple[str, str, tuple[str, ...]]]:
    """Yield deterministic generator proposals.

    The sequence intentionally contains both rule-following and rule-breaking
    outputs so users can see PMD train on generator behavior.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    rng = random.Random(int(seed))
    names = ("valid", "invalid")
    offset = int(seed) % len(names)
    for index in range(int(count)):
        name = names[(offset + index) % len(names)]
        prompt_name = PROMPT_ORDER[(int(seed) + index) % len(PROMPT_ORDER)]
        if name == "valid":
            yield name, prompt_name, _random_valid_symbols(rng, max_length=max_length, prompt_name=prompt_name)
        else:
            yield name, prompt_name, _random_invalid_symbols(rng, max_length=max_length, prompt_name=prompt_name)


def stream_training_examples(
    bundle,
    *,
    count: int,
    seed: int = 0,
    max_length: int = 6,
) -> tuple[StreamTrainingExample, ...]:
    """Generate a stream batch and convert every output into PMD data."""
    examples: list[StreamTrainingExample] = []
    for name, prompt_name, symbols in mock_generator_stream(count=count, seed=seed, max_length=max_length):
        labels = tuple(labels_for_symbols(bundle, symbols))
        prompt = prompt_spec(prompt_name)
        examples.append(
            StreamTrainingExample(
                name=name,
                prompt_name=prompt_name,
                prompt_text=str(prompt["text"]),
                prompt_token_id=int(prompt["token_id"]),
                symbols=tuple(symbols),
                labels=labels,
                sample_data=make_sample_data(bundle, symbols, prompt_name=prompt_name),
            )
        )
    return tuple(examples)


@dataclass(frozen=True)
class GeneratorTrainingSource:
    """Small object that turns generator outputs into PMD training batches."""

    bundle: object
    stream_count: int = 4
    seed: int = 0
    max_length: int = 6

    def __post_init__(self) -> None:
        if self.stream_count <= 0:
            raise ValueError("stream_count must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")

    def next_batch(self, step: int = 0) -> tuple[StreamTrainingExample, ...]:
        """Read the next deterministic generator batch for one PMD step."""
        return stream_training_examples(
            self.bundle,
            count=self.stream_count,
            seed=self.seed + int(step),
            max_length=self.max_length,
        )

    def training_data(self, batch: Sequence[StreamTrainingExample]) -> list[dict]:
        """Return the DomiKnowS sample dictionaries for a generated batch."""
        return [example.sample_data for example in batch]
