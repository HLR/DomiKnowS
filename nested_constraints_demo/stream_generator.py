"""Mock generator stream for the nested-constraints demo.

Produces a 50/50 mix of rule-satisfying and rule-breaking sequences across
two prompts.  Mirrors the API surface of
:mod:`Tasks.real_hmm_pmd_learning.stream_generator` so the same training
infrastructure (PrimalDualProgram, ModuleLearner sensors, snapshot helpers)
can drive both demos.

Generated sequences always satisfy the EOS-closure rule (no token after END)
and the at-most-one-B rule.  The "valid" path additionally satisfies the
``A xor C`` and forbidden-D rules; the "invalid" path violates exactly one
of them so the constraint loss has something to push against.
"""
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


PROMPTS = {
    "with_A": {
        "token_id": 70,
        "text": "prefer A (no C, no D)",
        "description": "Generator emits sequences containing A, optionally one B, never C or D.",
        "preferred_symbols": ("A",),
        "forbidden_symbols": ("C", "D"),
    },
    "with_C": {
        "token_id": 71,
        "text": "prefer C (no A, no D)",
        "description": "Generator emits sequences containing C, optionally one B, never A or D.",
        "preferred_symbols": ("C",),
        "forbidden_symbols": ("A", "D"),
    },
}
PROMPT_ORDER = ("with_A", "with_C")
PROMPT_VOCAB_SIZE = 128

PREFERRED_WEIGHT = 9
B_WEIGHT = 2


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


def prompt_spec(prompt_name: str) -> dict[str, object]:
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt_name!r}; expected one of {tuple(PROMPTS)}")
    return PROMPTS[prompt_name]


def make_sample_data(bundle, symbols: Sequence[str], *, prompt_name: str) -> dict:
    labels = labels_for_symbols(bundle, symbols)
    prompt = prompt_spec(prompt_name)
    return {
        "instruction_tokens": torch.tensor([[prompt["token_id"]]], dtype=torch.long),
        "sequence_labels_input": torch.tensor([labels], dtype=torch.long),
    }


def _choose(rng: random.Random, choices: Sequence[str], weights: Sequence[int]) -> str:
    return rng.choices(tuple(choices), weights=tuple(weights), k=1)[0]


def _bounded_body_length(rng: random.Random, *, max_length: int, low: int, high: int) -> int:
    high = min(int(high), max(1, int(max_length) - 1))
    low = min(int(low), high)
    return rng.randint(low, high)


def _random_valid_symbols(rng: random.Random, *, max_length: int, prompt_name: str) -> tuple[str, ...]:
    """Sequence satisfying every rule for *prompt_name*."""
    spec = prompt_spec(prompt_name)
    preferred = spec["preferred_symbols"]
    forbidden = set(spec["forbidden_symbols"]) | {"B"}  # treat B specially
    candidates = tuple(symbol for symbol in NORMAL_SYMBOLS if symbol not in forbidden)
    if not candidates:
        # Shouldn't happen with the demo's vocabulary, but guard anyway.
        candidates = preferred
    body_length = _bounded_body_length(rng, max_length=max_length, low=2, high=5)
    weights = [PREFERRED_WEIGHT if symbol in preferred else 1 for symbol in candidates]
    body = [_choose(rng, candidates, weights) for _index in range(body_length)]
    # Optionally drop one B in (preserves "at most one B").  ~50% chance.
    if rng.random() < 0.5 and body_length >= 2:
        body[rng.randrange(body_length)] = "B"
    return tuple((*body, EOS_TOKEN))


def _random_invalid_symbols(rng: random.Random, *, max_length: int, prompt_name: str) -> tuple[str, ...]:
    """Sequence that violates exactly one rule, drawn uniformly across modes."""
    mode = rng.choice(("two_B", "A_and_C", "forbidden_D", "after_end"))
    spec = prompt_spec(prompt_name)
    preferred = spec["preferred_symbols"]
    if mode == "two_B":
        body_length = _bounded_body_length(rng, max_length=max_length, low=3, high=5)
        body = [preferred[0]] * body_length
        # Drop two B's at distinct positions.
        first = rng.randrange(body_length)
        second = (first + 1) % body_length
        body[first] = "B"
        body[second] = "B"
        return tuple((*body, EOS_TOKEN))
    if mode == "A_and_C":
        body_length = _bounded_body_length(rng, max_length=max_length, low=2, high=4)
        # Always contain at least one A and one C -- violates not(existsA and existsC).
        body = ["A" if index == 0 else "C" if index == 1 else preferred[0] for index in range(body_length)]
        return tuple((*body, EOS_TOKEN))
    if mode == "forbidden_D":
        body_length = _bounded_body_length(rng, max_length=max_length, low=2, high=4)
        body = [preferred[0]] * body_length
        body[rng.randrange(body_length)] = "D"
        return tuple((*body, EOS_TOKEN))
    # after_end: a token follows END within the same sequence.
    body_length = _bounded_body_length(rng, max_length=max_length, low=2, high=3)
    body = [preferred[0]] * body_length
    return tuple((*body, EOS_TOKEN, preferred[0]))


def mock_generator_stream(
    *,
    count: int,
    seed: int = 0,
    max_length: int = 6,
) -> Iterator[tuple[str, str, tuple[str, ...]]]:
    """Yield deterministic generator proposals interleaving valid and invalid."""
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
    """Read-only stream-batch provider used by the PMD program."""

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
        return stream_training_examples(
            self.bundle,
            count=self.stream_count,
            seed=self.seed + int(step),
            max_length=self.max_length,
        )

    def training_data(self, batch: Sequence[StreamTrainingExample]) -> list[dict]:
        return [example.sample_data for example in batch]
