"""Small HuggingFace-shaped tokenizer/model used by the offline demo."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MockOutput:
    """Minimal object matching the HuggingFace ``.logits`` output shape."""

    logits: torch.Tensor


class MockTokenizer:
    """Tiny tokenizer with one id per demo token.

    The generation vocabulary uses strings with a leading space to mimic common
    GPT-style token surfaces, while prompts are encoded as regular words.
    Unknown prompt words map to ``<prompt>`` so the demo remains offline.
    """

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

    def encode(self, text, return_tensors=None):
        if text in self.token_to_id:
            ids = [self.token_to_id[text]]
        else:
            words = text.split()
            ids = [self.token_to_id.get(f" {word}", self.token_to_id["<prompt>"]) for word in words]
            if not ids:
                ids = [self.token_to_id["<prompt>"]]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(self, text, return_tensors=None):
        input_ids = self.encode(text, return_tensors=return_tensors)
        if return_tensors == "pt":
            return type("MockBatch", (), {"input_ids": input_ids})()
        return {"input_ids": input_ids}

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.id_to_token.get(int(token_id), "<unk>") for token_id in token_ids)


class MockCausalLM:
    """Deterministic causal LM with HuggingFace-compatible call semantics.

    The raw model prefers ``" dog"`` early, but graph constraints forbid that
    token and require ``" cat"``.  This makes the DFA masking visible across
    greedy, beam, and sampling runs.
    """

    def __init__(self, vocab_size: int = 6):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        logits = torch.full(
            (1, input_ids.shape[1], self.vocab_size),
            -8.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        generated = [int(token_id) for token_id in input_ids[0].tolist() if int(token_id) != 5]

        if not generated:
            logits[0, -1, 4] = 9.0   # dog: forbidden by graph constraint
            logits[0, -1, 1] = 7.0   # The
            logits[0, -1, 2] = 6.0   # cat
            logits[0, -1, 3] = 1.0   # mat
        elif 2 not in generated:
            logits[0, -1, 4] = 9.0
            logits[0, -1, 2] = 8.0
            logits[0, -1, 3] = 3.0
            logits[0, -1, 0] = 0.5
        else:
            logits[0, -1, 3] = 6.0
            logits[0, -1, 0] = 5.0
            logits[0, -1, 1] = 1.0

        return MockOutput(logits=logits)


class MockFrozenBackbone(torch.nn.Module):
    """Tiny frozen feature source for the learning demo."""

    def __init__(self, vocab_size: int = 6, hidden_size: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embedding.weight.zero_()
            for token_id in range(vocab_size):
                self.embedding.weight[token_id, token_id % hidden_size] = 1.0
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids):
        return self.embedding(input_ids.long())
