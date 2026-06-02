"""Greedy + DFA-constrained inference helpers for the demo's learned head."""
from __future__ import annotations

from typing import Sequence

import torch

from domiknows.generation import ConstrainedGenerationResult, greedy_label_inference


LearnedInferenceResult = ConstrainedGenerationResult


def labels_for_symbols(bundle, symbols: Sequence[str]) -> list[int]:
    """Encode surface symbols as compact generation labels."""
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def symbols_for_labels(bundle, labels: Sequence[int]) -> list[str]:
    """Decode compact labels as surface symbols."""
    return [bundle.vocabulary.token_for_label(label) for label in labels]


def inference_prompt_tokens(artifacts) -> torch.Tensor:
    """Return the prompt token tensor used for learned inference."""
    return torch.tensor([[artifacts.inference_prompt_token_id]], dtype=torch.long)


def learned_model_greedy_search(artifacts) -> LearnedInferenceResult:
    """Run unconstrained greedy search from the learned compact-label model."""
    prompt_ids = inference_prompt_tokens(artifacts).reshape(-1).tolist()
    if hasattr(artifacts.model, "greedy_label_inference"):
        return artifacts.model.greedy_label_inference(
            artifacts.bundle.vocabulary,
            prompt_ids,
            max_new_tokens=artifacts.model.pad_size,
        )
    return greedy_label_inference(
        artifacts.model,
        artifacts.bundle.vocabulary,
        prompt_ids,
        max_new_tokens=artifacts.model.pad_size,
    )
