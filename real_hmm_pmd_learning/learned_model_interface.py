"""Interfaces for scoring and greedy inference with the learned compact model."""
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


def padded_sequence_labels(artifacts) -> torch.Tensor:
    """Return padded compact labels for the first streamed training example."""
    if not artifacts.stream_examples:
        raise ValueError("artifacts.stream_examples is empty")
    labels = artifacts.stream_examples[0].sample_data["sequence_labels_input"][0].tolist()[: artifacts.model.pad_size]
    pad_label = artifacts.bundle.vocabulary.other_label
    if len(labels) < artifacts.model.pad_size:
        labels.extend([pad_label] * (artifacts.model.pad_size - len(labels)))
    return torch.tensor(labels, dtype=torch.long)


def predictions_for_sample(artifacts) -> dict[str, object]:
    """Return current teacher-forced argmax predictions for the first stream item."""
    if not artifacts.stream_examples:
        raise ValueError("artifacts.stream_examples is empty")
    labels = padded_sequence_labels(artifacts)
    log_probs = artifacts.model(None, artifacts.stream_examples[0].sample_data["instruction_tokens"], labels)
    pred_labels = log_probs.argmax(dim=-1).detach().cpu().tolist()
    return {
        "generator_label": artifacts.stream_examples[0].name,
        "prompt_name": artifacts.stream_examples[0].prompt_name,
        "prompt_text": artifacts.stream_examples[0].prompt_text,
        "sequence_labels": labels.tolist(),
        "pred_labels": pred_labels,
        "pred_symbols": symbols_for_labels(artifacts.bundle, pred_labels),
    }


def learned_model_greedy_search(artifacts) -> LearnedInferenceResult:
    """Run unconstrained greedy search from the learned compact-label model."""
    if hasattr(artifacts.model, "greedy_label_inference"):
        return artifacts.model.greedy_label_inference(
            artifacts.bundle.vocabulary,
            inference_prompt_tokens(artifacts).reshape(-1).tolist(),
            max_new_tokens=artifacts.model.pad_size,
        )
    return greedy_label_inference(
        artifacts.model,
        artifacts.bundle.vocabulary,
        inference_prompt_tokens(artifacts).reshape(-1).tolist(),
        max_new_tokens=artifacts.model.pad_size,
    )
