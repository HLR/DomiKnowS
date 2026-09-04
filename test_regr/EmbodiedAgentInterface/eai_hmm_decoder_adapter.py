"""Adapters from EAI Ctrl-G-style HMM files to DomiKnowS HMM decoders.

The EAI HMM training scripts save NumPy arrays using Ctrl-G-style names:

    alpha_exp: hidden-state transition probabilities, shape H x H
    beta:      hidden-state emission log-probabilities, shape H x V
    gamma:     initial hidden-state log-probabilities, shape H

The HuggingFace generation branch in DomiKnowS uses
``domiknows.generation.HMMGenerationHead`` / ``DiscreteHMM`` for the strict
HMM+DFA decoder.  This module converts the saved EAI arrays into those runtime
objects without retraining.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import torch

from domiknows.generation import HMMGenerationHead
from domiknows.generation.learners.hmm.discrete.discreteHMM import DiscreteHMM




@dataclass(frozen=True)
class EAIHMMParameters:
    """Ctrl-G-compatible HMM parameter arrays saved by EAI distillation."""

    alpha_exp: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray

    @property
    def hidden_states(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.beta.shape[1])


def build_hmm_from_label_sequences(
    sequences,
    vocab_size: int,
    *,
    smoothing: float = 0.1,
    emission_smoothing: float = 0.01,
    start_label: int | None = None,
    eos_label: int | None = None,
) -> EAIHMMParameters:
    """Estimate Ctrl-G-style HMM arrays from compact label sequences.

    This keeps the saved file format (`alpha_exp`, `beta`, `gamma`).
    """
    vocab_size = int(vocab_size)
    initial = np.full((vocab_size,), float(smoothing), dtype=np.float64)
    transition = np.full((vocab_size, vocab_size), float(smoothing), dtype=np.float64)
    emission = np.full((vocab_size, vocab_size), float(emission_smoothing), dtype=np.float64)
    emission[np.arange(vocab_size), np.arange(vocab_size)] += 1.0

    for sequence in sequences:
        labels = [int(x.item() if hasattr(x, "item") else x) for x in sequence]
        labels = [label for label in labels if 0 <= label < vocab_size]
        if not labels:
            continue
        if eos_label is not None:
            eos = int(eos_label)
            while len(labels) > 1 and labels[-1] == eos and labels[-2] == eos:
                labels.pop()
        if start_label is not None:
            labels = [int(start_label)] + labels
        initial[labels[0]] += 1.0
        for prev_label, next_label in zip(labels, labels[1:]):
            transition[prev_label, next_label] += 1.0

    alpha_exp = transition / transition.sum(axis=1, keepdims=True)
    beta = _safe_log_probs(emission / emission.sum(axis=1, keepdims=True))
    gamma = _safe_log_probs(initial / initial.sum())
    return EAIHMMParameters(alpha_exp=alpha_exp, beta=beta, gamma=gamma)


@dataclass(frozen=True)
class EAIHMMArtifacts:
    """Loaded EAI HMM arrays and metadata."""

    alpha_exp: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    tokens: tuple[str, ...]
    eos_label: int
    metadata: dict

    @property
    def state_count(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def label_count(self) -> int:
        return int(self.beta.shape[1])


def _as_prob_from_log(log_values: np.ndarray, axis: int = -1) -> np.ndarray:
    log_values = np.asarray(log_values, dtype=np.float64)
    shifted = log_values - np.max(log_values, axis=axis, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=axis, keepdims=True)
    return probs


def _safe_log_probs(probs: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.asarray(probs, dtype=np.float64).clip(min=eps))


def load_eai_ctrlg_hmm(path: str | Path) -> EAIHMMArtifacts:
    """Load an HMM produced by ``train_hmm.py`` or ``train_qwen_hmm.py``."""
    data = np.load(Path(path), allow_pickle=True)
    metadata_raw = data["metadata"].item() if "metadata" in data.files else "{}"
    metadata = json.loads(str(metadata_raw))
    tokens = tuple(str(item) for item in data["tokens"].tolist()) if "tokens" in data.files else ()
    eos_label = int(data["eos_label"].reshape(-1)[0]) if "eos_label" in data.files else int(metadata.get("eos_label", 0))
    artifacts = EAIHMMArtifacts(
        alpha_exp=np.asarray(data["alpha_exp"], dtype=np.float64),
        beta=np.asarray(data["beta"], dtype=np.float64),
        gamma=np.asarray(data["gamma"], dtype=np.float64),
        tokens=tokens,
        eos_label=eos_label,
        metadata=metadata,
    )
    validate_eai_hmm_artifacts(artifacts)
    return artifacts


def validate_eai_hmm_artifacts(artifacts: EAIHMMArtifacts, atol: float = 1e-5) -> None:
    """Validate shapes and row-normalization before decoder conversion."""
    h = artifacts.state_count
    v = artifacts.label_count
    if artifacts.alpha_exp.shape != (h, h):
        raise ValueError(f"alpha_exp must have shape {(h, h)}, got {artifacts.alpha_exp.shape}")
    if artifacts.beta.shape != (h, v):
        raise ValueError(f"beta must have shape {(h, v)}, got {artifacts.beta.shape}")
    if artifacts.gamma.shape != (h,):
        raise ValueError(f"gamma must have shape {(h,)}, got {artifacts.gamma.shape}")
    if artifacts.tokens and len(artifacts.tokens) != v:
        raise ValueError(f"tokens length {len(artifacts.tokens)} does not match label_count {v}")
    if not np.allclose(artifacts.alpha_exp.sum(axis=-1), 1.0, atol=atol):
        raise ValueError("alpha_exp rows are not normalized")
    if not np.allclose(_as_prob_from_log(artifacts.beta).sum(axis=-1), 1.0, atol=atol):
        raise ValueError("beta rows do not normalize as log probabilities")
    if not np.isclose(_as_prob_from_log(artifacts.gamma, axis=0).sum(), 1.0, atol=atol):
        raise ValueError("gamma does not normalize as log probabilities")


def eai_hmm_to_discrete_hmm(
    artifacts: EAIHMMArtifacts,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> DiscreteHMM:
    """Convert loaded EAI HMM arrays to DomiKnowS ``DiscreteHMM``."""
    initial = torch.as_tensor(_as_prob_from_log(artifacts.gamma, axis=0), dtype=dtype, device=device)
    transition = torch.as_tensor(artifacts.alpha_exp, dtype=dtype, device=device)
    emission = torch.as_tensor(_as_prob_from_log(artifacts.beta), dtype=dtype, device=device)
    return DiscreteHMM(
        transition=transition,
        emission=emission,
        initial=initial,
        symbols=tuple(range(artifacts.label_count)),
        normalize=False,
    )


def eai_hmm_to_generation_head(
    artifacts: EAIHMMArtifacts,
    *,
    pad_size: int | None = None,
    label_to_token_id: Sequence[int | None] | None = None,
    trainable: bool = False,
    device: str | torch.device | None = None,
) -> HMMGenerationHead:
    """Convert EAI HMM arrays to ``HMMGenerationHead`` for HMM+DFA decoding.

    For EAI compact-label decoding, ``label_to_token_id`` defaults to identity:
    label ``i`` maps to token id ``i``.  This matches our EAI generators, where
    labels are already compact action/object ids.
    """
    if label_to_token_id is None:
        label_to_token_id = tuple(range(artifacts.label_count))
    head = HMMGenerationHead(
        label_count=artifacts.label_count,
        state_count=artifacts.state_count,
        pad_size=pad_size or int(artifacts.metadata.get("max_steps", artifacts.label_count)),
        label_to_token_id=label_to_token_id,
        trainable=trainable,
    )
    with torch.no_grad():
        head.initial_logits.copy_(torch.as_tensor(artifacts.gamma, dtype=head.initial_logits.dtype))
        head.transition_logits.copy_(torch.as_tensor(_safe_log_probs(artifacts.alpha_exp), dtype=head.transition_logits.dtype))
        head.emission_logits.copy_(torch.as_tensor(artifacts.beta, dtype=head.emission_logits.dtype))
    if device is not None:
        head = head.to(device)
    return head


def load_eai_hmm_generation_head(
    path: str | Path,
    *,
    pad_size: int | None = None,
    label_to_token_id: Sequence[int | None] | None = None,
    trainable: bool = False,
    device: str | torch.device | None = None,
) -> HMMGenerationHead:
    """Load an EAI HMM ``.npz`` directly as ``HMMGenerationHead``."""
    return eai_hmm_to_generation_head(
        load_eai_ctrlg_hmm(path),
        pad_size=pad_size,
        label_to_token_id=label_to_token_id,
        trainable=trainable,
        device=device,
    )


def compare_head_to_artifacts(head: HMMGenerationHead, artifacts: EAIHMMArtifacts, atol: float = 1e-5) -> dict[str, float]:
    """Return max absolute differences between decoder-head probs and source HMM."""
    initial_diff = torch.max(
        torch.abs(head.initial_probs.detach().cpu() - torch.as_tensor(_as_prob_from_log(artifacts.gamma, axis=0), dtype=torch.float32))
    ).item()
    transition_diff = torch.max(
        torch.abs(head.transition_probs.detach().cpu() - torch.as_tensor(artifacts.alpha_exp, dtype=torch.float32))
    ).item()
    emission_diff = torch.max(
        torch.abs(head.emission_probs.detach().cpu() - torch.as_tensor(_as_prob_from_log(artifacts.beta), dtype=torch.float32))
    ).item()
    return {
        "initial_max_abs_diff": float(initial_diff),
        "transition_max_abs_diff": float(transition_diff),
        "emission_max_abs_diff": float(emission_diff),
        "within_tolerance": bool(max(initial_diff, transition_diff, emission_diff) <= atol),
    }
