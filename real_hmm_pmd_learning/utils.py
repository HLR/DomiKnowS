"""Small utility helpers for the one-constraint PMD learning demo."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from domiknows.generation import constrained_label_greedy_decode, explain_dfa_rejection

try:
    from .graph import CANDIDATES
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph import CANDIDATES


@dataclass(frozen=True)
class LearnedInferenceResult:
    """Greedy learned-learner inference result with decoded symbols."""

    labels: tuple[int, ...]
    symbols: tuple[str, ...]
    accepted: bool
    score: float | None


def candidate_map() -> dict[str, tuple[str, ...]]:
    """Return the tiny mock-generator candidates used by the demo."""
    return CANDIDATES


def labels_for_symbols(bundle, symbols: Sequence[str]) -> list[int]:
    """Encode surface symbols as compact generation labels."""
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def symbols_for_labels(bundle, labels: Sequence[int]) -> list[str]:
    """Decode compact labels as surface symbols."""
    return [bundle.vocabulary.token_for_label(label) for label in labels]


def padded_sequence_labels(artifacts) -> torch.Tensor:
    """Return padded compact labels for the first streamed training example."""
    if not artifacts.stream_examples:
        raise ValueError("artifacts.stream_examples is empty")
    labels = artifacts.stream_examples[0].sample_data["sequence_labels_input"][0].tolist()[: artifacts.model.pad_size]
    eos_label = artifacts.bundle.vocabulary.eos_label
    if len(labels) < artifacts.model.pad_size:
        labels.extend([eos_label] * (artifacts.model.pad_size - len(labels)))
    return torch.tensor(labels, dtype=torch.long)


def score_candidate_with_learner(artifacts, symbols: Sequence[str]) -> dict[str, object]:
    """Score one candidate with the active compact-label learner and verify it with the DFA."""
    labels = labels_for_symbols(artifacts.bundle, symbols)
    padded = labels[: artifacts.model.pad_size]
    if len(padded) < artifacts.model.pad_size:
        padded.extend([artifacts.bundle.vocabulary.eos_label] * (artifacts.model.pad_size - len(padded)))
    length = min(len(labels), artifacts.model.pad_size)
    log_probs = artifacts.model.sequence_log_probs(
        torch.tensor(padded, dtype=torch.long),
        lengths=torch.tensor([length]),
        instruction_tokens=inference_prompt_tokens(artifacts),
    )
    gold = torch.tensor(padded, dtype=torch.long)
    score = log_probs[:length].gather(1, gold[:length].unsqueeze(1)).sum()
    accepted = artifacts.dfa.accepts(labels)
    rejection = None if accepted else explain_dfa_rejection(artifacts.dfa, labels)
    return {
        "prompt_name": artifacts.inference_prompt_name,
        "symbols": tuple(symbols),
        "labels": labels,
        "score": float(score.detach().cpu().item()),
        "accepted": bool(accepted),
        "rejection": rejection,
    }


score_candidate_with_head = score_candidate_with_learner


def print_candidate_scores(artifacts) -> None:
    """Print all fixed diagnostic candidate scores."""
    print(f"Candidate scores for prompt={artifacts.inference_prompt_name!r}:")
    scored_candidates = [
        (name, score_candidate_with_learner(artifacts, symbols))
        for name, symbols in candidate_map().items()
    ]
    best_score = max(score["score"] for _name, score in scored_candidates)
    for name, score in scored_candidates:
        status = "accepted" if score["accepted"] else "rejected"
        preference = math.exp(float(score["score"]) - float(best_score))
        print(
            f"  {name:7s} {status:8s} "
            f"learner_log_score={score['score']:.4f} "
            f"relative_preference={preference:.2f}x "
            f"sequence={' '.join(score['symbols'])}"
        )
        if score["rejection"]:
            print(f"    dfa_rejection: {score['rejection']}")


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


def _inference_result(artifacts, result) -> LearnedInferenceResult:
    labels = tuple(int(label) for label in result.labels)
    return LearnedInferenceResult(
        labels=labels,
        symbols=tuple(symbols_for_labels(artifacts.bundle, labels)),
        accepted=bool(result.accepted),
        score=None if result.score is None else float(result.score),
    )


def constrained_greedy_inference(artifacts) -> LearnedInferenceResult:
    """Run DFA-constrained greedy inference from the learned compact-label learner."""
    result = constrained_label_greedy_decode(
        artifacts.model,
        inference_prompt_tokens(artifacts).reshape(-1),
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.model.pad_size,
    )
    return _inference_result(artifacts, result)


def print_greedy_inference(artifacts) -> None:
    """Print DFA-constrained greedy inference from the active learner."""
    result = constrained_greedy_inference(artifacts)
    score = "None" if result.score is None else f"{result.score:.4f}"
    print(f"Learned {artifacts.learner_name} greedy inference:")
    print("  purpose: let the learner generate a DFA-constrained sequence")
    print(f"  prompt: {artifacts.inference_prompt_name} ({artifacts.inference_prompt_text})")
    print("  DFA note: the decoder masks illegal next labels while generating")
    print(f"  labels: {_short_values(result.labels)}")
    print(f"  symbols: {_short_sequence(result.symbols)}")
    print(f"  learner_log_score: {score}")


def print_stream_batch(examples, *, title: str) -> None:
    """Print one materialized generator stream batch."""
    print(title)
    print("  This table is the DFA pre-check: generator proposals are verified before PMD training.")
    print("  #  prompt       generator_label  dfa_verdict  sequence  length")
    for index, example in enumerate(examples, start=1):
        status = "accepted" if example.accepted else "rejected"
        print(
            f"  {index}. {example.prompt_name:12s} {example.name:15s} {status:11s} "
            f"sequence={_short_sequence(example.symbols)} "
            f"length={len(example.symbols)}"
        )
        if example.rejection:
            print(f"     dfa_rejection: {example.rejection}")


def print_demo_header(artifacts) -> None:
    """Print the beginner-facing summary of the demo setup."""
    print("One-constraint DomiKnowS PMD learning demo")
    print("Rule: token B may appear at most once")
    print("Generator stream: prompt-conditioned outputs are used for PMD training, including DFA-invalid ones")
    print("Prompt meanings: AB prefers A/B tokens; CD prefers C/D tokens; short prefers early END")
    print(f"Inference prompt: {artifacts.inference_prompt_name} ({artifacts.inference_prompt_text})")
    print(f"Active compact-label learner: {artifacts.learner_name}")
    print("Trainable learner parameters:", artifacts.model.trainable_parameter_names())
    print_parameter_explanation(artifacts)


def print_learning_snapshot(artifacts, *, title: str) -> None:
    """Print current predictions, diagnostic scores, and constrained inference."""
    print(title)
    print("Snapshot guide:")
    print("  Predictions      = inspect learner on one generator-produced training sequence")
    print("  Candidate scores = score/rerank fixed diagnostic candidates for the inference prompt")
    print("  Greedy inference = let the learner generate a DFA-constrained sequence for the inference prompt")
    print_predictions(artifacts)
    print_candidate_scores(artifacts)
    print_greedy_inference(artifacts)


def print_training_header() -> None:
    """Print how live stream batches enter standard DomiKnowS training."""
    print("\nTraining uses PrimalDualProgram.train(...)")
    print("Training batches come from GeneratorTrainingSource.next_batch(step)")


def print_no_training_requested() -> None:
    """Print the no-op training message."""
    print("  no training batches requested")


def print_trained_batch(step: int, sample_count: int) -> None:
    """Print a stable line after one hidden tqdm training call completes."""
    print(f"  trained on batch {step + 1}: {sample_count} generated samples")


def print_parameter_explanation(artifacts) -> None:
    """Print a compact explanation of what the active learner parameters mean."""
    print("Parameter meaning:")
    if artifacts.learner_name == "graph-hmm":
        print("  prompt_embedding / prompt_initial_projector: learn how the prompt changes the initial hidden-state belief.")
        print("  initial_logits: learns which hidden state a generated string starts in.")
        print("  transition_logits: learns how hidden states move from one position to the next.")
        print("  emission_logits: learns which symbols each hidden state tends to emit.")
        print("  hidden-state example: one state can mean 'B has not appeared yet'; another can mean 'B already appeared'.")
        print("  emission example: the 'B already appeared' state should learn low probability for emitting another B.")
        print("  transition example: after emitting B, the learner can move into a state that avoids future B symbols.")
        return
    if artifacts.learner_name == "energy":
        print("  prompt_embedding: learns a vector for the prompt/context token.")
        print("  label_embedding: learns vectors for A, B, C, D, END, and prefix padding.")
        print("  energy_mlp: learns a compatibility cost for prefix + candidate next symbol.")
        print("  example: for prompt AB, the model can learn lower energy for A/B than for C/D, while DFA blocks a second B.")
        return
    print("  This learner exposes trainable compact-label parameters through ModuleLearner.")


def print_predictions(artifacts) -> None:
    """Print current teacher-forced predictions without dumping long padding."""
    predictions = predictions_for_sample(artifacts)
    print("Predictions:")
    print(f"  prompt: {predictions['prompt_name']} ({predictions['prompt_text']})")
    print(f"  generator_label: {predictions['generator_label']}")
    print(f"  sequence_labels: {_short_values(predictions['sequence_labels'])}")
    print(f"  pred_labels:   {_short_values(predictions['pred_labels'])}")
    print(f"  pred_symbols:  {_short_sequence(predictions['pred_symbols'])}")


def _short_values(values: Sequence[int], *, limit: int = 16) -> str:
    values = list(values)
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]} ... (+{len(values) - limit} more)"


def _short_sequence(symbols: Sequence[str], *, limit: int = 16) -> str:
    if len(symbols) <= limit:
        return " ".join(symbols)
    head = " ".join(symbols[:limit])
    return f"{head} ... (+{len(symbols) - limit} more)"


def inference_prompt_tokens(artifacts) -> torch.Tensor:
    """Return the prompt token tensor used for learned inference and candidate scoring."""
    return torch.tensor([[artifacts.inference_prompt_token_id]], dtype=torch.long)
