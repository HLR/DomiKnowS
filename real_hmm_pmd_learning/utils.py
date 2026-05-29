"""Small utility helpers for the one-constraint PMD learning demo."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

import torch

from domiknows.generation import constrained_label_greedy_decode

try:
    from .learned_model_interface import (
        learned_model_greedy_search,
        predictions_for_sample,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from learned_model_interface import (
        learned_model_greedy_search,
        predictions_for_sample,
    )


_OPTIMIZER_GRAD_SNAPSHOT: dict[str, object] | None = None


def reset_optimizer_grad_snapshot() -> None:
    """Clear stored optimizer gradient diagnostics before a train call."""
    global _OPTIMIZER_GRAD_SNAPSHOT
    _OPTIMIZER_GRAD_SNAPSHOT = None


def _capture_optimizer_grad_snapshot(param_groups) -> dict[str, object]:
    """Capture gradient norms from optimizer parameter groups before update."""
    total_sq_norm = 0.0
    grad_param_count = 0
    tracked_param_count = 0
    max_grad_norm = 0.0
    max_param_id = None
    for group in param_groups:
        for param in group.get("params", []):
            tracked_param_count += 1
            if param.grad is None:
                continue
            grad_param_count += 1
            grad_norm = float(param.grad.detach().norm().item())
            total_sq_norm += grad_norm * grad_norm
            if grad_norm > max_grad_norm:
                max_grad_norm = grad_norm
                max_param_id = id(param)
    return {
        "grad_param_count": grad_param_count,
        "tracked_param_count": tracked_param_count,
        "total_l2": total_sq_norm ** 0.5,
        "max_l2": max_grad_norm,
        "max_param_id": max_param_id,
    }


class AdamWithGradSnapshot(torch.optim.Adam):
    """Adam optimizer that stores pre-step gradient diagnostics for printing."""

    def step(self, closure=None):
        global _OPTIMIZER_GRAD_SNAPSHOT
        _OPTIMIZER_GRAD_SNAPSHOT = _capture_optimizer_grad_snapshot(self.param_groups)
        return super().step(closure=closure)


def get_optimizer_grad_snapshot() -> dict[str, object] | None:
    """Return the latest optimizer gradient diagnostics if available."""
    if _OPTIMIZER_GRAD_SNAPSHOT is None:
        return None
    return dict(_OPTIMIZER_GRAD_SNAPSHOT)


def _enable_domiknows_production_logging() -> None:
    """Quiet DomiKnowS diagnostic loggers before the demo stack is imported."""
    try:
        from domiknows.utils import setProductionLogMode
    except ImportError:  # pragma: no cover - direct script execution fallback
        return
    setProductionLogMode(no_UseTimeLog=True)


def _enable_remote_debug(host: str, port: int, *, wait: bool = False) -> None:
    """Start optional debugpy remote debugging for local demo inspection."""
    try:
        import debugpy
    except ImportError as exc:
        raise RuntimeError("Install debugpy to use --remote-debug") from exc

    debugpy.listen((host, int(port)))
    print(f"Remote debugger listening on {host}:{port}")
    if wait:
        print("Waiting for debugger client to attach...")
        debugpy.wait_for_client()


def print_greedy_inference(artifacts, result=None) -> None:
    """Print greedy inference from the active learned compact-label model."""
    if result is None:
        result = learned_model_greedy_search(artifacts)
    score = "None" if result.score is None else f"{result.score:.4f}"
    print(f"Learned {artifacts.learner_name} greedy inference:")
    print("  purpose: let the trained learner generate a sequence")
    print(f"  prompt: {artifacts.inference_prompt_name} ({artifacts.inference_prompt_text})")
    print(f"  labels: {_short_values(result.labels)}")
    print(f"  symbols: {_short_sequence(result.symbols)}")
    print(f"  learner_log_score: {score}")
    if result.score is not None and result.labels:
        avg_log_score = result.score / len(result.labels)
        avg_probability = math.exp(avg_log_score)
        print(
            "  interpretation: higher/less-negative log scores mean the learner considers "
            "the greedy sequence more likely; raw scores get more negative for longer sequences."
        )
        print(
            f"  per_token_view: avg_log_score={avg_log_score:.4f}, "
            f"avg_probability~{avg_probability:.4f}"
        )


def constrained_greedy_inference(artifacts):
    """Run DFA-constrained greedy decoding from the learned compact-label model."""
    result = constrained_label_greedy_decode(
        artifacts.model,
        [int(artifacts.inference_prompt_token_id)],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.model.pad_size,
    )
    labels = tuple(int(label) for label in result.labels)
    return SimpleNamespace(
        labels=labels,
        symbols=tuple(artifacts.bundle.vocabulary.token_for_label(label) for label in labels),
        accepted=bool(result.accepted),
        score=None if result.score is None else float(result.score),
        token_ids=tuple(int(token_id) for token_id in result.token_ids),
        scores=tuple(float(score) for score in (result.scores or ())),
    )


def print_constrained_greedy_inference(artifacts, result=None) -> None:
    """Print DFA-constrained greedy decoding from the active learned model."""
    if result is None:
        result = constrained_greedy_inference(artifacts)
    labels = tuple(int(label) for label in result.labels)
    symbols = getattr(result, "symbols", None)
    if symbols is None:
        symbols = tuple(artifacts.bundle.vocabulary.token_for_label(label) for label in labels)
    score = "None" if result.score is None else f"{result.score:.4f}"
    print("DFA-constrained greedy inference:")
    print("  purpose: run the trained learner through the graph-discovered DFA decoder")
    print("  decoder_call: constrained_label_greedy_decode(...)")
    print(f"  prompt: {artifacts.inference_prompt_name} ({artifacts.inference_prompt_text})")
    print("  DFA note: the decoder masks illegal next labels while generating")
    print(f"  labels: {_short_values(labels)}")
    print(f"  symbols: {_short_sequence(symbols)}")
    print(f"  dfa_accepted: {result.accepted}")
    print(f"  learner_log_score: {score}")


def print_hybrid_controller_ranking(ranked) -> None:
    """Print HybridController reranking results for inference candidates."""
    print("HybridController reranking:")
    print("  purpose: verify and score inference candidates with the same DFA and compact learner")
    print("  controller_call: hybrid_controller.generate_verify_rerank(...)")
    for index, item in enumerate(ranked, start=1):
        labels = tuple(int(label) for label in (item.candidate.labels or ()))
        tokens = tuple(item.score.diagnostics.get("tokens", ()))
        print(
            f"  {index}. source={item.candidate.source or 'candidate'} "
            f"accepted={item.score.accepted} total={item.score.total:.4f} "
            f"head_logprob={item.score.head_logprob:.4f} risk={item.score.risk:.4f} "
            f"labels={_short_values(labels)} symbols={_short_sequence(tokens)}"
        )
        if item.score.rejection:
            print(f"     dfa_rejection: {item.score.rejection}")


def print_stream_batch(examples, *, title: str) -> None:
    """Print one materialized generator stream batch."""
    print(title)
    print("  #  prompt       generator_label  sequence  length")
    for index, example in enumerate(examples, start=1):
        print(
            f"  {index}. {example.prompt_name:12s} {example.name:15s} "
            f"sequence={_short_sequence(example.symbols)} "
            f"length={len(example.symbols)}"
        )


def print_demo_header(artifacts) -> None:
    """Print the beginner-facing summary of the demo setup."""
    print("One-constraint DomiKnowS PMD learning demo")
    print("Rule: token B may appear at most once")
    print("Generator stream: prompt-conditioned outputs are used directly for PMD training")
    print("Prompt meanings: AB prefers A/B tokens; CD prefers C/D tokens; short prefers early END")
    print("Padding: unused fixed-width positions use _other, not END, so padding does not teach early stopping")
    print(f"Inference prompt: {artifacts.inference_prompt_name} ({artifacts.inference_prompt_text})")
    print(f"Active compact-label learner: {artifacts.learner_name}")
    print(f"PMD constraint weight beta: {artifacts.program.beta:g}")
    print("Trainable learner parameters:", artifacts.model.trainable_parameter_names())
    print_parameter_explanation(artifacts)


def print_learning_snapshot(artifacts, *, title: str) -> None:
    """Print current predictions and greedy inference."""
    print(title)
    print("Snapshot guide:")
    print("  Predictions      = inspect learner on one generator-produced training sequence")
    print("  Greedy inference = let the learner generate a sequence for the inference prompt")
    print_predictions(artifacts)
    print_greedy_inference(artifacts)


def print_training_header() -> None:
    """Print how live stream batches enter standard DomiKnowS training."""
    print("\nTraining uses PrimalDualProgram.train(...)")
    print("Training batches come from GeneratorTrainingSource.next_batch(step)")


def print_inference_header() -> None:
    """Print the explicit post-training inference step header."""
    print("\nInference after training")
    print("Greedy search reads from the learned compact-label model.")


def print_no_training_requested() -> None:
    """Print the no-op training message."""
    print("  no training batches requested")


def print_gradient_snapshot(model, *, step: int) -> None:
    """Print compact gradient diagnostics for the learner after a train step."""
    optimizer_snapshot = get_optimizer_grad_snapshot()
    if optimizer_snapshot is not None:
        grad_param_count = int(optimizer_snapshot["grad_param_count"])
        tracked_param_count = int(optimizer_snapshot["tracked_param_count"])
        if grad_param_count == 0:
            print(
                f"  gradient snapshot after batch {step + 1}: "
                "no gradients found on optimizer-tracked learner parameters"
            )
            return
        name_lookup = {
            id(param): name
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        max_name = name_lookup.get(optimizer_snapshot["max_param_id"], "unknown_parameter")
        print(
            "  gradient snapshot after batch "
            f"{step + 1}: trainable_with_grad={grad_param_count}/{tracked_param_count}, "
            f"total_l2={float(optimizer_snapshot['total_l2']):.6g}, "
            f"max_l2={float(optimizer_snapshot['max_l2']):.6g} ({max_name})"
        )
        print_gradient_interpretation(
            grad_param_count=grad_param_count,
            trainable_param_count=tracked_param_count,
            max_grad_name=max_name,
        )
        return

    # Fallback for cases where a non-wrapped optimizer is used.
    total_sq_norm = 0.0
    grad_param_count = 0
    trainable_param_count = 0
    max_grad_name = None
    max_grad_norm = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_param_count += 1
        if param.grad is None:
            continue
        grad_param_count += 1
        grad_norm = float(param.grad.detach().norm().item())
        total_sq_norm += grad_norm * grad_norm
        if grad_norm > max_grad_norm:
            max_grad_norm = grad_norm
            max_grad_name = name
    if grad_param_count == 0:
        print(f"  gradient snapshot after batch {step + 1}: no gradients found on trainable learner parameters")
        return
    total_norm = total_sq_norm ** 0.5
    print(
        "  gradient snapshot after batch "
        f"{step + 1}: trainable_with_grad={grad_param_count}/{trainable_param_count}, "
        f"total_l2={total_norm:.6g}, max_l2={max_grad_norm:.6g} ({max_grad_name})"
    )
    print_gradient_interpretation(
        grad_param_count=grad_param_count,
        trainable_param_count=trainable_param_count,
        max_grad_name=max_grad_name,
    )


def capture_parameter_snapshot(model, *, hmm_only: bool = True) -> dict[str, torch.Tensor]:
    """Capture trainable learner parameters for update diagnostics."""
    keywords = ("hmm", "transition", "emission", "initial", "start", "prompt")
    all_trainable = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not hmm_only:
        return all_trainable
    filtered = {
        name: value
        for name, value in all_trainable.items()
        if any(keyword in name.lower() for keyword in keywords)
    }
    return filtered if filtered else all_trainable


def print_parameter_update_snapshot(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    step: int,
    hmm_matched: bool,
) -> None:
    """Print parameter movement diagnostics after optimizer updates."""
    shared_names = [name for name in before if name in after]
    if not shared_names:
        print(f"  parameter update after batch {step + 1}: no shared trainable parameters to compare")
        return

    total_sq_delta = 0.0
    changed_count = 0
    max_delta_name = None
    max_delta_norm = 0.0

    for name in shared_names:
        delta_norm = float((after[name] - before[name]).norm().item())
        total_sq_delta += delta_norm * delta_norm
        if delta_norm > 0.0:
            changed_count += 1
        if delta_norm > max_delta_norm:
            max_delta_norm = delta_norm
            max_delta_name = name

    total_delta = total_sq_delta ** 0.5
    scope = "hmm" if hmm_matched else "all-trainable-fallback"
    print(
        "  parameter update after batch "
        f"{step + 1}: scope={scope}, changed={changed_count}/{len(shared_names)}, "
        f"total_l2_delta={total_delta:.6g}, max_l2_delta={max_delta_norm:.6g} ({max_delta_name})"
    )
    print_update_interpretation(
        changed_count=changed_count,
        parameter_count=len(shared_names),
        max_delta_name=max_delta_name,
        scope=scope,
    )


def print_trained_batch(step: int, sample_count: int) -> None:
    """Print a stable line after one hidden tqdm training call completes."""
    print(f"  trained on batch {step + 1}: {sample_count} generated samples")


def print_gradient_interpretation(*, grad_param_count: int, trainable_param_count: int, max_grad_name: str | None) -> None:
    """Explain the gradient snapshot in plain language."""
    coverage = "all" if grad_param_count == trainable_param_count else "some"
    print(
        f"    interpretation: {coverage} trainable learner parameters received a learning signal; "
        "total_l2 is the overall pre-update gradient size."
    )
    print(
        f"    strongest_signal: {max_grad_name} had the largest gradient, so this batch most directly pushed that parameter group."
    )


def print_update_interpretation(*, changed_count: int, parameter_count: int, max_delta_name: str | None, scope: str) -> None:
    """Explain the parameter update snapshot in plain language."""
    coverage = "all" if changed_count == parameter_count else "some"
    print(
        f"    interpretation: {coverage} tracked {scope} parameters moved after the optimizer step; "
        "total_l2_delta is actual parameter movement, not the loss."
    )
    print(
        f"    largest_move: {max_delta_name} changed the most; with Adam this can be larger or smaller than the raw gradient scale."
    )


def print_parameter_explanation(artifacts) -> None:
    """Print a compact explanation of what the active learner parameters mean."""
    print("Parameter meaning:")
    if artifacts.learner_name == "discrete-hmm":
        print("  initial_logits: learns which hidden state a generated string starts in.")
        print("  transition_logits: learns how hidden states move from one position to the next.")
        print("  emission_logits: learns which symbols each hidden state tends to emit.")
        print("  hidden-state example: one state can mean 'B has not appeared yet'; another can mean 'B already appeared'.")
        print("  emission example: the 'B already appeared' state should learn low probability for emitting another B.")
        print("  note: this is the plain DiscreteHMM-backed learner; graph-hmm adds graph-shaped initialization and prompt conditioning.")
        return
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
        print("  example: for prompt AB, the model can learn lower energy for A/B than for C/D.")
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
