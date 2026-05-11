"""Readable loss logging helpers for HF generation demos."""
from __future__ import annotations


NON_LOSS_DIAGNOSTIC_KEYS = {"latent_terms", "transition_potentials"}


def print_loss_log_note() -> None:
    """Explain signed PMD objectives before step logs."""
    print(
        "Loss log: pmd_constraint_objective is signed by PrimalDualProgram; "
        "optimization_objective may become negative. "
        "positive_training_terms excludes that signed PMD term."
    )


def format_loss_log(losses: dict[str, float]) -> dict[str, float]:
    """Rename ambiguous loss keys and add a positive-terms summary."""
    result: dict[str, float] = {}
    positive_terms = 0.0
    for key, value in losses.items():
        value = float(value)
        if key == "constraint_loss":
            result["pmd_constraint_objective"] = value
            continue
        if key == "total_loss":
            continue
        result[key] = value
        if key not in NON_LOSS_DIAGNOSTIC_KEYS:
            positive_terms += value
    if any(key not in {"constraint_loss", "total_loss", *NON_LOSS_DIAGNOSTIC_KEYS} for key in losses):
        result["positive_training_terms"] = positive_terms
    if "total_loss" in losses:
        result["optimization_objective"] = float(losses["total_loss"])
    return result

