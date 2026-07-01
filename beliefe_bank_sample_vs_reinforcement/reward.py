from __future__ import annotations

from typing import Any

import torch

from domiknows.reinforcement.rewards import binary_label_name, flatten_generator_output


def _label(value: Any) -> str:
    # BeliefBank stores binary fact states as domain labels; the yes/no vocabulary
    # itself is shared by domiknows.reinforcement.rewards.
    return binary_label_name(value, true_label="yes", false_label="no", default="no")


def _predictions(generator_output: Any, facts: list[str]) -> dict[str, str]:
    # The decoder may return a fact->label dict or a flat list aligned with facts.
    if isinstance(generator_output, dict):
        for key in ("predictions", "prediction", "output", "answer"):
            if key in generator_output:
                return _predictions(generator_output[key], facts)
        return {fact: _label(generator_output.get(fact, "no")) for fact in facts}

    values = flatten_generator_output(generator_output)

    return {
        fact: _label(values[idx]) if idx < len(values) else "no"
        for idx, fact in enumerate(facts)
    }


def _edge_satisfaction(preds: dict[str, str], edges: list[tuple[str, str]], target_label: str) -> float:
    # Implications are only violated when the source is yes and the target has
    # the wrong label; inactive sources satisfy the implication vacuously.
    if not edges:
        return 1.0
    satisfied = 0
    for source, target in edges:
        source_yes = preds.get(source, "no") == "yes"
        target_ok = preds.get(target, "no") == target_label
        satisfied += int((not source_yes) or target_ok)
    return satisfied / len(edges)


def reward_from_belief_state(
    generator_output: Any,
    facts: list[str],
    gold_labels: dict[str, str],
    positive_edges: list[tuple[str, str]],
    negative_edges: list[tuple[str, str]],
) -> torch.Tensor:
    """Dense BeliefBank reward over labels and implication consistency."""
    preds = _predictions(generator_output, facts)

    if facts:
        label_accuracy = sum(
            int(preds.get(fact, "no") == _label(gold_labels.get(fact, "no")))
            for fact in facts
        ) / len(facts)
    else:
        label_accuracy = 1.0

    positive_satisfaction = _edge_satisfaction(preds, positive_edges, "yes")
    negative_satisfaction = _edge_satisfaction(preds, negative_edges, "no")

    # This calibration term discourages degenerate all-yes or all-no policies
    # without making exact fact matching the only source of reward.
    predicted_yes = sum(1 for fact in facts if preds.get(fact, "no") == "yes")
    gold_yes = sum(1 for fact in facts if _label(gold_labels.get(fact, "no")) == "yes")
    yes_count_calibration = 1.0 - abs(predicted_yes - gold_yes) / max(1, len(facts))

    reward = (
        0.40 * label_accuracy
        + 0.25 * positive_satisfaction
        + 0.25 * negative_satisfaction
        + 0.10 * yes_count_calibration
    )
    return torch.tensor([max(0.0, min(1.0, reward))], dtype=torch.float32)


def make_beliefbank_reward_function(
    subject_name: str,
    facts: list[str],
    labels: list[str],
    positive_edges: list[tuple[str, str]],
    negative_edges: list[tuple[str, str]],
):
    """
        It pairs items from facts and labels position-by-position using zip.
        For each pair, it normalizes the label through _label (to yes or no).
        It stores them in a dictionary:
        fact -> normalized gold label
        Example:

        facts = [f1, f2, f3]
        labels = [yes, no, yes]
        result = {f1: yes, f2: no, f3: yes}
    """
    gold_labels = {fact: _label(label) for fact, label in zip(facts, labels)}

    """Create a per-item reward closure with metadata for inspection."""
    
    # ReinforcementProgram calls this closure with only generator_output unless
    # the user-defined reward asks for optional context keywords.
    """ The inner function _reward is created inside make_beliefbank_reward_function.
        That inner function is a closure because it remembers values from the outer function, 
        even after the outer function has finished.
    """
    def _reward(generator_output: Any) -> torch.Tensor:
        return reward_from_belief_state(
            generator_output,
            facts,
            gold_labels,
            positive_edges,
            negative_edges,
        )

    ''' Attach metadata to the reward function for inspection. Another part of the pipeline can inspect these fields later (for logging, analysis, tracing per-item behavior) '''
    _reward.subject_name = subject_name      # stores which sample/item this reward function belongs to.
    _reward.facts = facts                    # stores the fact list used for scoring.
    _reward.gold_labels = gold_labels        # stores the gold truth labels (fact -> yes/no)..
    _reward.positive_edges = positive_edges  # stores implication edges expected to end in yes.
    _reward.negative_edges = negative_edges  # stores implication edges expected to end in no.
    return _reward
