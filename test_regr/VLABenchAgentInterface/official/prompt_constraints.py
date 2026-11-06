"""The knowledge levels' constraints in plain English, for a prompt-only frozen-VLM baseline.

``--prompt-constraints`` tells the model the rules instead of enforcing them: there is no DFA, no
perception call and no extra question, so the model may ignore a rule and is scored on what it writes.
The logical constraints stay the single declaration of the rules. Each rule below names the ones it
words, and ``test_prompt_constraints.py`` fails if the declarations and this text drift apart.

``ANSWER_GATED_LCS`` are left out: they fire only on an answer to ``--press-question``, ``--scene-ids``
or ``--pattern-first``, which are off by default and need extra model calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

LEVELS = ("grammar", "causal", "official")
DEFAULT_LEVEL = "official"
ANSWER_GATED_LCS = ("press_answer_single_press", "scene_ids_bound_targets",
                    "chosen_patterns_continue", "chosen_patterns_complete")


@dataclass(frozen=True)
class PromptRule:
    text: str
    lcs: tuple[str, ...] = ()  # declared names; the end-of-plan and budget rules are unnamed DFA constraints


GRAMMAR_RULES = (
    PromptRule("Finish the plan once it is complete, and write nothing after it."),
    PromptRule("Use at most {max_operations} skill calls in total."),
    PromptRule("Start the plan with the first skill of one of the patterns listed in the rule below.",
               ("plan_progress_positions",)),
    PromptRule("Give each skill exactly these parameters, each with one integer object ID, and no others: {arguments}.",
               ("plan_operation_arguments",)),
    # Selected on the validation split: a plain list of patterns shortened multi-object plans (Complex fell 14 points).
    PromptRule("Build the plan as a sequence of subtasks, with one subtask for each object the instruction involves. "
               "Each subtask follows one of these patterns, step by step: {patterns}. For example, moving two objects "
               "is pick, place, pick, place.",
               ("plan_pattern_steps",)),
    PromptRule("Never finish the plan or start another pattern in the middle of a pattern. A pick on its own is "
               "not a plan, and a press cannot follow a pick.",
               ("plan_only_completed_subtasks_end",)),
)
CAUSAL_RULES = (
    PromptRule("Never use the object you picked as the destination of place, insert, pour or push. "
               "For example, pick 2 followed by place 2 is not allowed.",
               ("anti_self_placement",)),
    PromptRule("Within one subtask, use a destination only once, even with different skills. A later subtask may "
               "use it again. For example, pick 2, pour 3, place 3 is not allowed.",
               ("distinct_subtask_destinations",)),
    PromptRule("Never pick an object that was an earlier destination: it stays where it is for the rest of the "
               "plan. For example, pick 2, place 3, pick 3 is not allowed.",
               ("destinations_stay_put",)),
)
OFFICIAL_RULES = (
    PromptRule("Use only object IDs whose numbered tags you can see in the labeled image. For example, if tags "
               "1, 2 and 3 are visible, pick 9 is not allowed.",
               ("observation_supported_references",)),
    PromptRule("When the instruction names an object or a destination, use only the tag of something that matches "
               "it. For example, if the apple is tag 2 and the instruction says to pick the apple, pick 5 is not "
               "allowed.",
               ("instruction_candidates_bound_targets",)),
    PromptRule("When the instruction asks for the k-th object of a group in some order, choose the group member "
               "with exactly k - 1 group members before it in that order. For example, for the 2nd poker from "
               "left to right among tags 4, 7 and 9, choose tag 7.",
               tuple(f"ordinal_rank_{k}" for k in range(1, 9))),
    PromptRule("When the instruction names a card, choose the tag that shows that card face. For example, for the "
               "ace of spades, choose only the tag showing that card.",
               ("named_card_face",)),
)
_SECTIONS = {"grammar": ("Plan structure", GRAMMAR_RULES),
             "causal": ("Causal rules", CAUSAL_RULES),
             "official": ("Rules about the images and the instruction", OFFICIAL_RULES)}


def prompt_rules(level: str = DEFAULT_LEVEL) -> tuple[PromptRule, ...]:
    """The rules of a level. Each level keeps the ones below it, as ``--knowledge`` does."""
    if level not in LEVELS:
        raise ValueError(f"unknown prompt constraint level {level!r}; expected one of {LEVELS}")
    return tuple(rule for name in LEVELS[:LEVELS.index(level) + 1] for rule in _SECTIONS[name][1])


def _arguments_text(arguments) -> str:
    return "; ".join(f"{skill} takes {' and '.join(roles)}" if roles else f"{skill} takes no parameters"
                     for skill, roles in arguments.items())


def prompt_constraints_text(level: str = DEFAULT_LEVEL, *, grammar_profile: str, max_operations: int) -> str:
    """The rules as prompt text, worded for the skill language of ``grammar_profile``."""
    try:
        from ..native_planning import native_schema
    except ImportError:  # pragma: no cover - flat-module execution
        from native_planning import native_schema
    if max_operations <= 0:
        raise ValueError("--max-operations must be positive")
    arguments, patterns = native_schema(grammar_profile)
    values = {"max_operations": max_operations, "arguments": _arguments_text(arguments),
              "patterns": ", ".join(json.dumps(list(pattern)) for pattern in patterns)}
    prompt_rules(level)  # rejects an unknown level before anything is worded
    lines, number = ["### Plan constraints:", "", "The skill sequence you generate must follow all of these rules."], 0
    for name in LEVELS[:LEVELS.index(level) + 1]:
        title, rules = _SECTIONS[name]
        lines += ["", f"{title}:"]
        for rule in rules:
            number += 1
            lines.append(f"{number}. {rule.text.format(**values)}")
    return "\n".join(lines)


def prompt_constraints_record(level: str, text: str) -> dict[str, Any]:
    """What a report keeps: the exact wording decides the result, so it is stored with its hash."""
    return {"level": level, "text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
