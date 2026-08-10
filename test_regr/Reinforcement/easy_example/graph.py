from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import sys

import torch

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[2]
for path in (RUN_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from domiknows.reinforcement.rewards import count_reward


def _build_reward_function(args, expected_value: Any) -> Callable[[Any], torch.Tensor]:
    expected_label = "zero" if args.expected_value == 0 else "one"

    def _reward(generator_output: Any) -> torch.Tensor:
        # Convert the easy example's constraint-style CLI options into the
        # generic count reward used by ReinforcementProgram.
        if args.atLeastL and args.atMostL:
            lower = count_reward(
                generator_output, expected_label, args.expected_atLeastL, mode="at_least"
            ).item()
            upper = count_reward(
                generator_output, expected_label, args.expected_atMostL, mode="at_most"
            ).item()
            return torch.tensor([1.0 if lower and upper else 0.0], dtype=torch.float32)
        if args.atMostL:
            return count_reward(generator_output, expected_label, args.expected_atMostL, mode="at_most")
        if args.atLeastL:
            return count_reward(generator_output, expected_label, args.expected_atLeastL, mode="at_least")
        return count_reward(generator_output, expected_label, args.expected_atLeastL, mode="exact")

    _reward.expected_value = expected_value
    _reward.expected_atLeastL = args.expected_atLeastL
    _reward.expected_atMostL = args.expected_atMostL
    _reward.atLeastL = args.atLeastL
    _reward.atMostL = args.atMostL
    return _reward


def get_graph(args):
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph import EnumConcept
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global_PMD') as graph:
        a = Concept(name='a')
        b = Concept(name='b')
        a_contain_b, = a.contains(b)

        b_answer = b(name='answer_b', ConceptClass=EnumConcept, values=['zero', 'one'])

        expected_zero = b_answer.__getattr__('zero')
        expected_one = b_answer.__getattr__('one')
        expected_value = expected_zero if args.expected_value == 0 else expected_one

        # TODO: use reward_function in the program instead of logical constraints.
        reward_function = _build_reward_function(args, expected_value)

    return graph, a, b, a_contain_b, b_answer, reward_function
