from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable
import re
import sys

import torch

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[2]
for path in (RUN_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\+]+", "", text)
    return text


def _flatten_generator_output(generator_output: Any) -> list[Any]:
    if isinstance(generator_output, dict):
        for key in ("generated_text", "text", "output", "answer", "prediction", "predictions"):
            if key in generator_output:
                return _flatten_generator_output(generator_output[key])
        return [generator_output]
    if isinstance(generator_output, (list, tuple)):
        flattened: list[Any] = []
        for item in generator_output:
            flattened.extend(_flatten_generator_output(item))
        return flattened
    if isinstance(generator_output, torch.Tensor):
        if generator_output.ndim == 0:
            return [generator_output.item()]
        return generator_output.detach().cpu().reshape(-1).tolist()
    return [generator_output]


def _extract_number(text: Any) -> int | None:
    match = re.search(r"-?\d+", _normalize_text(text))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _build_reward_function(args, expected_value: Any) -> Callable[[Any], torch.Tensor]:
    expected_target = 0 if args.expected_value == 0 else 1

    def _reward(generator_output: Any) -> torch.Tensor:
        outputs = _flatten_generator_output(generator_output)
        if not outputs:
            return torch.zeros(1, dtype=torch.float32)

        predicted_matches = 0
        for output in outputs:
            number = _extract_number(output)
            if number is not None:
                predicted_matches += int(number == expected_target)
                continue

            normalized = _normalize_text(output)
            if expected_target == 0:
                predicted_matches += int(normalized in {"zero", "0", "false", "no", "n"})
            else:
                predicted_matches += int(normalized in {"one", "1", "true", "yes", "y"})

        if args.atLeastL and args.atMostL:
            passed = predicted_matches >= args.expected_atLeastL and predicted_matches <= args.expected_atMostL
        elif args.atMostL:
            passed = predicted_matches <= args.expected_atMostL
        elif args.atLeastL:
            passed = predicted_matches >= args.expected_atLeastL
        else:
            passed = predicted_matches == args.expected_atLeastL

        return torch.tensor([1.0 if passed else 0.0], dtype=torch.float32)

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
