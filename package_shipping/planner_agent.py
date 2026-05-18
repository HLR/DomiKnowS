"""Offline mock planner agent for the package shipping demo."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from domiknows.generation.planning import PlanningBundle


@dataclass(frozen=True)
class PlanCandidate:
    """One proposed compact shipping plan."""

    actions: tuple[str, ...]
    source: str


class MockPackageShippingPlannerAgent:
    """Deterministic offline planner that proposes valid and invalid shipments."""

    def __init__(self, *, seed: int = 0):
        self._rng = random.Random(seed)

    def propose(self, bundle: PlanningBundle, *, count: int = 6) -> tuple[PlanCandidate, ...]:
        valid = bundle.selected_reference_plan
        missing_padding = (
            PlanCandidate(self._missing_padding(valid), "invalid_missing_padding")
            if "add_padding" in valid
            else PlanCandidate(self._missing_required(valid, bundle.selected_required_actions), "invalid_missing_required")
        )
        missing_label = (
            PlanCandidate(self._missing_return_label(valid), "invalid_missing_return_label")
            if "print_return_label" in valid
            else PlanCandidate(self._missing_shipping_label(valid), "invalid_missing_shipping_label")
        )
        proposals = [
            PlanCandidate(valid, "graph_reference_plan"),
            missing_padding,
            missing_label,
            PlanCandidate(self._drop_before_seal(valid), "invalid_drop_before_seal"),
            PlanCandidate(self._insert_before_box(valid), "invalid_insert_before_box"),
            PlanCandidate(self._seal_twice(valid), "invalid_seal_twice"),
            PlanCandidate(self._missing_required(valid, bundle.selected_required_actions), "invalid_missing_required"),
        ]
        if count <= len(proposals):
            return tuple(proposals[:count])
        extra = list(proposals)
        while len(extra) < count:
            shuffled = list(valid)
            middle = shuffled[:-1]
            self._rng.shuffle(middle)
            extra.append(PlanCandidate(tuple(middle + [bundle.terminal_action]), "invalid_shuffled"))
        return tuple(extra[:count])

    @staticmethod
    def _missing_padding(plan: Sequence[str]) -> tuple[str, ...]:
        return tuple(action for action in plan if action != "add_padding")

    @staticmethod
    def _missing_return_label(plan: Sequence[str]) -> tuple[str, ...]:
        return tuple(action for action in plan if action != "print_return_label")

    @staticmethod
    def _missing_shipping_label(plan: Sequence[str]) -> tuple[str, ...]:
        return tuple(action for action in plan if action != "print_label")

    @staticmethod
    def _drop_before_seal(plan: Sequence[str]) -> tuple[str, ...]:
        delivery = next((action for action in plan if action in {"drop_off", "request_pickup"}), None)
        if delivery is None:
            return tuple(plan)
        remaining = [action for action in plan if action != delivery]
        insert_at = max(1, remaining.index("seal_box")) if "seal_box" in remaining else 1
        return tuple(remaining[:insert_at] + [delivery] + remaining[insert_at:])

    @staticmethod
    def _insert_before_box(plan: Sequence[str]) -> tuple[str, ...]:
        if "insert_item" not in plan:
            return tuple(plan)
        remaining = [action for action in plan if action != "insert_item"]
        return tuple(["insert_item"] + remaining)

    @staticmethod
    def _seal_twice(plan: Sequence[str]) -> tuple[str, ...]:
        if "seal_box" not in plan:
            return tuple(plan)
        output = []
        for action in plan:
            output.append(action)
            if action == "seal_box":
                output.append("seal_box")
        return tuple(output)

    @staticmethod
    def _missing_required(plan: Sequence[str], required_actions: Sequence[str]) -> tuple[str, ...]:
        removable = next((action for action in required_actions if action in plan), None)
        if removable is None:
            return tuple(plan)
        return tuple(action for action in plan if action != removable)
