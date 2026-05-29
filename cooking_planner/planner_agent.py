"""Offline mock planner agent for the cooking planner demo."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from domiknows.generation.applications.planning import PlanningBundle


@dataclass(frozen=True)
class PlanCandidate:
    """One proposed compact action plan."""

    actions: tuple[str, ...]
    source: str


class MockCookingPlannerAgent:
    """Deterministic offline planner that proposes valid and invalid plans."""

    def __init__(self, *, seed: int = 0):
        self._rng = random.Random(seed)

    def propose(self, bundle: PlanningBundle, *, count: int = 6) -> tuple[PlanCandidate, ...]:
        """Return candidate plans for the selected dish."""

        valid = bundle.selected_reference_plan
        proposals = [
            PlanCandidate(valid, "graph_reference_plan"),
            PlanCandidate(self._missing_close(valid), "invalid_missing_close_fridge"),
            PlanCandidate(self._take_before_open(valid), "invalid_take_before_open"),
            PlanCandidate(self._too_many_fridge_opens(valid), "invalid_too_many_fridge_opens"),
            PlanCandidate(self._prep_before_table(valid), "invalid_prep_before_table"),
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
    def _missing_close(plan: Sequence[str]) -> tuple[str, ...]:
        return tuple(action for action in plan if action != "close_fridge")

    @staticmethod
    def _take_before_open(plan: Sequence[str]) -> tuple[str, ...]:
        takes = [action for action in plan if action.startswith("take_")]
        if not takes:
            return tuple(plan)
        remaining = [action for action in plan if action != takes[0]]
        return tuple([takes[0]] + remaining)

    @staticmethod
    def _too_many_fridge_opens(plan: Sequence[str]) -> tuple[str, ...]:
        return tuple(["open_fridge", "close_fridge", "open_fridge", "close_fridge"] + list(plan))

    @staticmethod
    def _prep_before_table(plan: Sequence[str]) -> tuple[str, ...]:
        prep_actions = {"mix_dough", "chop_lettuce", "bake_cookies", "cook_omelette"}
        prep = next((action for action in plan if action in prep_actions), None)
        if prep is None:
            return tuple(plan)
        remaining = [action for action in plan if action != prep]
        insert_at = 1 if len(remaining) > 1 else 0
        return tuple(remaining[:insert_at] + [prep] + remaining[insert_at:])

    @staticmethod
    def _missing_required(plan: Sequence[str], required_actions: Sequence[str]) -> tuple[str, ...]:
        removable = next((action for action in required_actions if action in plan), None)
        if removable is None:
            return tuple(plan)
        return tuple(action for action in plan if action != removable)
