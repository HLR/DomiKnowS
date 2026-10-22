"""EAI vocabulary and state contracts; declarations are expanded in world_graph.py.

Name normalization and compatibility observations are input grounding. Default
policies use primitive facts and the formulas in the world graph.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from ..world_graph import EAIWorldGraphBundle

Fact = tuple[str, ...]


ABSENT_ENTITY = "__absent_action_argument__"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    min_args: int
    max_args: int
    is_goal_action: bool = False
    requires_task_entity: bool = False


@dataclass(frozen=True)
class StatePredicateSpec:
    name: str
    arity: int
    aliases: tuple[str, ...] = ()
    positive_counterpart: str | None = None


@dataclass(frozen=True)
class ActionPreconditionSpec:
    name: str
    action: str
    kind: str


_ACTION_NAMES = frozenset(
    {
        "clean",
        "close",
        "cook",
        "drink",
        "drop",
        "find",
        "freeze",
        "grab",
        "left_grasp",
        "left_place_inside",
        "left_place_nextto",
        "left_place_nextto_ontop",
        "left_place_on_top",
        "left_place_ontop",
        "left_place_under",
        "left_release",
        "lie",
        "lookat",
        "open",
        "plugin",
        "plugout",
        "pointat",
        "pour",
        "pull",
        "push",
        "put",
        "putback",
        "putin",
        "putobjback",
        "putoff",
        "puton",
        "read",
        "right_grasp",
        "right_place_inside",
        "right_place_nextto",
        "right_place_nextto_ontop",
        "right_place_on_top",
        "right_place_ontop",
        "right_place_under",
        "right_realease",
        "right_release",
        "right_transfer_contents_inside",
        "rinse",
        "run",
        "scrub",
        "sit",
        "sleep",
        "slice",
        "soak",
        "squeeze",
        "standup",
        "switch_off",
        "switch_on",
        "switchoff",
        "switchon",
        "toggle_off",
        "toggle_on",
        "touch",
        "turn_off",
        "turn_on",
        "turnto",
        "type",
        "unfreeze",
        "walk",
        "wash",
        "watch",
        "wipe",
    }
)


_TASK_ENTITY_ACTION_NAMES = frozenset(
    {
        "clean",
        "close",
        "cook",
        "freeze",
        "left_place_inside",
        "left_place_nextto",
        "left_place_nextto_ontop",
        "left_place_nextto_on_top",
        "left_place_on_top",
        "left_place_ontop",
        "left_place_under",
        "open",
        "right_place_inside",
        "right_place_nextto",
        "right_place_nextto_ontop",
        "right_place_nextto_on_top",
        "right_place_on_top",
        "right_place_ontop",
        "right_place_under",
        "slice",
        "soak",
        "switch_off",
        "switch_on",
        "toggle_off",
        "toggle_on",
        "turn_off",
        "turn_on",
        "unfreeze",
    }
)


ACTION_GOAL_NAMES = frozenset(
    {
        "drink",
        "grab",
        "lookat",
        "pour",
        "push",
        "read",
        "rinse",
        "scrub",
        "sleep",
        "switchoff",
        "touch",
        "type",
        "wash",
        "watch",
        "wipe",
    }
)


_UNARY_STATE_NAMES = frozenset(
    {
        "clean",
        "closed",
        "cooked",
        "dusty",
        "frozen",
        "lying",
        "not_closed",
        "not_dusty",
        "not_frozen",
        "not_on",
        "not_open",
        "not_plugged_in",
        "not_stained",
        "off",
        "on",
        "open",
        "plugged_in",
        "released",
        "rinsed",
        "sitting",
        "sliced",
        "soaked",
        "stained",
        "touch",
        "washed",
    }
)


_BINARY_STATE_NAMES = frozenset(
    {
        "facing",
        "holds_lh",
        "holds_rh",
        "inside",
        "near",
        "nextto",
        "not_inside",
        "not_ontop",
        "ontop",
        "onfloor",
        "touching",
        "under",
    }
)


STATE_PREDICATES = _UNARY_STATE_NAMES | _BINARY_STATE_NAMES


PREDICATE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "obj_inside": "inside",
        "obj_next_to": "nextto",
        "obj_ontop": "ontop",
        "next_to": "nextto",
        "on_top": "ontop",
        "switch_on": "on",
        "switchon": "on",
        "toggle_on": "on",
        "turn_on": "on",
        "turnon": "on",
        "toggled_on": "on",
        "switch_off": "off",
        "toggle_off": "off",
        "turn_off": "off",
        "turnoff": "off",
        "toggled_off": "off",
        "toggledon": "on",
        "unfrozen": "not_frozen",
    }
)


NEGATIVE_TO_POSITIVE: Mapping[str, str] = MappingProxyType(
    {
        "not_closed": "closed",
        "not_dusty": "dusty",
        "not_frozen": "frozen",
        "not_inside": "inside",
        "not_on": "on",
        "not_open": "open",
        "not_ontop": "ontop",
        "not_plugged_in": "plugged_in",
        "not_stained": "stained",
    }
)


def _aliases_for(name: str) -> tuple[str, ...]:
    return tuple(
        sorted(alias for alias, canonical in PREDICATE_ALIASES.items() if canonical == name)
    )


ACTION_SPECS: Mapping[str, ActionSpec] = MappingProxyType(
    {
        name: ActionSpec(
            name=name,
            min_args=0 if name in {"sleep", "standup"} else 1,
            max_args=0 if name in {"sleep", "standup"} else 2,
            is_goal_action=name in ACTION_GOAL_NAMES,
            requires_task_entity=name in _TASK_ENTITY_ACTION_NAMES,
        )
        for name in sorted(_ACTION_NAMES)
    }
)


TASK_ENTITY_ACTION_NAMES = frozenset(
    name for name, spec in ACTION_SPECS.items() if spec.requires_task_entity
)


STATE_SPECS: Mapping[str, StatePredicateSpec] = MappingProxyType(
    {
        name: StatePredicateSpec(
            name=name,
            arity=1 if name in _UNARY_STATE_NAMES else 2,
            aliases=_aliases_for(name),
            positive_counterpart=NEGATIVE_TO_POSITIVE.get(name),
        )
        for name in sorted(STATE_PREDICATES)
    }
)


def canonical_state_name(name: str, bundle: EAIWorldGraphBundle | None = None) -> str:
    return (
        bundle.canonical_state_name(name)
        if bundle is not None
        else PREDICATE_ALIASES.get(name, name)
    )


def is_known_action(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    return bundle.is_action(name) if bundle is not None else name in ACTION_SPECS


def is_goal_action(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    return bundle.is_goal_action(name) if bundle is not None else name in ACTION_GOAL_NAMES


def is_state_predicate(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    if bundle is not None:
        return bundle.is_state_predicate(name)
    return canonical_state_name(name) in STATE_SPECS


def positive_state_name(name: str, bundle: EAIWorldGraphBundle | None = None) -> str | None:
    if bundle is not None:
        return bundle.positive_state_name(name)
    return NEGATIVE_TO_POSITIVE.get(canonical_state_name(name))


@dataclass(frozen=True)
class WorldConstraintEvaluation:
    score: float
    # Number of constraints that contributed to this trajectory's aggregate.
    constraint_count: int
    results: Mapping[str, Any]
    # Total number declared on the graph, including inactive constraints.
    declared_constraint_count: int
    grounding_report: Any = None


class WorldGraphConfigurationError(RuntimeError):
    """A declared world constraint could not be materialized or verified."""


def _action_side(action: str) -> str | None:
    if action.startswith("left_"):
        return "left"
    if action.startswith("right_"):
        return "right"
    return None


def _placement_relation_name(action: str) -> str | None:
    if "inside" in action or action in {"put", "putin", "put_inside"}:
        return "inside"
    if (
        "ontop" in action
        or "on_top" in action
        or action in {"putback", "puton", "putontop", "put_ontop"}
    ):
        return "ontop"
    if "nextto" in action or "next_to" in action:
        return "nextto"
    if "under" in action:
        return "under"
    return None


def _held_by_side(state: set[Fact]) -> dict[str, set[str]]:
    return {
        "left": {
            fact[2]
            for fact in state
            if len(fact) == 3 and fact[0] == "holds_lh" and fact[1] == "character"
        },
        "right": {
            fact[2]
            for fact in state
            if len(fact) == 3 and fact[0] == "holds_rh" and fact[1] == "character"
        },
    }
