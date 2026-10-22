"""Primitive EAI observations; logical combinations belong to world_graph.py."""

from .world_domain import _action_side, _held_by_side, _placement_relation_name

PRECONDITION_FACTS = (
    "task_entities_supplied",
    "first_argument_supplied",
    "first_argument_in_task",
    "second_argument_supplied",
    "second_argument_in_task",
    "source_state_known",
    "selected_hand_occupied",
    "left_hand_occupied",
    "right_hand_occupied",
    "matching_placement_exists",
    "matching_release_relation_exists",
    "destination_supplied",
    "destination_observed_open",
    "destination_observed_not_closed",
    "destination_observed_closed",
    "destination_observed_not_open",
)


def precondition_observations(
    action,
    args,
    source,
    task_entity_types=(),
    *,
    source_known=True,
    selected_hand=None,
    negative=(),
):
    """Observe membership, hand contents, relation matches and signed access facts."""
    from test_regr.EmbodiedAgentInterface.dataset import entity_type_for_token

    held = _held_by_side(source)
    side = _action_side(action) if selected_hand is None else selected_hand
    occupied = held[side] if side is not None else held["left"] | held["right"]
    available = frozenset(task_entity_types)
    first = args[0] if args else None
    second = args[1] if len(args) > 1 else None
    relation = _placement_relation_name(action)
    signed = set(source) | {("not_" + fact[0], *fact[1:]) for fact in negative}
    return {
        "task_entities_supplied": bool(available),
        "first_argument_supplied": first is not None,
        "first_argument_in_task": first is not None and entity_type_for_token(first) in available,
        "second_argument_supplied": second is not None,
        "second_argument_in_task": second is not None
        and entity_type_for_token(second) in available,
        "source_state_known": bool(source_known),
        "selected_hand_occupied": bool(occupied),
        "left_hand_occupied": bool(held["left"]),
        "right_hand_occupied": bool(held["right"]),
        "matching_placement_exists": bool(
            relation
            and first
            and any(len(fact) == 3 and fact[0] == relation and fact[2] == first for fact in source)
        ),
        "matching_release_relation_exists": bool(
            first
            and any(
                len(fact) == 3
                and fact[0] in {"inside", "nextto", "ontop", "onfloor", "under"}
                and fact[1] == first
                for fact in source
            )
        ),
        "destination_supplied": first is not None,
        "destination_observed_open": ("open", first) in signed,
        "destination_observed_not_closed": ("not_closed", first) in signed,
        "destination_observed_closed": ("closed", first) in signed,
        "destination_observed_not_open": ("not_open", first) in signed,
    }
