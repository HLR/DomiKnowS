"""Bind abstract effects to entity/fact identities and apply declared writes.

Entity matching, selecting the legacy hand/source, and enumerating matching
relation tuples are grounding. The DomiKnowS formulas choose which writes run.
"""

from .state_effects import INPUTS, RESULTS, state_effect_program
from domiknows.generation.dfa.predicate_monitor import _evaluate


def effect_observations(event, state, held, goal_facts):
    from ..reward import _side_for_action, _placement_relation, _entities_match

    action = event.name
    obj = event.args[0] if event.args else None
    side = _side_for_action(action)
    relation = _placement_relation(action)
    grasp = any(word in action for word in ("grasp", "grab", "pickup", "take"))
    release = any(word in action for word in ("release", "realease", "drop", "putobjback"))
    hand = side or ("right" if held["right"] is None else "left")
    source = None
    if relation:
        source = next(
            (
                value
                for value in held.values()
                if value and action == "puton" and obj and _entities_match(value, obj)
            ),
            None,
        )
        source = (
            source or (held.get(side) if side else None) or held.get("right") or held.get("left")
        )
    elif release:
        source = (
            (
                held.get(side)
                if side
                else next((value for value in held.values() if obj and value == obj), None)
            )
            or held.get("right")
            or held.get("left")
        )
    elif action == "pour":
        goal_source = next(
            (
                fact[1]
                for fact in goal_facts
                if len(fact) == 3 and fact[0] == "inside" and obj and _entities_match(fact[2], obj)
            ),
            None,
        )
        source = next(
            (
                value
                for value in held.values()
                if value and goal_source and _entities_match(goal_source, value)
            ),
            None,
        )
        source = source or held.get("right") or held.get("left")
    destination = (
        "character"
        if action == "puton" and source and obj and _entities_match(source, obj)
        else obj
    )
    effective_relation = (
        "onfloor"
        if relation == "ontop" and destination and destination.startswith("room_floor_")
        else relation
    )
    observations = {name: False for name in INPUTS}
    observations.update(
        argument_supplied=obj is not None,
        source_exists=source is not None,
        grasp_action=grasp,
        placement_action=relation is not None,
        release_action=release,
        placement_nextto=relation == "nextto",
        placement_ontop=relation == "ontop",
        compound_nextto="nextto_ontop" in action,
        clean_action=action in {"clean", "wipe", "scrub", "wash", "rinse"},
        wash_action=action in {"wash", "rinse"},
        turn_on_action=action in {"toggle_on", "switchon", "switch_on", "turn_on", "turnon"},
        turn_off_action=action in {"toggle_off", "switchoff", "switch_off", "turn_off", "turnoff"},
        walk_action=action in {"walk", "run"},
        look_action=action in {"turnto", "lookat", "watch"},
    )
    for name in (
        "open",
        "close",
        "slice",
        "soak",
        "freeze",
        "unfreeze",
        "cook",
        "plugin",
        "plugout",
        "pour",
        "touch",
        "sit",
        "lie",
    ):
        observations[name + "_action"] = action == name
    # Result bindings identify fact tuples. Their truth is chosen by the formulas.
    writes = {name: set() for name in RESULTS}
    for name in (
        "clean",
        "not_dusty",
        "not_stained",
        "dusty",
        "stained",
        "washed",
        "rinsed",
        "open",
        "closed",
        "not_open",
        "not_closed",
        "on",
        "off",
        "not_on",
        "sliced",
        "soaked",
        "frozen",
        "not_frozen",
        "cooked",
        "plugged_in",
        "not_plugged_in",
        "touch",
    ):
        writes[name] = {(name, obj)}
    predicate = "holds_lh" if hand == "left" else "holds_rh"
    writes["selected_hand_contents"] = {
        fact for fact in state if len(fact) == 3 and fact[0] == predicate and fact[1] == "character"
    }
    writes["held_argument"] = {(predicate, "character", obj)}
    # Bind previous contents to different tuples: re-grasping the same object
    # does not require its final hold to be both true and false.
    writes["selected_hand_contents"] -= writes["held_argument"]
    writes["negative_spatial_goal"] = {
        fact
        for fact in goal_facts
        if fact
        and fact[0] in {"not_inside", "not_nextto", "not_ontop", "not_under"}
        and len(fact) >= 2
        and obj
        and _entities_match(fact[1], obj)
    }
    writes["positive_spatial_goal"] = {
        (fact[0][4:], *fact[1:]) for fact in writes["negative_spatial_goal"]
    }
    writes["previous_placement"] = {
        fact
        for fact in state
        if len(fact) >= 2
        and fact[0] in {"inside", "onfloor", "ontop", "under"}
        and source
        and _entities_match(fact[1], source)
    }
    writes["placement"] = {
        (
            effective_relation if source else relation,
            source or "object",
            destination if source else obj,
        )
    }
    writes["previous_placement"] -= writes["placement"]
    writes["negative_placement"] = (
        {("not_" + effective_relation, source, destination)} if effective_relation else set()
    )
    writes["touching"] = {("touching", source, destination)}
    writes["nextto"] = {("nextto", source, destination)}
    writes["source_hand_contents"] = {
        ("holds_lh" if hand_name == "left" else "holds_rh", "character", source)
        for hand_name in ("left", "right")
        if source and held.get(hand_name) == source
    }
    writes["released"] = {("released", source)}
    writes["floor_goal"] = {
        fact
        for fact in goal_facts
        if len(fact) == 3 and fact[0] == "onfloor" and source and _entities_match(fact[1], source)
    }
    writes.update(
        poured_inside={("inside", source, obj)},
        near_object={("near", obj)},
        near_character={("near", "character", obj)},
        character_inside={("inside", "character", obj)},
        character_nextto={("nextto", "character", obj)},
        facing={("facing", "character", obj)},
        sitting={("sitting", "character")},
        lying={("lying", "character")},
        character_ontop={("ontop", "character", obj)},
    )
    return {"input:" + key: value for key, value in observations.items()}, writes


def apply_event_effects(event, state, held, goal_facts=(), *, program=None):
    """Execute implication consequences generically; rule names are not read."""
    values, bindings = effect_observations(event, state, held, goal_facts)
    additions, removals = set(), set()

    def assign(expression, truth=True):
        operator, *args = expression
        if operator == "andL" and truth:
            for child in args:
                assign(child)
        elif operator == "notL":
            assign(args[0], not truth)
        elif operator == "predicate" and args[0].startswith("result:"):
            (additions if truth else removals).update(bindings[args[0][len("result:") :]])
        else:
            raise ValueError(f"Unsupported effect consequence: {expression}")

    for rule in (program or state_effect_program()).rules:
        if rule.expression[0] != "ifL":
            raise ValueError("Effect rules must be implications")
        if _evaluate(rule.expression[1], values):
            assign(rule.expression[2])
    if additions & removals:
        raise ValueError("Declared effects assign conflicting truth to the same fact")
    result = (set(state) - removals) | additions
    for hand_name in ("left", "right"):
        predicate = "holds_lh" if hand_name == "left" else "holds_rh"
        held[hand_name] = next(
            (fact[2] for fact in result if len(fact) == 3 and fact[:2] == (predicate, "character")),
            None,
        )
    return result
