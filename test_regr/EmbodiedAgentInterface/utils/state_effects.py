"""Declarative effects of EAI's abstract action model.

A result predicate names the truth of the bound fact after an action. Positive
literals establish facts; negative literals remove facts. Unmentioned facts
persist. Runtime binds entities and relation instances, then executes these
ordinary DomiKnowS implications; no action-name switch chooses the effects.
"""

from dataclasses import dataclass
from functools import lru_cache
from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import ifL, andL, orL, notL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    compile_predicate_constraints,
)

INPUTS = (
    "argument_supplied",
    "source_exists",
    "grasp_action",
    "placement_action",
    "release_action",
    "placement_nextto",
    "placement_ontop",
    "compound_nextto",
    "clean_action",
    "wash_action",
    "open_action",
    "close_action",
    "turn_on_action",
    "turn_off_action",
    "slice_action",
    "soak_action",
    "freeze_action",
    "unfreeze_action",
    "cook_action",
    "plugin_action",
    "plugout_action",
    "pour_action",
    "walk_action",
    "look_action",
    "touch_action",
    "sit_action",
    "lie_action",
)
RESULTS = (
    "selected_hand_contents",
    "held_argument",
    "negative_spatial_goal",
    "positive_spatial_goal",
    "previous_placement",
    "placement",
    "negative_placement",
    "touching",
    "nextto",
    "source_hand_contents",
    "released",
    "floor_goal",
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
    "poured_inside",
    "near_object",
    "near_character",
    "character_inside",
    "character_nextto",
    "facing",
    "touch",
    "sitting",
    "character_ontop",
    "lying",
)


@dataclass(frozen=True)
class EffectGraph:
    graph: object
    inputs: dict
    results: dict


def build_state_effect_graph():
    with Graph("eai_state_effects") as graph:
        transition = Concept(name="action_transition")
        inputs = {name: transition(name="observed__" + name) for name in INPUTS}
        results = {name: transition(name="result__" + name) for name in RESULTS}
        observed = lambda name: inputs[name]("transition")
        after = lambda name: results[name]("transition")
        grasp = andL(observed("grasp_action"), observed("argument_supplied"))
        place = andL(
            observed("placement_action"), observed("argument_supplied"), observed("source_exists")
        )
        release = andL(observed("release_action"), observed("source_exists"))

        # A grasp replaces this hand's contents and records requested negative spatial facts.
        ifL(
            grasp,
            andL(
                notL(after("selected_hand_contents")),
                after("held_argument"),
                after("negative_spatial_goal"),
                notL(after("positive_spatial_goal")),
            ),
            name="grasp_effects",
        )
        # Placement establishes its relation, clears its negation, and releases the source.
        ifL(
            place,
            andL(
                after("placement"),
                notL(after("negative_placement")),
                notL(after("source_hand_contents")),
            ),
            name="placement_effects",
        )
        ifL(
            andL(place, notL(observed("placement_nextto"))),
            notL(after("previous_placement")),
            name="placement_replaces_location",
        )
        ifL(
            andL(place, observed("placement_ontop")),
            after("touching"),
            name="placement_on_top_touches",
        )
        ifL(
            andL(place, observed("compound_nextto")),
            after("nextto"),
            name="compound_placement_nextto",
        )
        # This placeholder belongs to legacy abstract reward simulation only.
        ifL(
            andL(
                observed("placement_action"),
                observed("argument_supplied"),
                notL(observed("source_exists")),
            ),
            after("placement"),
            name="legacy_unbound_placement",
        )
        ifL(
            release,
            andL(after("released"), after("floor_goal"), notL(after("source_hand_contents"))),
            name="release_effects",
        )

        # Cleaning removes dirt; washing and rinsing additionally establish both outcomes.
        ifL(
            andL(observed("clean_action"), observed("argument_supplied")),
            andL(
                after("clean"),
                after("not_dusty"),
                after("not_stained"),
                notL(after("dusty")),
                notL(after("stained")),
            ),
            name="cleaning_effects",
        )
        ifL(
            andL(observed("wash_action"), observed("argument_supplied")),
            andL(after("washed"), after("rinsed")),
            name="washing_effects",
        )
        ifL(
            andL(observed("open_action"), observed("argument_supplied")),
            andL(
                after("open"), after("not_closed"), notL(after("closed")), notL(after("not_open"))
            ),
            name="opening_effects",
        )
        ifL(
            andL(observed("close_action"), observed("argument_supplied")),
            andL(
                after("closed"), after("not_open"), notL(after("open")), notL(after("not_closed"))
            ),
            name="closing_effects",
        )
        ifL(
            andL(observed("turn_on_action"), observed("argument_supplied")),
            andL(after("on"), notL(after("off")), notL(after("not_on"))),
            name="switch_on_effects",
        )
        ifL(
            andL(observed("turn_off_action"), observed("argument_supplied")),
            andL(after("off"), after("not_on"), notL(after("on"))),
            name="switch_off_effects",
        )
        # Single-state effects share the same argument guard.
        for action, result in (
            ("slice", "sliced"),
            ("soak", "soaked"),
            ("cook", "cooked"),
            ("touch", "touch"),
        ):
            ifL(
                andL(observed(action + "_action"), observed("argument_supplied")),
                after(result),
                name=action + "_effects",
            )
        ifL(
            andL(observed("freeze_action"), observed("argument_supplied")),
            andL(after("frozen"), notL(after("not_frozen"))),
            name="freezing_effects",
        )
        ifL(
            andL(observed("unfreeze_action"), observed("argument_supplied")),
            andL(after("not_frozen"), notL(after("frozen"))),
            name="unfreezing_effects",
        )
        ifL(
            andL(observed("plugin_action"), observed("argument_supplied")),
            andL(after("plugged_in"), notL(after("not_plugged_in"))),
            name="plugging_in_effects",
        )
        ifL(
            andL(observed("plugout_action"), observed("argument_supplied")),
            andL(after("not_plugged_in"), notL(after("plugged_in"))),
            name="unplugging_effects",
        )
        ifL(
            andL(observed("pour_action"), observed("argument_supplied"), observed("source_exists")),
            after("poured_inside"),
            name="pouring_effects",
        )
        ifL(
            andL(observed("walk_action"), observed("argument_supplied")),
            andL(
                after("near_object"),
                after("near_character"),
                after("character_inside"),
                after("character_nextto"),
            ),
            name="navigation_effects",
        )
        ifL(
            andL(observed("look_action"), observed("argument_supplied")),
            after("facing"),
            name="looking_effects",
        )
        ifL(
            andL(observed("sit_action"), observed("argument_supplied")),
            andL(after("sitting"), after("character_ontop")),
            name="sitting_effects",
        )
        ifL(
            andL(observed("lie_action"), observed("argument_supplied")),
            andL(after("lying"), after("character_ontop")),
            name="lying_effects",
        )
    return EffectGraph(graph, inputs, results)


def compile_state_effects(bundle):
    bindings = [
        PredicateBinding(concept, "input:" + name, "transition")
        for name, concept in bundle.inputs.items()
    ]
    bindings.extend(
        PredicateBinding(concept, "result:" + name, "transition")
        for name, concept in bundle.results.items()
    )
    return compile_predicate_constraints(bundle.graph, bindings)


@lru_cache(maxsize=1)
def state_effect_program():
    return compile_state_effects(build_state_effect_graph())
