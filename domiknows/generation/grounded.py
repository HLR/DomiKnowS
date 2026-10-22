"""Readable DomiKnowS policies for explicitly grounded planning requirements.

Each capability, precondition, compatible destination, goal and invariant is a
requirement observation. The graph enforces: a known required fact must hold.
Unknown observations stay unknown and are disclosed in grounding reports.

Parsing, numeric measurements/comparisons and symbolic state updates live in
``grounded_facts.py``. They supply evidence; the LCs here decide enforcement.
The established imports remain available for EAI and other adapters.
"""
from .grounded_facts import (
    FAMILIES, FactState, GroundedState, Operation, RequirementObservation,
    Check, SelectionRequirement, PlanningFacts, observations, report, combined,
)

FAMILY_MEANINGS = {
    "destination": "use the explicitly requested destination",
    "holding": "hold the required subject before manipulating it",
    "capacity": "respect the observed hand capacity",
    "affordance": "satisfy supplied capabilities, preconditions and measured selection criteria",
    "compatibility": "use a compatible subject/destination pair and container",
    "milestone_order": "respect declared predecessor progress",
    "goal_completion": "establish every supplied goal and required milestone",
    "preservation": "preserve protected entities and supplied invariants",
}

REQUIREMENT_INPUTS = (
    "idempotent", "hand_known", "held_exists", "subject_unspecified", "held_matches_subject",
    "capacity_known", "capacity_zero", "capacity_positive",
)

# Graph declarations ---------------------------------------------------------

class GroundedConcepts(dict):
    """Legacy observations plus optional per-requirement graph declarations."""

    constraints = ()
    requirement = None
    known = None
    holds = None
    families = None
    inputs = None


def declare_predicates(operation, episode, *, active=True, constraints=None, detailed=False):
    """Declare the eight grounding policies as ordinary DomiKnowS formulas.

    In detailed mode, each supplied requirement is a graph node. For example:

        affordance_requirement(r) AND observation_known(r) -> fact_holds(r)

    Universal grounding over those nodes expresses that every known required
    fact must hold. Unknown observations are not asserted true. Numeric
    comparisons and effect application supply observations, not policy verdicts.

    Legacy operation/episode Boolean concepts stay available to older adapters
    and custom constraints. They are not the enforcement source in detailed
    mode. ``constraints`` separates rule existence from tier activation.
    """
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, ifL, notL, orL
    if constraints is None:
        constraints = active  # preserve the older adapter API
    concepts = GroundedConcepts()
    for family in FAMILIES:
        parent = episode if family == "goal_completion" else operation
        for suffix in ("applicable", "satisfied"):
            key = family + "__" + suffix
            concepts[key] = parent(name=parent.name + "__grounded__" + key)

    if detailed:
        requirement = Concept(name=operation.name + "__grounded_requirement")
        operation.contains(requirement)
        episode.contains(requirement)
        concepts.requirement = requirement
        concepts.known = requirement(name="grounded__observation_known")
        concepts.holds = requirement(name="grounded__fact_holds")
        concepts.families = {
            family: requirement(name="grounded__requires__" + family) for family in FAMILIES
        }
        concepts.inputs = {name: requirement(name="grounded__observed__" + name) for name in REQUIREMENT_INPUTS}
    rules = []
    if constraints:
        for family in FAMILIES:
            if detailed:
                # Each capability, precondition, goal, invariant, etc. is a
                # separate requirement. No aggregate satisfied flag is read.
                fact = lambda name: concepts.inputs[name]("requirement")
                known, conclusion = concepts.known("requirement"), concepts.holds("requirement")
                if family == "holding":
                    # Idempotent actions need no new grasp. Otherwise the hand
                    # must hold an object matching the supplied subject.
                    known = orL(fact("idempotent"), fact("hand_known"))
                    conclusion = orL(fact("idempotent"), andL(
                        fact("held_exists"), orL(fact("subject_unspecified"), fact("held_matches_subject"))))
                elif family == "capacity":
                    # A zero-capacity hand is unusable even if its contents are
                    # unknown; otherwise known occupancy must fit this grasp.
                    known = andL(fact("capacity_known"), orL(fact("capacity_zero"), fact("hand_known")))
                    conflict = andL(fact("hand_known"), fact("held_exists"), notL(fact("held_matches_subject")))
                    conclusion = andL(fact("capacity_positive"), notL(conflict))
                rule = ifL(andL(concepts.families[family]("requirement"), known), conclusion,
                           name="grounded__" + family, active=active)
            else:
                parent = episode if family == "goal_completion" else operation
                rule = ifL(
                    andL(parent("x"), concepts[family + "__applicable"]("x")),
                    concepts[family + "__satisfied"]("x"), name="grounded__" + family, active=active,
                )
            rules.append(rule)
    concepts.constraints = tuple(rules)
    return concepts


def requirement_observations(checks):
    """Yield primitive valuations for independent requirement nodes/events.

    Applicability is supplied by the explicit operation/request facts. An
    unknown requirement gets known=False and holds=False; the LC guards it
    with known, while the grounding report retains the missing observation.
    """
    for check in checks:
        if check.family not in FAMILIES or not check.applicable:
            continue
        requirements = check.requirements or (RequirementObservation(check.family, check.satisfied),)
        for requirement in requirements:
            values = {"requirement": True, "requirement_known": requirement.truth is not None,
                      "requirement_holds": requirement.truth is True}
            values.update({"requirement_family:" + family: family == check.family for family in FAMILIES})
            inputs = dict(requirement.inputs)
            values.update({"requirement_input:" + name: bool(inputs.get(name, False)) for name in REQUIREMENT_INPUTS})
            yield check.family, requirement.name, values


def requirement_events(concepts, checks):
    """Bind the same requirement facts used by DataNode materialization."""
    from .dfa.predicate_monitor import PredicateEvent
    if getattr(concepts, "requirement", None) is not None:
        for family, _name, values in requirement_observations(checks):
            yield PredicateEvent("grounded_requirement", values,
                                 "completion" if family == "goal_completion" else "immediate")


def predicate_bindings(concepts):
    """Register observations by concept identity, without interpreting LC names."""
    from .dfa.predicate_monitor import PredicateBinding
    bindings = [PredicateBinding(concept, key, "episode" if key.startswith("goal_completion__") else "operation",
                                 "completion" if key.startswith("goal_completion__") else "immediate")
                for key, concept in concepts.items()]
    if getattr(concepts, "requirement", None) is not None:
        bindings.extend((
            PredicateBinding(concepts.requirement, "requirement", "grounded_requirement"),
            PredicateBinding(concepts.known, "requirement_known", "grounded_requirement"),
            PredicateBinding(concepts.holds, "requirement_holds", "grounded_requirement"),
        ))
        bindings.extend(PredicateBinding(concept, "requirement_family:" + family, "grounded_requirement",
                                         "completion" if family == "goal_completion" else "immediate")
                        for family, concept in concepts.families.items())
        bindings.extend(PredicateBinding(concept, "requirement_input:" + name, "grounded_requirement")
                        for name, concept in concepts.inputs.items())
    return bindings
